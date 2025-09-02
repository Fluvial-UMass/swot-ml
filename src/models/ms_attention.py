import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from data import Batch
from .base_model import BaseModel
from .layers.transformer import CrossAttnDecoder


class MS_ATTN(BaseModel):
    """Model that uses attention to mix multiple data sources.

    Attributes
    ----------
    active_source: dict
        Booleans indicating which data sources are being used.
    static_embedder: StaticEmbedder
        Embedder for static data.
    decoders: dict
        Decoders, one for each cross-attention or self-attention.
    head: eqx.nn.Linear
        Linear layer that maps the output of the decoders to the target
        variable(s).
    target: list
        Names of the target variables.
    """

    source_var: str
    cross_vars: list[str]
    active_source: dict[str:bool]
    static_embedder: eqx.nn.Linear
    dynamic_embedders: dict[str : eqx.nn.Linear]
    cross_attn: dict[str:CrossAttnDecoder]

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        dynamic_sizes: dict,
        static_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        seed: int,
        dropout: float,
        active_source: dict,
    ):
        """Initializes an LSTM_MLP_ATTN model.

        Parameters
        ----------
        target: list
            The names of the target variables.
        seq_length: int
            The length of the input sequence.
        dynamic_sizes: dict
            A dictionary of the sizes of the dynamic features.
        static_size: int
            The number of static features.
        hidden_size: int
            The size of the hidden state.
        num_layers: int
            The number of layers in each attention head.
        num_heads: int
            The number of heads in each attention block.
        seed: int
            A seed for the random number generator.
        dropout: float
            The dropout rate.
        active_source: dict, optional
            A dicitonary of booleans indicating whether each dyanmic source will be used.
            Defaults to using all sources.
        """
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, 4)

        # Default all sources to true. Can be specified in the model args.
        if len(active_source) == 0:
            for source in dynamic_sizes.keys():
                active_source[source] = True
        self.active_source = active_source

        # Encoder for static data if used.
        use_static = static_size > 0
        if use_static:
            self.static_embedder = eqx.nn.Linear(static_size, hidden_size, key=keys[1])
        else:
            self.static_embedder = None

        # The first dynamic group is used as the 'source var' which is used as the key variable in
        # each cross attention block.
        self.source_var = list(dynamic_sizes.keys())[0]
        self.cross_vars = list(dynamic_sizes.keys())[1:]

        # Create the dynamic input projections
        self.dynamic_embedders = {}
        for var_name, var_size in dynamic_sizes.items():
            proj = eqx.nn.Linear(var_size, hidden_size, key=keys[1])
            self.dynamic_embedders[var_name] = proj

        # Helper fn for making attention blocks with common args
        def make_attn_block(sz, ea, k):
            return CrossAttnDecoder(
                seq_length, hidden_size, sz, hidden_size, num_layers, num_heads, dropout, ea, k
            )

        # Cross attention blocks for each irregular dynamic data source.
        self.cross_attn = {}
        if self.cross_vars:
            cross_keys = jrandom.split(keys[2], len(self.cross_vars))
            for var_name, var_key in zip(self.cross_vars, cross_keys):
                self.cross_attn[var_name] = make_attn_block(hidden_size, use_static, var_key)
        else:
            self.cross_attn["self"] = make_attn_block(hidden_size, use_static, keys[2])

        # # Pooler is another attn block after concatenating the cross attn outputs.
        # pool_size = hidden_size * len(self.cross_attn)
        # self.pooler = make_attn_block(pool_size, False, keys[3])

        super().__init__(hidden_size * len(self.cross_attn), target, key=keys[0])

    def finetune_update(self, *, active_source: dict):
        """Updates the model configuration after initialization.

        These updates must not break the forward call of the model. Only some
        things can reasonably change to ensure it does not break.

        Parameters
        ----------
        active_source: dict
            Boolean indicating if this data source (and encoder) are to be used.
        """
        for source, active in active_source.items():
            if source in self.active_source:
                self.active_source[source] = active
            else:
                raise ValueError(f"Source '{source}' not found in active_source.")
        print(self.active_source)

    def __call__(self, data: Batch, key: PRNGKeyArray):
        """The forward pass of the data through the model

        Parameters
        ----------
        data: dict[str, Array | dict[str, Array]]
            The input data.
        key: PRNGKeyArray
            A PRNG key used to apply the model.

        Returns
        -------
        Array
            The output of the model.
        """
        keys = jrandom.split(key)

        # Static embedding
        static_bias = self.static_embedder(data.static) if self.static_embedder else None

        # Embed / project the input dimensions into
        data_emb = {}
        masks = {}
        for var_name, embedder in self.dynamic_embedders.items():
            x = data.dynamic[var_name]
            mask = ~jnp.any(jnp.isnan(x), axis=1)
            x_filled = jnp.where(jnp.expand_dims(mask, 1), x, 0.0)

            data_emb[var_name] = jax.vmap(embedder)(x_filled)
            masks[var_name] = mask

        q = data_emb[self.source_var]  # Query

        # Use cross-attention with multiple sources
        if self.cross_vars:
            cross_keys = jrandom.split(keys[0], len(self.cross_vars))
            cross_list = []
            for var_name, var_key in zip(self.cross_vars, cross_keys):
                kv = data_emb[var_name] if self.active_source[var_name] else q
                var_cross = self.cross_attn[var_name](q, kv, static_bias, masks[var_name], var_key)
                cross_list.append(var_cross)

            cross_output = jnp.concat(cross_list, axis=0)
        # Use self-attention for a single source
        else:
            mask = ~jnp.any(jnp.isnan(q), axis=1)
            cross_output = self.cross_attn["self"](
                q, q, static_bias, masks[self.source_var], keys[0]
            )

        # print(cross_output.shape) # (hidden_size * len(self.cross_attn),)
        # pooled = self.pooler(cross_output, cross_output, None, masks[self.source_var], keys[1])

        return self.head(cross_output)
