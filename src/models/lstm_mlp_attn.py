import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import Batch
from .base_model import BaseModel
from .layers.static_mlp import StaticMLP
from .layers.ealstm import EALSTM
from .layers.transformer import CrossAttnDecoder


class LSTM_MLP_ATTN(BaseModel):
    """Model that uses LSTMs, MLPs, and attention to mix time frequencies.

    Attributes
    ----------
    active_source: dict
        Boolean indicating if this data source (and encoder) are to be used.
    encoders: dict
        Encoders, one for each dynamic data source.
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

    active_source: dict[str:bool]
    encoders: dict[str : eqx.Module]
    static_embedder: eqx.nn.Linear
    decoders: dict[str : eqx.Module]

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
        time_aware: dict,
        active_source: dict = {},
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
            The number of layers in the LSTM and MLP.
        num_heads: int
            The number of heads in the attention.
        seed: int
            A seed for the random number generator.
        dropout: float
            The dropout rate.
        time_aware: dict
            A dictionary of booleans indicating whether each dynamic feature is
            time-aware.
        active_source: dict, optional
            A dicitonary of booleans indicating whether each dyanmic source
            (collection of features), and the accompanying encoder will be used.
            Defaults to using all sources.
        """
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, 4)

        super().__init__(hidden_size, target, key=keys[0])

        # Encoder for static data if used.
        entity_aware = static_size > 0
        if entity_aware:
            self.static_embedder = eqx.nn.Linear(static_size, hidden_size, key=keys[1])
            static_size = hidden_size
        else:
            self.static_embedder = None
            static_size = 0

        # Default all sources to true. Can be specified in the model args.
        if len(active_source) == 0:
            for source in dynamic_sizes.keys():
                active_source[source] = True
        self.active_source = active_source

        # Encoders for each dynamic data source.
        encoder_keys = jrandom.split(keys[2], len(dynamic_sizes))
        self.encoders = {}
        for (var_name, var_size), var_key in zip(dynamic_sizes.items(), encoder_keys):
            if time_aware[var_name]:
                encoder = StaticMLP(
                    dynamic_in_size=var_size,
                    static_in_size=static_size,
                    out_size=hidden_size,
                    width_size=hidden_size * 2,
                    depth=num_layers,
                    key=var_key,
                )
            else:
                encoder = EALSTM(
                    dynamic_in_size=var_size,
                    static_in_size=static_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    return_all=True,
                    key=var_key,
                )
            self.encoders[var_name] = encoder

        # Set up each cross-attention decoder
        def make_decoder(k):
            return CrossAttnDecoder(
                seq_length,
                hidden_size,
                hidden_size,
                hidden_size,
                num_layers,
                num_heads,
                dropout,
                entity_aware,
                k,
            )

        self.decoders = {}
        cross_vars = list(dynamic_sizes.keys())[1:]

        if cross_vars:
            decoder_keys = jrandom.split(keys[3], len(cross_vars))
            for var_name, var_key in zip(cross_vars, decoder_keys):
                self.decoders[var_name] = make_decoder(var_key)
        else:
            self.decoders["self"] = make_decoder(keys[3])

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
        data: Batch
            The input data.
        key: PRNGKeyArray
            A PRNG key used to apply the model.

        Returns
        -------
        Array
            The output of the model.
        """
        keys = jrandom.split(key, 3)

        # Static embedding
        static_bias = self.static_embedder(data.static) if self.static_embedder else None

        # Encoders
        encoder_keys = jrandom.split(keys[0], len(self.encoders))
        encoded_data = {}
        masks = {}
        for (var_name, encoder), e_key in zip(self.encoders.items(), encoder_keys):
            if not self.active_source[var_name]:
                encoded_data[var_name] = None
                continue

            masks[var_name] = ~jnp.any(jnp.isnan(data.dynamic[var_name]), axis=1)
            x_d = jnp.where(jnp.expand_dims(masks[var_name], 1), data.dynamic[var_name], 0.0)
            encoded_data[var_name] = encoder(x_d, static_bias, e_key)

        # Decoders
        source_var = list(encoded_data.keys())[0]
        cross_vars = list(encoded_data.keys())[1:]
        query = encoded_data[source_var]

        if len(cross_vars) > 0:
            # Use cross-attention with multiple sources
            decoder_keys = jrandom.split(keys[2], len(cross_vars))
            decoded_list = []
            for k, d_key in zip(cross_vars, decoder_keys):
                if self.active_source[k]:
                    decoded = self.decoders[k](query, encoded_data[k], static_bias, masks[k], d_key)
                else:
                    decoded = self.decoders[k](query, query, static_bias, masks[source_var], d_key)
                decoded_list.append(decoded)

            pooled_output = jnp.concat(decoded_list, axis=0)

        else:
            # Use self-attention for a single source
            pooled_output = self.decoders["self"](
                query, query, static_bias, masks[source_var], keys[-1]
            )

        return self.head(pooled_output)
