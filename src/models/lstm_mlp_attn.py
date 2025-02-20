import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from models.lstm import EALSTM
from models.transformer import StaticEmbedder, CrossAttnDecoder


class StackedMLP(eqx.Module):
    """
    Wrapper for plain MLP that appends static data to time series data. 
    Applies the same weights to each time step.
    """
    append_static: bool
    mlp: eqx.nn.MLP

    def __init__(self, dynamic_in_size: int, static_in_size: int, out_size: int, width_size: int, depth: int, *,
                 key: PRNGKeyArray):
        self.append_static = static_in_size > 0
        self.mlp = eqx.nn.MLP(in_size=dynamic_in_size + static_in_size,
                              out_size=out_size,
                              width_size=width_size,
                              depth=depth,
                              key=key)

    def __call__(self, x_d: Array, x_s: Array, key: PRNGKeyArray):
        # vmap function that optionally adds static data
        def mlp_apply(x):
            input = jnp.concatenate([x, x_s], axis=-1) if self.append_static else x
            return self.mlp(input, key=key)

        # Use vmap to apply mlp_apply to each row in x_d
        return jax.vmap(mlp_apply)(x_d)


class LSTM_MLP_ATTN(eqx.Module):
    encoders: dict
    static_embedder: StaticEmbedder
    decoders: dict
    head: eqx.nn.Linear
    target: list

    def __init__(self,
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
                 time_aware: dict = {}):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 5)

        # Encoder for static data if used.
        entity_aware = static_size > 0
        if entity_aware:
            self.static_embedder = StaticEmbedder(static_size, hidden_size, dropout, keys[1])
            static_size = hidden_size
        else:
            self.static_embedder = None
            static_size = 0

        # Encoders for each dynamic data source.
        encoder_keys = jax.random.split(keys[0], len(dynamic_sizes))
        self.encoders = {}
        for (var_name, var_size), var_key in zip(dynamic_sizes.items(), encoder_keys):
            if time_aware[var_name]:
                encoder = StackedMLP(dynamic_in_size=var_size,
                                     static_in_size=static_size,
                                     out_size=hidden_size,
                                     width_size=hidden_size * 2,
                                     depth=num_layers,
                                     key=var_key)
            else:
                encoder = EALSTM(dynamic_in_size=var_size,
                                 static_in_size=static_size,
                                 hidden_size=hidden_size,
                                 dense_size=None,
                                 dropout=dropout,
                                 return_all=True,
                                 key=var_key)
            self.encoders[var_name] = encoder

        # Cross-attn or Self-attn decoders.
        self.decoders = {}
        cross_vars = list(dynamic_sizes.keys())[1:]
        decoder_keys = jax.random.split(keys[2], len(cross_vars))
        # Set up each cross-attention decoder
        if len(cross_vars) > 0:
            for var_name, var_key in zip(cross_vars, decoder_keys):
                self.decoders[var_name] = CrossAttnDecoder(seq_length, hidden_size, hidden_size, num_layers, num_heads,
                                                           dropout, entity_aware, var_key)
        else:
            self.decoders['self'] = CrossAttnDecoder(seq_length, hidden_size, hidden_size, num_layers, num_heads,
                                                     dropout, entity_aware, var_key)

        self.head = eqx.nn.Linear(in_features=hidden_size * len(self.decoders), out_features=len(target), key=keys[3])
        self.target = target

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        keys = jax.random.split(key, 3)

        # Static embedding
        if self.static_embedder:
            static_bias = self.static_embedder(data['static'], keys[1])
        else:
            static_bias = None

        # Encoders
        encoder_keys = jax.random.split(keys[0], len(self.encoders))
        encoded_data = {}
        masks = {}
        for (var_name, encoder), e_key in zip(self.encoders.items(), encoder_keys):
            masks[var_name] = ~jnp.any(jnp.isnan(data['dynamic'][var_name]), axis=1)
            x_d = jnp.where(jnp.expand_dims(masks[var_name], 1), data['dynamic'][var_name], 0.0)
            encoded_data[var_name] = encoder(x_d, static_bias, e_key)

        # Decoders
        source_var = list(encoded_data.keys())[0]
        cross_vars = list(encoded_data.keys())[1:]
        query = encoded_data[source_var]

        if len(cross_vars) > 0:
            # Use cross-attention with multiple sources
            decoder_keys = jax.random.split(keys[2], len(cross_vars))
            decoded_list = []
            for k, d_key in zip(cross_vars, decoder_keys):
                decoded_list.append(self.decoders[k](query, encoded_data[k], static_bias, masks[k], d_key))
            pooled_output = jnp.concatenate(decoded_list, axis=0)

        else:
            # Use self-attention for a single source
            pooled_output = self.decoders['self'](query, query, static_bias, masks[source_var], keys[-1])

        return self.head(pooled_output)
