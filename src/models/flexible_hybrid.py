import equinox as eqx
import jax
import jax.numpy as jnp

from models.lstm import TEALSTM
from models.transformer import StaticEmbedder, CrossAttnDecoder


class FlexibleHybrid(eqx.Module):
    encoders: dict
    static_embedder: StaticEmbedder
    decoders: dict
    head: eqx.nn.Linear
    target: list

    def __init__(self, *, target, dynamic_sizes, static_size, hidden_size, num_layers, num_heads, seed, dropout):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 5) 

        # Encoders for each dynamic data source.
        encoder_keys = jax.random.split(keys[0], len(dynamic_sizes))
        self.encoders = {}
        for (var_name, var_size), var_key in zip(dynamic_sizes.items(), encoder_keys):
            self.encoders[var_name] = TEALSTM(var_size, hidden_size, hidden_size, None, dropout, return_all=True, key=var_key)

        # Encoder for static data if used.
        if static_size>0:
            self.static_embedder = StaticEmbedder(static_size, hidden_size, dropout, keys[1])
        else:
            self.static_embedder = None
        
        # Cross-attn or Self-attn decoders. 
        self.decoders = {}
        cross_vars = list(dynamic_sizes.keys())[1:]
        decoder_keys = jax.random.split(keys[2], len(cross_vars))
        # Set up each cross-attention decoder
        if len(cross_vars) > 0:
            for var_name, var_key in zip(cross_vars, decoder_keys):
                self.decoders[var_name] = CrossAttnDecoder(hidden_size, hidden_size, num_layers, num_heads, dropout, var_key)
        else:
            self.decoders['self'] = CrossAttnDecoder(hidden_size, hidden_size, num_layers, num_heads, dropout, var_key)

        self.head = eqx.nn.Linear(in_features = hidden_size*len(self.decoders), 
                                  out_features = len(target), 
                                  key = keys[3])
        self.target = target

    def __call__(self, data, key):
        keys = jax.random.split(key, 3)

        # Encoder
        encoder_keys = jax.random.split(keys[0],len(self.encoders))
        encoded_data = {}
        masks = {}
        for e_key, (var_name, encoder) in zip(self.encoders.items(), encoder_keys):
            encoded_data[var_name] = encoder(data[var_name], data['static'], e_key)
            masks[var_name] = ~jnp.any(jnp.isnan(data[var_name]),axis=1)

        if self.static_embedder:
            head_bias = self.static_embedder(data['static'], keys[1])
        else:
            head_bias = 0

        # Decoder
        source_var = list(encoded_data.keys())[0]
        cross_vars = list(encoded_data.keys())[1:]
        query = encoded_data[source_var]   

        # Use cross-attention with multiple sources
        if len(cross_vars)>0:
            decoder_keys = jax.random.split(keys[2],len(cross_vars))
            decoded_list = []
            for k, d_key in zip(cross_vars, decoder_keys):
                decoded_list.append(self.decoders[k](query, encoded_data[k], head_bias, masks[k], d_key))
            pooled_output = jnp.concatenate(decoded_list, axis=0)

        # If only one data source, use self-attention
        else:
            pooled_output = self.decoders['self'](query, query, head_bias, masks[source_var], keys[-1])

        return self.head(pooled_output)

