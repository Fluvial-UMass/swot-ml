import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Mapping, Optional

class EmbedderBlock(eqx.Module):
    """
    Embeds input data from daily and irregular time series along with position indices.
    Includes dropout on the embeddings.
    """
    daily_embedder: eqx.nn.Linear
    irregular_embedder: eqx.nn.Linear
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, 
                 daily_in_size: int, 
                 irregular_in_size: int, 
                 max_length: int, 
                 hidden_size: int,
                 dropout_rate: float, 
                 key: jax.random.PRNGKey):
        
        self.daily_embedder = eqx.nn.Linear(in_features=daily_in_size, out_features=hidden_size, key=key)
        self.irregular_embedder = eqx.nn.Linear(in_features=irregular_in_size, out_features=hidden_size, key=key)
        self.position_embedder = eqx.nn.Embedding(num_embeddings=max_length, embedding_size=hidden_size, key=key)
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size*3,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, 
                 daily_data: jnp.ndarray, 
                 irregular_data: jnp.ndarray, 
                 position_ids: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:  
        
        daily_embeds = jax.vmap(self.daily_embedder)(daily_data)
        irregular_embeds = jax.vmap(self.irregular_embedder)(irregular_data)
        position_embeds = jax.vmap(self.position_embedder)(position_ids)
        
        # embedded_inputs = daily_embeds + irregular_embeds + position_embeds
        embedded_inputs = jnp.concatenate([daily_embeds, irregular_embeds, position_embeds], axis=-1)
        embedded_inputs = self.dropout(embedded_inputs, key=key)
        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        return embedded_inputs


class AttentionBlock(eqx.Module):
    """
    Implements a multi-head self-attention mechanism, integrating static data into the attention process.
    Includes dropout in the output of the attention.
    """
    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    static_linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 static_in_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey): 
        keys = jax.random.split(key)
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads, 
            query_size=hidden_size, 
            key_size=hidden_size, 
            value_size=hidden_size, 
            output_size=hidden_size, 
            key=keys[0])
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.static_linear = eqx.nn.Linear(in_features=static_in_size, out_features=hidden_size, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, 
                 inputs: jnp.ndarray, 
                 static_data: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        static_embedding = self.static_linear(static_data)
        static_embedding = jnp.expand_dims(static_embedding, axis=0)
        modified_keys = inputs + static_embedding
        modified_values = inputs + static_embedding
        
        attention_output = self.attention(inputs, modified_keys, modified_values)
        attention_output = self.dropout(attention_output, key=key)
        result = attention_output + inputs
        result = jax.vmap(self.layernorm)(result)
        return result

class FeedForwardBlock(eqx.Module):
    """
    Applies a two-layer feed-forward network with GELU activation in between. Includes dropout after the MLP layer.
    """
    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey): 
        keys = jax.random.split(key)
        
        self.mlp = eqx.nn.Linear(in_features=hidden_size, out_features=intermediate_size, key=keys[0])
        self.output = eqx.nn.Linear(in_features=intermediate_size, out_features=hidden_size, key=keys[1])
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self,
                 inputs: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray: 
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)
        hidden = self.dropout(hidden, key=key)
        output = self.output(hidden)
        output += inputs
        output = self.layernorm(output) 
        return output

class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_heads: int, 
                 static_in_size: int, 
                 dropout_p: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)
        
        self.attention_block = AttentionBlock(hidden_size, num_heads, static_in_size, dropout_p, keys[0])
        self.ff_block = FeedForwardBlock(hidden_size, intermediate_size, dropout_p, keys[1])

    def __call__(self, 
                 inputs: jnp.ndarray, 
                 static_data: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key)
        
        attention_output = self.attention_block(inputs, static_data, keys[0])
        ff_keys = jax.random.split(keys[1], attention_output.shape[0])
        output = jax.vmap(self.ff_block)(attention_output, ff_keys)
        return output

class Encoder(eqx.Module):
    embedder_block: EmbedderBlock
    layers: List[TransformerLayer]
    pooler: eqx.nn.Linear

    def __init__(self, 
                 daily_in_size: int, 
                 irregular_in_size: int, 
                 static_in_size: int, 
                 max_length: int, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 dropout_p: float,
                 key: jax.random.PRNGKey): 
        keys = jax.random.split(key, num=3)
        
        self.embedder_block = EmbedderBlock(daily_in_size, irregular_in_size, max_length, hidden_size, dropout_p, keys[0])
        layer_keys = jax.random.split(keys[1], num=num_layers)
        self.layers = [TransformerLayer(3*hidden_size, intermediate_size, num_heads, static_in_size, dropout_p, layer_key) for layer_key in layer_keys]
        self.pooler = eqx.nn.Linear(in_features=hidden_size*3, out_features=hidden_size, key=keys[2])

    def __call__(self, 
                 daily_data: jnp.ndarray, 
                 irregular_data: jnp.ndarray, 
                 position_ids: jnp.ndarray, 
                 static_data: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key)
        
        embeddings = self.embedder_block(daily_data, irregular_data, position_ids, keys[0])

        x = embeddings
        layer_keys = jax.random.split(keys[1], len(self.layers))
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, static_data, layer_key)
        first_token_last_layer = x[..., 0, :]
        pooled = self.pooler(first_token_last_layer)
        pooled = jnp.tanh(pooled)
        return pooled

class EATransformer(eqx.Module):
    encoder: Encoder
    head: eqx.nn.Linear

    def __init__(self, 
                 daily_in_size: int, 
                 irregular_in_size: int, 
                 static_in_size: int, 
                 max_length: int, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 out_size: int,
                 dropout_p: float, 
                 seed: int):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key)
        
        self.encoder = Encoder(daily_in_size=daily_in_size,
                               irregular_in_size=irregular_in_size,
                               static_in_size=static_in_size,
                               max_length=max_length,
                               hidden_size=hidden_size,
                               intermediate_size=intermediate_size,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               dropout_p=dropout_p,
                               key=keys[0])
        self.head = eqx.nn.Linear(in_features=hidden_size, out_features=out_size, key=keys[1])

    def __call__(self, data: dict, key: jax.random.PRNGKey) -> jnp.ndarray:
        # Kluge to get it running now. Will address data loader later.
        position_ids = jnp.arange(data['x_dd'].shape[0]).astype(jnp.int32)
        x_di = jnp.nan_to_num(data['x_di'], nan=-999.0)
        
        pooled_output = self.encoder(data['x_dd'], x_di, position_ids, data['x_s'], key)
        return self.head(pooled_output)

















