
from typing import Dict, List, Union, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

    
class StaticEmbedder(eqx.Module):
    """Embeds static data to modulate attention."""
    out_shape: tuple
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    layernorm: eqx.nn.LayerNorm

    def __init__(self,
                 static_size: int, 
                 hidden_size: int,
                 num_heads: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey):
        
        size_per_head = hidden_size // num_heads
        out_size = num_heads * size_per_head
        self.out_shape = (num_heads, hidden_size // num_heads)
        
        self.linear = eqx.nn.Linear(static_size, out_size, key=key)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.layernorm = eqx.nn.LayerNorm(self.out_shape)

    def __call__(self, 
                 data: jnp.ndarray,
                 key) -> jnp.ndarray:
        embed = self.linear(data)
        embed = self.dropout(embed, key=key)
        embed = jnp.reshape(embed, self.out_shape)
        embed = self.layernorm(embed)
        return embed

class DynamicEmbedder(eqx.Module):
    """Embeds input data using a linear layer to model dimension, adds positional encoding,"""
    dynamic_embedder: eqx.nn.Linear
    positional_encoding: Union[None, jnp.ndarray]
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self,
                 seq_length: int,
                 dynamic_in_size: int,
                 hidden_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)
        self.dynamic_embedder = eqx.nn.Linear(in_features=dynamic_in_size, out_features=hidden_size, key=keys[0])
        self.positional_encoding = self.create_positional_encoding(seq_length, hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))

    def __call__(self, 
                 data: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray: 
        embed = jax.vmap(self.dynamic_embedder)(data)
        embed += self.positional_encoding
        embed = self.dropout(embed, key=key)
        embed = jax.vmap(self.layernorm)(embed)
        return embed

    @staticmethod
    def create_positional_encoding(seq_length: int, d_model: int) -> jnp.ndarray:
        pos = jnp.arange(seq_length)[:, jnp.newaxis]
        i = jnp.arange(d_model // 2)[jnp.newaxis, :]
        angle_rads = pos / jnp.power(10000, (2 * i) / d_model)

        pos_encoding = jnp.zeros((seq_length, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(angle_rads))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(angle_rads))
        return pos_encoding

class AttentionBlock(eqx.Module):
    """
    Implements a multi-head self-attention mechanism, integrating static data into the attention process.
    Includes dropout in the output of the attention.
    """
    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self,
                 hidden_size: int, 
                 num_heads: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey): 
        self.attention = eqx.nn.MultiheadAttention(num_heads, hidden_size, hidden_size, hidden_size, hidden_size, key=key)
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, 
                 inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 static_bias: jnp.ndarray,
                 mask: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        # Arg 'inputs' can be a tuple of three arrays for cross attention,
        # or a single array for self attention.
        if isinstance(inputs, tuple) and len(inputs) == 3:
            q, k, v = inputs
        else:
            q = k = v = inputs

        def process_heads(q_h, k_h, v_h):
            q_h += static_bias
            k_h += static_bias
            return q_h, k_h, v_h
        
        attention_output = self.attention(q, k, v, mask, process_heads=process_heads)
        attention_output = self.dropout(attention_output, key=key)
        result = attention_output + q # Residual connection
        result = jax.vmap(self.layernorm)(result)
        return result

class FeedForwardBlock(eqx.Module):
    """
    Applies a two-layer feed-forward network with GELU activation in between. Includes dropout after the MLP layer.
    """
    mlp: eqx.nn.MLP
    dropout: eqx.nn.Dropout
    layernorm: eqx.nn.LayerNorm

    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey): 
        self.mlp = eqx.nn.MLP(hidden_size, hidden_size, intermediate_size, 
                              depth=2, activation=jax.nn.gelu, key=key)
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self,
                 x: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray: 
        output = self.mlp(x) + x
        output = self.dropout(output, key=key)
        output = self.layernorm(output) 
        return output

class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(self,
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_heads: int, 
                 dropout: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)
        self.attention_block = AttentionBlock(hidden_size, num_heads, dropout, keys[0])
        self.ff_block = FeedForwardBlock(hidden_size, intermediate_size, dropout, keys[1])

    def __call__(self, 
                 inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 static_bias: jnp.ndarray,
                 mask: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key) 
        attention_output = self.attention_block(inputs, static_bias, mask, keys[0])

        ff_keys = jax.random.split(keys[1], attention_output.shape[0])
        output = jax.vmap(self.ff_block)(attention_output, ff_keys)
        return output

class CrossAttnFusion(eqx.Module):
    layers: List[TransformerLayer]

    def __init__(self,
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 dropout: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)

        layer_keys = jax.random.split(keys[0], num=num_layers)
        layer_args = (hidden_size, intermediate_size, num_heads, dropout)
        self.layers = [TransformerLayer(*layer_args, k) for k in layer_keys]
        
    def __call__(self, 
                 daily: jnp.ndarray,
                 irregular: jnp.ndarray,
                 static: jnp.ndarray,
                 mask: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key)
        
        # Mask the keys and values according to 1d mask
        mask = jnp.tile(mask, (irregular.shape[0], 1))

        layer_keys = jax.random.split(keys[1], num=len(self.layers))
        x = (daily, daily, irregular) # k, q, v
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, static, mask, layer_key)
            mask = None
        return x
    
class MeanPooler(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self,
                 hidden_size: int,
                 key: jax.random.PRNGKey):
        self.linear = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_mu = jnp.mean(x, axis=0)
        return self.linear(x_mu)

class FusionTransformer(eqx.Module):
    static_embedder: StaticEmbedder
    daily_embedder: DynamicEmbedder
    irregular_embedder: DynamicEmbedder
    fusion: CrossAttnFusion
    pooler: MeanPooler
    head: eqx.nn.Linear

    def __init__(self, 
                 daily_in_size: int,
                 irregular_in_size: int,
                 static_in_size: int, 
                 seq_length: int, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 out_size: int,
                 dropout: float, 
                 seed: int):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, num=6)

        self.static_embedder = StaticEmbedder(static_in_size, hidden_size, num_heads, dropout, keys[0])
        self.daily_embedder = DynamicEmbedder(seq_length, daily_in_size, hidden_size, dropout, keys[1])
        self.irregular_embedder = DynamicEmbedder(seq_length, irregular_in_size, hidden_size, dropout, keys[2])

        self.fusion = CrossAttnFusion(hidden_size, intermediate_size, 
                                      num_layers, num_heads, dropout, keys[3])
        self.pooler = MeanPooler(hidden_size, keys[4])
        self.head = eqx.nn.Linear(in_features=hidden_size, out_features=out_size, key=keys[5])

    def __call__(self, data: dict, key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key, num=4)

        # Kluge to get it running now. Will address data loader later.        
        mask = ~jnp.any(jnp.isnan(data['x_di']),axis=1)
        data['x_di'] = jnp.where(jnp.isnan(data['x_di']), 0, data['x_di'])
        # Kluge to get it running now. Will address data loader later.

        daily = self.daily_embedder(data['x_dd'], keys[0])
        irregular = self.irregular_embedder(data['x_di'], keys[1])
        static = self.static_embedder(data['x_s'], keys[2])
        
        combined = self.fusion(daily, irregular, static, mask, keys[3])
        pooled = self.pooler(combined)
        return self.head(pooled)