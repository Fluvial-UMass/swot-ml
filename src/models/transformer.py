import math
from functools import partial
from typing import cast, Dict, List, Union, Tuple, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class StaticEmbedder(eqx.Module):
    linear: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    seq_length: int
    num_heads: int
    
    def __init__(self, 
                 seq_length: int,
                 static_in_size:int, 
                 num_heads: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey):
        self.linear = eqx.nn.Linear(static_in_size, seq_length * num_heads * seq_length, key=key)
        self.layernorm = eqx.nn.LayerNorm(seq_length * num_heads * seq_length)  # Adding LayerNorm
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.seq_length = seq_length
        self.num_heads = num_heads
    
    def __call__(self, 
                 data:jnp.ndarray, 
                 key: jax.random.PRNGKey) -> jnp.ndarray: 
        embed = self.linear(data)
        embed = jax.nn.relu(embed) 
        embed = self.dropout(embed, key=key)
        embed = self.layernorm(embed)
        embed = jnp.reshape(embed, (self.seq_length, self.num_heads * self.seq_length))
        return embed

class DynamicEmbedder(eqx.Module):
    """
    Embeds input data using a linear layer. Includes dropout on the embeddings.
    """
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
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

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

    
def biased_dot_product_attention_weights(query, key, bias, mask = None):
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key) + bias

    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights

def dot_product_attention(query, key_, value, bias, mask = None, dropout = None, *, key = None, inference = None):
    weights = biased_dot_product_attention_weights(query, key_, bias, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn

class LogitBiasedMultiheadAttention(eqx.nn.MultiheadAttention):        
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        logit_bias: Float[Array, "q_seq num_heads*kv_size"],
        mask = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        process_heads = None) -> Float[Array, "q_seq o_size"]:
        
        """Extends the call method to pass the bias term through to the attention computation."""
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        if (logit_bias.shape != (query_seq_length, self.num_heads*kv_seq_length)):
            raise ValueError(
                f"Logit bias must have shape (query_seq_length, num_heads * kv_seq_length)",
                f"({query.shape[0]},{key.shape[0]}). Got {mask.shape}."
            )
            
        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)
        logit_bias_heads = logit_bias.reshape(query_seq_length, self.num_heads, kv_seq_length)
    
        attn_fn = partial(dot_product_attention, dropout=self.dropout, inference=inference)
        
        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        if mask is not None and mask.ndim == 3:
            # Batch `mask` and `keys` down their 0-th dimension.
            attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, logit_bias_heads, mask=mask, key=keys
            )
        else:
            # Batch `keys` down its 0-th dimension.
            attn = jax.vmap(partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, logit_bias_heads, key=keys
            )
        attn = attn.reshape(query_seq_length, -1)
        return jax.vmap(self.output_proj)(attn)

class AttentionBlock(eqx.Module):
    """
    Implements a multi-head self-attention mechanism, integrating static data into the attention process.
    Includes dropout in the output of the attention.
    """
    attention: LogitBiasedMultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey): 
        keys = jax.random.split(key)

        self.attention = LogitBiasedMultiheadAttention(num_heads, hidden_size, hidden_size, hidden_size, hidden_size, key=keys[0])
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, 
                 inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 logit_bias: jnp.ndarray,
                 mask: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        # Arg 'inputs' can be a tuple of three arrays for cross attention,
        # or a single array for self attention.
        if isinstance(inputs, tuple) and len(inputs) == 3:
            q, k, v = inputs
        else:
            q = k = v = inputs

        attention_output = self.attention(q, k, v, logit_bias, mask)
        attention_output = self.dropout(attention_output, key=key)
        result = attention_output + q # Residual connection
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
        output += inputs # Residual connection
        output = self.layernorm(output) 
        return output

class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_heads: int, 
                 dropout_p: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)
        self.attention_block = AttentionBlock(hidden_size, num_heads, dropout_p, keys[0])
        self.ff_block = FeedForwardBlock(hidden_size, intermediate_size, dropout_p, keys[1])

    def __call__(self, 
                 inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 logit_bias: jnp.ndarray,
                 mask: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key) 
        attention_output = self.attention_block(inputs, logit_bias, mask, keys[0])
        ff_keys = jax.random.split(keys[1], attention_output.shape[0])
        output = jax.vmap(self.ff_block)(attention_output, ff_keys)
        return output

class SelfAttnEncoder(eqx.Module):
    embedder: DynamicEmbedder
    layers: List[TransformerLayer]

    def __init__(self,
                 seq_length: int,
                 dynamic_size: int,
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 dropout_p: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key, num=3)
        
        self.embedder = DynamicEmbedder(seq_length, dynamic_size, hidden_size, dropout_p, keys[0])
        
        layer_keys = jax.random.split(keys[1], num=num_layers)
        layer_args = (hidden_size, intermediate_size, num_heads, dropout_p)
        self.layers = [TransformerLayer(*layer_args, k) for k in layer_keys]

    def __call__(self, 
                 dynamic_data: jnp.ndarray,
                 logit_bias: jnp.ndarray,
                 mask: Union[None, jnp.ndarray],
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key, num=3)
        
        dynamic_embedded = self.embedder(dynamic_data, keys[0])
        
        if mask is not None:
            # For self attn, mask queries AND keys/values
            mask = jnp.outer(mask, mask)
        
        layer_keys = jax.random.split(keys[1], num=len(self.layers))
        x = dynamic_embedded
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, logit_bias, mask, layer_key)
        return x

class CrossAttnDecoder(eqx.Module):
    layers: List[TransformerLayer]
    pooler: eqx.nn.Linear

    def __init__(self,
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 dropout_p: float,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key)

        layer_keys = jax.random.split(keys[0], num=num_layers)
        layer_args = (hidden_size, intermediate_size, num_heads, dropout_p)
        self.layers = [TransformerLayer(*layer_args, k) for k in layer_keys]
        
        self.pooler = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=keys[1])

    def __call__(self, 
                 daily_encoded: jnp.ndarray,
                 irregular_encoded: jnp.ndarray,
                 logit_bias: jnp.ndarray,
                 mask: Union[None, jnp.ndarray],
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key)

        if mask is not None:
            # For cross attn, mask keys/values where they are invalid.
            mask = jnp.tile(mask, (irregular_encoded.shape[0], 1))
            # const_mask = jnp.ones_like(irregular_mask)
            # dual_mask = jnp.stack((irregular_mask, const_mask), axis=0)

        q = daily_encoded
        k = v = irregular_encoded
        
        layer_keys = jax.random.split(keys[1], num=len(self.layers))
        x = (q, k, v)
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, logit_bias, mask, layer_key)
            mask = None

        final_token = x[0, :]
        # final_token = jnp.mean(x, axis=0)
        pooled = self.pooler(final_token)
        # pooled = jnp.tanh(pooled)
        return pooled


class EATransformer(eqx.Module):
    static_embedder: StaticEmbedder
    d_encoder: SelfAttnEncoder
    i_encoder: SelfAttnEncoder
    decoder: CrossAttnDecoder
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
                 dropout_p: float, 
                 seed: int):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, num=5)
        self.static_embedder = StaticEmbedder(seq_length, static_in_size, num_heads, dropout_p, keys[0])
        
        static_args = (hidden_size, intermediate_size, num_layers, num_heads, dropout_p) 
        self.d_encoder = SelfAttnEncoder(seq_length, daily_in_size, *static_args, keys[1])
        self.i_encoder = SelfAttnEncoder(seq_length, irregular_in_size, *static_args, keys[2])
        self.decoder = CrossAttnDecoder(*static_args, keys[3])
        self.head = eqx.nn.Linear(in_features=hidden_size, out_features=out_size, key=keys[4])

    def __call__(self, data: dict, key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key, num=4)

        # Kluge to get it running now. Will address data loader later.        
        mask = ~jnp.any(jnp.isnan(data['x_di']),axis=1)
        data['x_di'] = jnp.where(jnp.isnan(data['x_di']), -999.0, data['x_di'])
        # Kluge to get it running now. Will address data loader later.

        logit_bias = self.static_embedder(data['x_s'], keys[0])
        # static_embedded = 0
        d_encoded = self.d_encoder(data['x_dd'], logit_bias, None, keys[1])
        i_encoded = self.i_encoder(data['x_di'], logit_bias, mask, keys[2])
        pooled_output = self.decoder(d_encoded, i_encoded, logit_bias, mask, keys[3])

        return self.head(pooled_output)