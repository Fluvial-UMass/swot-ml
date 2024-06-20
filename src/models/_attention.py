import math
from functools import partial
from typing import cast, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


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


class LogitBiasedMHA(eqx.nn.MultiheadAttention):        
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
                f"({query.shape[0]},{key_.shape[0]}). Got {mask.shape}."
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