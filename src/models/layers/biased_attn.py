import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class BiasedMultiHeadAttention(eqx.Module):
    """A custom multi-head attention layer that fuses a spatial graph bias."""

    num_heads: int
    query_size: int
    qk_size: int
    vo_size: int
    output_size: int

    use_query_bias: bool
    use_key_bias: bool
    use_value_bias: bool
    use_output_bias: bool

    w_query: eqx.nn.Linear
    w_key: eqx.nn.Linear
    w_value: eqx.nn.Linear
    w_output: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        *,
        key: PRNGKeyArray,
        qk_size: int = None,
        vo_size: int = None,
        output_size: int = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
    ):
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        self.num_heads = num_heads
        self.query_size = query_size
        self.qk_size = qk_size if qk_size is not None else query_size // num_heads
        self.vo_size = vo_size if vo_size is not None else query_size // num_heads
        self.output_size = output_size if output_size is not None else query_size

        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

        self.w_query = eqx.nn.Linear(
            query_size, num_heads * self.qk_size, use_bias=use_query_bias, key=qkey
        )
        self.w_key = eqx.nn.Linear(
            query_size, num_heads * self.qk_size, use_bias=use_key_bias, key=kkey
        )
        self.w_value = eqx.nn.Linear(
            query_size, num_heads * self.vo_size, use_bias=use_value_bias, key=vkey
        )
        self.w_output = eqx.nn.Linear(
            num_heads * self.vo_size, self.output_size, use_bias=use_output_bias, key=okey
        )

        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self, x: Array, spatial_bias: Array, mask: Array = None, *, key: PRNGKeyArray
    ) -> Array:
        seq_len, _ = x.shape

        # Project and reshape for multi-head attention
        query_heads = self._project(self.w_query, x)
        key_heads = self._project(self.w_key, x)
        value_heads = self._project(self.w_value, x)

        # Calculate attention scores
        # (num_heads, seq_len, qk_size) @ (num_heads, qk_size, seq_len) -> (num_heads, seq_len, seq_len)
        scores = query_heads @ key_heads.transpose(0, 2, 1)
        scores = scores / jnp.sqrt(self.qk_size)

        # --- Fused Spatio-Temporal Masking ---
        # 1. Add the spatial bias directly to the attention scores
        # Broadcast spatial_bias from [seq, seq] to [num_heads, seq, seq]
        scores = scores + spatial_bias[None, :, :]

        # 2. Apply the temporal mask for missing data
        if mask is not None:
            # Broadcast mask from [seq_len] to [1, 1, seq_len]
            mask = mask[None, None, :]
            scores = jnp.where(mask, -1e9, scores)

        # 3. Apply causal mask to prevent attending to future timesteps
        causal_mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + causal_mask[None, :, :]
        # --- End Fused Masking ---

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, key=key)

        # Apply attention to value heads
        # (num_heads, seq_len, seq_len) @ (num_heads, seq_len, vo_size) -> (num_heads, seq_len, vo_size)
        attn_output = attn_weights @ value_heads

        # Concatenate heads and apply final projection
        # (num_heads, seq_len, vo_size) -> (seq_len, num_heads * vo_size)
        concatenated_output = attn_output.transpose(1, 0, 2).reshape(seq_len, -1)

        return jax.vmap(self.w_output)(concatenated_output)

    def _project(self, weight_matrix: eqx.nn.Linear, x: Array):
        seq_len, _ = x.shape
        projection = jax.vmap(weight_matrix)(x)
        return projection.reshape(seq_len, self.num_heads, -1).transpose(1, 0, 2)


class BiasedTransformerLayer(eqx.Module):
    """A complete transformer encoder layer with fused spatio-temporal attention."""

    attention: BiasedMultiHeadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    feed_forward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(self, input_size: int, num_heads: int, dropout_p: float, *, key: PRNGKeyArray):
        akey, fkey = jrandom.split(key, 2)
        self.attention = BiasedMultiHeadAttention(
            num_heads=num_heads, query_size=input_size, dropout_p=dropout_p, key=akey
        )
        self.norm1 = eqx.nn.LayerNorm(input_size)
        self.norm2 = eqx.nn.LayerNorm(input_size)
        self.feed_forward = eqx.nn.MLP(
            in_size=input_size, out_size=input_size, width_size=4 * input_size, depth=2, key=fkey
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        x: Array,
        spatial_bias: Array,
        mask: Array = None,
        *,
        key: PRNGKeyArray,
    ) -> Array:
        # The input x has shape (seq_len, num_nodes, hidden_size). We vmap over the nodes.
        # This treats each node's time series independently in the attention mechanism,
        # with the spatial information injected as a bias.

        # This implementation assumes attention is over the time dimension.
        # Let's adjust for applying attention over nodes at each timestep.
        # New expected x shape: (num_nodes, seq_len, hidden_size)

        attn_key, ff_key = jrandom.split(key, 2)

        # Self-attention + residual connection
        attn_input = jax.vmap(self.norm1)(x)
        attn_output = self.attention(attn_input, spatial_bias, mask, key=attn_key)
        x = x + self.dropout(attn_output, key=attn_key)

        # Feed-forward + residual connection
        ff_input = jax.vmap(self.norm2)(x)
        ff_output = jax.vmap(self.feed_forward)(ff_input)
        x = x + self.dropout(ff_output, key=ff_key)

        return x
