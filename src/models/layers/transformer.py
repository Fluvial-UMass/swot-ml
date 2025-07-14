import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import jax.random as jrandom


class StaticContextHeadBias(eqx.Module):
    out_shape: tuple
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    layernorm: eqx.nn.LayerNorm

    def __init__(self, hidden_size, num_heads, dropout, key):
        size_per_head = hidden_size // num_heads
        out_size = num_heads * size_per_head
        self.out_shape = (num_heads, hidden_size // num_heads)

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=key)
        self.dropout = eqx.nn.Dropout(dropout)
        self.layernorm = eqx.nn.LayerNorm(self.out_shape)

    def __call__(self, data, key):
        embed = self.linear(data)
        embed = self.dropout(embed, key=key)
        embed = jnp.reshape(embed, self.out_shape)
        embed = self.layernorm(embed)
        return embed


class DynamicEmbedder(eqx.Module):
    """
    Embeds input data using a linear layer. Includes dropout on the embeddings.
    """

    dynamic_embedder: eqx.nn.Linear
    positional_encoding: Array | None
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        seq_length: int,
        dynamic_in_size: int,
        hidden_size: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key)
        self.dynamic_embedder = eqx.nn.Linear(
            in_features=dynamic_in_size, out_features=hidden_size, key=keys[0]
        )
        self.positional_encoding = self.create_positional_encoding(seq_length, hidden_size)
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, data: Array, key: PRNGKeyArray) -> Array:
        embed = jax.vmap(self.dynamic_embedder)(data)
        embed += self.positional_encoding
        embed = self.dropout(embed, key=key)
        embed = jax.vmap(self.layernorm)(embed)
        return embed

    @staticmethod
    def create_positional_encoding(seq_length: int, d_model: int) -> Array:
        pos = jnp.arange(seq_length)[:, jnp.newaxis]
        i = jnp.arange(d_model // 2)[jnp.newaxis, :]
        angle_rads = pos / jnp.power(10000, (2 * i) / d_model)

        pos_encoding = jnp.zeros((seq_length, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(angle_rads))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(angle_rads))
        return pos_encoding


def _create_time_encoding(hidden_size: int, seq_len: int) -> jnp.ndarray:
    """Generates time encodings based on position."""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, hidden_size, 2) * -(jnp.log(10000.0) / hidden_size))

    time_encoding = jnp.zeros((seq_len, hidden_size))
    time_encoding = time_encoding.at[:, : len(div_term)].set(jnp.sin(position * div_term))
    if hidden_size > 1:
        time_encoding = time_encoding.at[:, 1 : len(div_term) + 1].set(jnp.cos(position * div_term))

    return time_encoding


class AttentionBlock(eqx.Module):
    """
    Implements a multi-head self-attention mechanism, integrating static data into the attention process.
    Includes dropout in the output of the attention.
    """

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        size: int | tuple[int],
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        if isinstance(size, int):
            self.attention = eqx.nn.MultiheadAttention(num_heads, size, key=key)
            output_size = size
        elif isinstance(size, tuple) and len(size) == 4:
            self.attention = eqx.nn.MultiheadAttention(num_heads, *size, key=key)
            query_size, key_size, value_size, output_size = size
        else:
            raise ValueError(f"'size' argument must an int or a tuple of 4 ints. Received {size}")
        self.layernorm = eqx.nn.LayerNorm(shape=(output_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Array | tuple[Array, Array, Array],
        static_bias: Array,
        mask: Array,
        key: PRNGKeyArray,
    ) -> Array:
        keys = jrandom.split(key)
        # Arg 'inputs' can be a tuple of three arrays for cross attention,
        # or a single array for self attention.
        if isinstance(inputs, tuple) and len(inputs) == 3:
            q, k, v = inputs
        else:
            q = k = v = inputs

        if static_bias is None:
            process_heads = None
        else:
            # function that is called on the Q, K, V after projection
            def process_heads(q_h, k_h, v_h):
                q_h += static_bias
                k_h += static_bias
                return q_h, k_h, v_h

        attention_output = self.attention(q, k, v, mask, process_heads=process_heads, key=keys[0])
        attention_output = self.dropout(attention_output, key=keys[1])
        result = attention_output + q  # Residual connection
        result = jax.vmap(self.layernorm)(result)
        return result


class FeedForwardBlock(eqx.Module):
    """
    Applies a two-layer feed-forward network with GELU activation in between. Includes dropout after the MLP layer.
    """

    one: eqx.nn.Linear
    two: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, size: int | tuple[int], dropout_rate: float, key: PRNGKeyArray):
        keys = jrandom.split(key)

        if isinstance(size, int):
            hidden_size = intermediate_size = size
        elif isinstance(size, tuple) and len(size) == 2:
            hidden_size, intermediate_size = size
        else:
            raise ValueError(f"F'size' argument must an int or a tuple of 2 ints. Received {size}")

        self.one = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=keys[0]
        )
        self.two = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=keys[1]
        )
        self.layernorm = eqx.nn.LayerNorm(shape=(hidden_size,))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs: Array, key: PRNGKeyArray) -> Array:
        hidden = self.one(inputs)
        hidden = jax.nn.gelu(hidden)
        hidden = self.dropout(hidden, key=key)
        output = self.two(hidden)
        output += inputs  # Residual connection
        output = self.layernorm(output)
        return output


class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        attn_size: int | tuple[int],
        ff_size: int | tuple[int],
        num_heads: int,
        dropout: float,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key)
        self.attention_block = AttentionBlock(attn_size, num_heads, dropout, keys[0])
        self.ff_block = FeedForwardBlock(ff_size, dropout, keys[1])

    def __call__(
        self,
        inputs: Array | tuple[Array, Array, Array],
        logit_bias: Array,
        mask: Array,
        key: PRNGKeyArray,
    ) -> Array:
        keys = jrandom.split(key)
        attention_output = self.attention_block(inputs, logit_bias, mask, keys[0])
        ff_keys = jrandom.split(keys[1], attention_output.shape[0])
        output = jax.vmap(self.ff_block)(attention_output, ff_keys)
        return output


class SelfAttnEncoder(eqx.Module):
    head_proj: StaticContextHeadBias
    embedder: DynamicEmbedder
    layers: list[TransformerLayer]

    def __init__(
        self,
        seq_len: int,
        dynamic_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key, num=3)

        self.head_proj = StaticContextHeadBias(hidden_size, num_heads, dropout, keys[0])
        self.embedder = DynamicEmbedder(seq_len, dynamic_size, hidden_size, dropout, keys[0])

        layer_keys = jrandom.split(keys[1], num=num_layers)
        layer_args = (hidden_size, intermediate_size, num_heads, dropout)
        self.layers = [TransformerLayer(*layer_args, k) for k in layer_keys]

    def __call__(
        self,
        dynamic_data: Array,
        static_encoded: Array,
        mask: Array | None,
        key: PRNGKeyArray,
    ) -> Array:
        keys = jrandom.split(key, 3)

        head_bias = self.head_proj(static_encoded, keys[0])
        dynamic_embedded = self.embedder(dynamic_data, keys[1])

        if mask is not None:
            # # For self attn, mask queries AND keys/values
            mask = jnp.outer(mask, mask)

        layer_keys = jrandom.split(keys[2], num=len(self.layers))
        x = dynamic_embedded
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, head_bias, mask, layer_key)
        return x


class CrossAttnDecoder(eqx.Module):
    head_proj: StaticContextHeadBias
    layers: list[TransformerLayer]
    time_encoding: Array
    pooler: eqx.nn.Linear

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        attn_size: int | tuple[int],
        ff_size: int | tuple[int],
        num_layers: int,
        num_heads: int,
        dropout: float,
        entity_aware: bool,
        key: PRNGKeyArray,
        pool: bool = True,
    ):
        keys = jrandom.split(key, 3)

        if entity_aware:
            self.head_proj = StaticContextHeadBias(hidden_size, num_heads, dropout, keys[0])
        else:
            self.head_proj = None

        layer_keys = jrandom.split(keys[1], num=num_layers)
        layer_args = (attn_size, ff_size, num_heads, dropout)
        self.layers = [TransformerLayer(*layer_args, k) for k in layer_keys]
        self.time_encoding = _create_time_encoding(hidden_size, seq_len)
        if pool:
            self.pooler = eqx.nn.Linear(
                in_features=hidden_size, out_features=hidden_size, key=keys[2]
            )
        else:
            self.pooler = None

    def __call__(
        self,
        daily_encoded: Array,
        irregular_encoded: Array,
        static_encoded: Array | None,
        mask: Array | None,
        key: PRNGKeyArray,
    ) -> Array:
        keys = jrandom.split(key)

        if mask is not None:
            # For cross attn, mask keys/values where they are invalid.
            mask = jnp.tile(mask, (irregular_encoded.shape[0], 1))

        q = daily_encoded + self.time_encoding
        k = irregular_encoded + self.time_encoding
        v = irregular_encoded

        if self.head_proj:
            head_bias = self.head_proj(static_encoded, keys[0])
        else:
            head_bias = None

        layer_keys = jrandom.split(keys[1], num=len(self.layers))
        x = (q, k, v)
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, head_bias, mask, layer_key)
            mask = None

        if self.pooler:
            final_token = x[-1, :]
            # final_token = jnp.mean(x, axis=0)
            pooled = self.pooler(final_token)
            # pooled = jnp.tanh(pooled)
            return pooled
        else:
            return x
