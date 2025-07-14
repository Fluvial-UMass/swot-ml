import jax.numpy as jnp


def sinusoidal_encoding(embedding_size: int, seq_len: int) -> jnp.ndarray:
    """Generates time encodings based on position."""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embedding_size, 2) * -(jnp.log(10000.0) / embedding_size))
    angles = position * div_term  # (seq_len, hidden_size // 2)
    time_encoding = jnp.zeros((seq_len, embedding_size))
    time_encoding = time_encoding.at[:, 0::2].set(jnp.sin(angles))
    time_encoding = time_encoding.at[:, 1::2].set(jnp.cos(angles))
    return time_encoding
