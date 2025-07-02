import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .lstm import LSTM


class BiLSTM(eqx.Module):
    """Wraps two LSTM instances (forward and backward) into a bidirectional model."""

    lstm_fwd: LSTM
    lstm_bwd: LSTM

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        *,
        dropout: float = 0,
        return_all: bool = False,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key)

        common_kwargs = {
            "in_size": in_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "return_all": return_all,
        }
        self.lstm_fwd = LSTM(key=keys[0], **common_kwargs)
        self.lstm_bwd = LSTM(key=keys[1], reverse=True, **common_kwargs)

    def __call__(self, x_d: Array, key: PRNGKeyArray):
        keys = jrandom.split(key)

        out_fwd = self.lstm_fwd(x_d, keys[0])
        out_bwd = self.lstm_bwd(x_d, keys[1])
        out = jnp.concatenate([out_fwd, out_bwd], axis=-1)  # [T, 2H]

        return out
