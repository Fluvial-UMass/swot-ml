import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class LSTM(eqx.Module):
    """Standard LSTM model built on the BaseLSTM class."""

    hidden_size: int
    reverse: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    cell: eqx.nn.LSTMCell
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        *,
        dropout: float = 0,
        reverse: bool = False,
        return_all: bool = False,
        key: PRNGKeyArray,
    ):
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)
        self.dropout = eqx.nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.reverse = reverse
        self.return_all = return_all

    def __call__(self, x: Array, key: PRNGKeyArray):
        def scan_fn(state, xd):
            return self.cell(xd, state), state[0]

        zeros = jnp.zeros(self.hidden_size)
        init_state = (zeros, zeros)
        (h_t, _), h_all = jax.lax.scan(scan_fn, init_state, x, reverse=self.reverse)

        out = h_all if self.return_all else h_t
        return self.dropout(out, key=key)
