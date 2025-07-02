import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from lstm import LSTM


class StaticLSTM(LSTM):
    """LSTM model that concatenates static features at each time step."""

    append_static: bool

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        hidden_size: int,
        *,
        dropout: float = 0,
        reverse: bool = False,
        return_all: bool = False,
        key: PRNGKeyArray,
    ):
        """
        Args:
            dynamic_in_size: Dimensionality of the dynamic input per time step.
            static_in_size: Dimensionality of the static input.
            hidden_size: Dimensionality of the hidden and cell state.
            dropout: Dropout rate.
            reverse: Whether to process the sequence in reverse.
            return_all: Whether to return all hidden states or just the final one.
            key: JAX PRNG key.
        """
        # Call the parent class's initializer with the total input size
        super().__init__(
            in_size=dynamic_in_size + static_in_size,
            hidden_size=hidden_size,
            dropout=dropout,
            reverse=reverse,
            return_all=return_all,
            key=key,
        )
        self.append_static = static_in_size > 0

    def __call__(self, x_d: Array, x_s: Array, key: PRNGKeyArray):
        def scan_fn(state, x):
            input = jnp.concat([x, x_s], axis=-1) if self.append_static else x
            return self.cell(input, state), state[0]

        zeros = jnp.zeros(self.hidden_size)
        init_state = (zeros, zeros)
        (h_t, _), h_all = jax.lax.scan(scan_fn, init_state, x_d, reverse=self.reverse)

        out = h_all if self.return_all else h_t
        return self.dropout(out, key=key)
