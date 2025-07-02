import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class sLSTMCell(eqx.Module):
    # Based on Equation 34-40 and Figure 10 in the paper
    # Simplified for a single head and without convolution for clarity.
    # Full implementation would require handling multiple heads and convolution.

    input_size: int
    hidden_size: int
    weight_z: eqx.nn.Linear
    weight_i: eqx.nn.Linear
    weight_f: eqx.nn.Linear
    weight_o: eqx.nn.Linear
    recurrent_z: eqx.nn.Linear
    recurrent_i: eqx.nn.Linear
    recurrent_f: eqx.nn.Linear
    recurrent_o: eqx.nn.Linear
    bias_z: Array
    bias_i: Array
    bias_f: Array
    bias_o: Array

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        keys = jrandom.split(key, 8)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_z = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.weight_i = eqx.nn.Linear(input_size, hidden_size, key=keys[1])
        self.weight_f = eqx.nn.Linear(input_size, hidden_size, key=keys[2])
        self.weight_o = eqx.nn.Linear(input_size, hidden_size, key=keys[3])

        # Recurrent weights for h_{t-1}
        self.recurrent_z = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[4])
        self.recurrent_i = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[5])
        self.recurrent_f = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[6])
        self.recurrent_o = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[7])

        self.bias_z = jnp.zeros(hidden_size)
        self.bias_i = jnp.zeros(hidden_size)
        self.bias_f = jnp.zeros(hidden_size)
        self.bias_o = jnp.zeros(hidden_size)

    def __call__(
        self, x: Array, state: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], Array]:
        c_prev, n_prev, h_prev = state

        # Cell input
        tilde_z = self.weight_z(x) + self.recurrent_z(h_prev) + self.bias_z
        z = jnp.tanh(tilde_z)

        # Gates
        tilde_i = self.weight_i(x) + self.recurrent_i(h_prev) + self.bias_i
        i = jnp.exp(tilde_i)  # Exponential gating

        tilde_f = self.weight_f(x) + self.recurrent_f(h_prev) + self.bias_f
        # The paper mentions sigmoid OR exp for forget gate
        f = jax.nn.sigmoid(tilde_f)

        tilde_o = self.weight_o(x) + self.recurrent_o(h_prev) + self.bias_o
        o = jax.nn.sigmoid(tilde_o)

        # Cell state and normalizer state
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i

        # Hidden state
        h_t_tilde = c_t / n_t
        h_t = o * h_t_tilde

        # Stabilization (simplified, actual implementation needs to handle log/exp carefully)
        # This part is complex due to the log/exp and max operations for numerical stability
        # For a direct implementation, refer to Equations 15-17 in the paper and ensure
        # gradients are handled correctly as per Section A.2

        return (c_t, n_t, h_t), h_t


class sLSTM(eqx.Module):
    hidden_size: int
    reverse: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    cell: sLSTMCell
    dropout: eqx.nn.Dropout
    conv: eqx.nn.Conv1d  # Added for optional convolution in sLSTM block

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
        keys = jax.random.split(key, 2)
        self.cell = sLSTMCell(in_size, hidden_size, key=keys[0])
        self.dropout = eqx.nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.reverse = reverse
        self.return_all = return_all
        # Conv1d for window size 4
        # Assuming input is (seq_len, feature_dim), conv needs (batch, in_channels, length)
        # So, in_channels = feature_dim, out_channels = feature_dim
        self.conv = eqx.nn.Conv1d(
            1, 1, kernel_size=4, padding=3, key=keys[1]
        )  # Padding to maintain sequence length

    def __call__(self, x: Array, key: PRNGKeyArray):
        # Apply optional causal convolution if input has sequence dimension
        # x is (sequence_length, feature_dimension)
        # Transpose to (feature_dimension, sequence_length) for Conv1d, then add batch dim
        if x.ndim == 2:
            x_conv_input = x.T[jnp.newaxis, :, :]  # (1, feature_dim, seq_len)
            x_conv = self.conv(x_conv_input)
            x_conv = jax.nn.swish(x_conv)  # Swish activation
            x_processed = x_conv[0].T  # Remove batch dim, transpose back to (seq_len, feature_dim)
        else:  # Handle batched input (batch_size, sequence_length, feature_dimension)
            # Apply convolution to each item in batch
            x_conv_input = jnp.transpose(x, (0, 2, 1))  # (batch, feature_dim, seq_len)
            x_conv = jax.vmap(self.conv)(x_conv_input)
            x_conv = jax.nn.swish(x_conv)
            x_processed = jnp.transpose(x_conv, (0, 2, 1))

        x_dropped = self.dropout(x_processed, key=key)

        def scan_fn(state, xd):
            new_state, output = self.cell(xd, state)
            return new_state, output

        zeros_h = jnp.zeros(self.hidden_size)
        init_state = (zeros_h, zeros_h, zeros_h)

        _, h_all = jax.lax.scan(scan_fn, init_state, x_dropped, reverse=self.reverse)

        out = h_all if self.return_all else h_all[-1, ...]
        return out
