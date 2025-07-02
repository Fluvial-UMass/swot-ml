import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class mLSTMCell(eqx.Module):
    # Based on Equation 19-27 and Figure 11 in the paper
    # Simplified for a single head and without convolution for clarity.
    # Full implementation would require handling multiple heads and convolution.

    input_size: int
    hidden_size: int  # This will be 'd' in the paper for key/value/query dimensions
    weight_q: eqx.nn.Linear
    weight_k: eqx.nn.Linear
    weight_v: eqx.nn.Linear
    bias_q: Array
    bias_k: Array
    bias_v: Array
    weight_i: eqx.nn.Linear
    weight_f: eqx.nn.Linear
    weight_o: eqx.nn.Linear
    bias_i: Array
    bias_f: Array
    bias_o: Array

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        keys = jrandom.split(key, 9)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_q = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.weight_k = eqx.nn.Linear(input_size, hidden_size, key=keys[1])
        self.weight_v = eqx.nn.Linear(input_size, hidden_size, key=keys[2])

        self.bias_q = jnp.zeros(hidden_size)
        self.bias_k = jnp.zeros(hidden_size)
        self.bias_v = jnp.zeros(hidden_size)

        self.weight_i = eqx.nn.Linear(input_size, hidden_size, key=keys[3])
        self.weight_f = eqx.nn.Linear(input_size, hidden_size, key=keys[4])
        self.weight_o = eqx.nn.Linear(input_size, hidden_size, key=keys[5])

        self.bias_i = jnp.zeros(hidden_size)
        self.bias_f = jnp.zeros(hidden_size)
        self.bias_o = jnp.zeros(hidden_size)

    def __call__(self, x: Array, state: tuple[Array, Array]) -> tuple[tuple[Array, Array], Array]:
        C_prev, n_prev = state  # C is the matrix memory, n is the normalizer state

        # Query, Key, Value
        q = self.weight_q(x) + self.bias_q
        k = (1 / jnp.sqrt(self.hidden_size)) * self.weight_k(x) + self.bias_k
        v = self.weight_v(x) + self.bias_v

        # Gates
        tilde_i = self.weight_i(x) + self.bias_i
        i = jnp.exp(tilde_i)

        tilde_f = self.weight_f(x) + self.bias_f
        # The paper mentions sigmoid OR exp for forget gate
        f = jax.nn.sigmoid(tilde_f)

        tilde_o = self.weight_o(x) + self.bias_o
        o = jax.nn.sigmoid(tilde_o)

        # Cell state (Matrix Memory) and normalizer state
        C_t = f * C_prev + i * jnp.outer(v, k)  # Outer product v_t k_t^T
        n_t = f * n_prev + i * k  # Normalizer state is weighted sum of key vectors

        # Hidden state
        # The normalizer state dot product with query can be near zero, lower bound by 1.0
        denominator = jnp.maximum(jnp.abs(jnp.dot(n_t, q)), 1.0)
        h_t_tilde = jnp.dot(C_t, q) / denominator
        h_t = o * h_t_tilde

        return (C_t, n_t), h_t


class mLSTM(eqx.Module):
    hidden_size: int
    reverse: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    cell: mLSTMCell
    dropout: eqx.nn.Dropout
    conv: eqx.nn.Conv1d  # Added for optional convolution in mLSTM block

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
        self.cell = mLSTMCell(in_size, hidden_size, key=keys[0])
        self.dropout = eqx.nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.reverse = reverse
        self.return_all = return_all
        # Conv1d for kernel size 4
        self.conv = eqx.nn.Conv1d(
            1, 1, kernel_size=4, padding=3, key=keys[1]
        )  # Padding to maintain sequence length

    def __call__(self, x: Array, key: PRNGKeyArray):
        # Apply optional dimension-wise causal convolution
        # x is (sequence_length, feature_dimension)
        if x.ndim == 2:
            x_conv_input = x.T[jnp.newaxis, :, :]
            x_conv = self.conv(x_conv_input)
            x_processed = x_conv[0].T
        else:  # Handle batched input (batch_size, sequence_length, feature_dimension)
            x_conv_input = jnp.transpose(x, (0, 2, 1))
            x_conv = jax.vmap(self.conv)(x_conv_input)
            x_processed = jnp.transpose(x_conv, (0, 2, 1))

        x_dropped = self.dropout(x_processed, key=key)

        def scan_fn(state, xd):
            new_state, output = self.cell(xd, state)
            return new_state, output

        zeros_C = jnp.zeros((self.hidden_size, self.hidden_size))
        zeros_n = jnp.zeros(self.hidden_size)
        init_state = (zeros_C, zeros_n)

        _, h_all = jax.lax.scan(scan_fn, init_state, x_dropped, reverse=self.reverse)

        out = h_all if self.return_all else h_all[-1, ...]
        return out
