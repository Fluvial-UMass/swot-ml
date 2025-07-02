import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class TEALSTMCell(eqx.Module):
    """
    A configurable LSTM cell that can include time- and/or entity-aware modifications
    based on the provided options.
    """

    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: eqx.nn.Linear | None
    weight_decomp: jax.Array | None
    bias_decomp: jax.Array | None
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        hidden_size: int,
        time_aware: bool = True,
        entity_aware: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        wkey, bkey, ikey, dkey = jrandom.split(key, 4)
        self.time_aware = time_aware
        self.entity_aware = entity_aware

        if self.entity_aware:
            num_gates = 3
            self.input_linear = eqx.nn.Linear(static_in_size, hidden_size, use_bias=True, key=ikey)
        else:
            num_gates = 4
            self.input_linear = None
        self.weight_ih = jax.nn.initializers.glorot_normal()(
            wkey, (num_gates * hidden_size, dynamic_in_size)
        )
        self.weight_hh = jax.nn.initializers.glorot_normal()(
            wkey, (num_gates * hidden_size, hidden_size)
        )
        self.bias = jax.nn.initializers.zeros(bkey, (num_gates * hidden_size,))

        if self.time_aware:
            self.weight_decomp = jax.nn.initializers.glorot_normal()(
                dkey, (hidden_size, hidden_size)
            )
            self.bias_decomp = jax.nn.initializers.zeros(dkey, (hidden_size,))
        else:
            self.weight_decomp = None
            self.bias_decomp = None

    def _decomp_and_decay(self, c_0: Array, skip_count: int):
        if not self.time_aware:
            return c_0
        cs_0 = jnp.tanh(jnp.dot(c_0, self.weight_decomp.T) + self.bias_decomp)
        ct_0 = c_0 - cs_0
        cs_hat_0 = cs_0 / (1 + skip_count)
        c_star = ct_0 + cs_hat_0
        return c_star

    def _skip_update(self, operand):
        state, _, _, skip_count = operand
        skip_count += 1
        return state, skip_count

    def _update_cell(self, operand):
        state, x_d, i, skip_count = operand
        h_0, c_0 = state

        # Apply time decay if we have skipped any updates.
        c_0 = lax.cond(
            skip_count > 0,
            lambda _: self._decomp_and_decay(c_0, skip_count),  # Pass c_0 to the decay function
            lambda _: c_0,  # Return c_0 as is
            operand=None,
        )  # The operand is not used in the functions

        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(h_0, self.weight_hh.T) + self.bias
        if self.entity_aware:
            f, g, o = jnp.split(gates, 3, axis=-1)
            # i is passed in since it is static
        else:
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update the state
        c_1 = f * c_0 + i * g
        h_1 = o * jnp.tanh(c_1)

        # return state and 0 for skip_count
        skip_count = 0
        return (h_1, c_1), skip_count

    def __call__(self, state: tuple[Array, Array], x_d: Array, x_s: Array, skip_count: int):
        """
        Forward pass of the TEALSTMCell module.

        Args:
            state (tuple): Tuple containing the hidden and cell states.
            x_d (jax.Array): Dynamic input features.
            x_s (jax.Array): Static input features.
            skip_count (int): Number of skipped updates due to missing data.

        Returns:
            Updated state and current skip count
        """
        is_nan = jnp.any(jnp.isnan(x_d))
        state = lax.cond(
            is_nan,
            self._skip_update,
            self._update_cell,
            operand=(state, x_d, x_s, skip_count),
        )

        return state


class TEALSTM(eqx.nn.Module):
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    hidden_size: int
    cell: TEALSTMCell
    dropout: eqx.nn.Dropout
    """
    Time- and Entity-Aware LSTM (TEALSTM) model.
    """

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        hidden_size: int,
        dropout: float,
        time_aware: bool = True,
        return_all: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.time_aware = time_aware
        self.entity_aware = static_in_size > 0
        self.return_all = return_all

        self.cell = TEALSTMCell(
            dynamic_in_size,
            static_in_size,
            hidden_size,
            time_aware,
            self.entity_aware,
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x_d: Array, x_s: Array, key: PRNGKeyArray):
        """
        Forward pass of the TEALSTM.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM and final skip count.
        """
        i = jax.nn.sigmoid(self.cell.input_linear(x_s)) if self.entity_aware else None

        def scan_fn(state, x):
            skip_count = state[2]
            new_state, skip_count = self.cell(state[:2], x, i, skip_count)
            return (*new_state, skip_count), new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), int(0))
        (h_t, _, _), h_all = jax.lax.scan(scan_fn, init_state, x_d)

        out = h_t if self.return_all else h_all

        return self.dropout(out, key=key)
