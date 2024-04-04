import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class BaseLSTM(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        self.hidden_size = hidden_size
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=key)

    def __call__(self, data):
        raise NotImplementedError("Subclasses must implement this method.")


class LSTM(BaseLSTM):
    def __init__(self, in_size, out_size, hidden_size, *, key):
        super().__init__(in_size, out_size, hidden_size, key=key)
        ckey, _ = jrandom.split(key)
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)

    def __call__(self, data):
        def scan_fn(state, xd):
            return self.cell(xd, state), None
        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, data['xd'])
        return (self.linear(out))


class Gated_LSTM(BaseLSTM):
    def __init__(self, in_size, out_size, hidden_size, *, key):
        super().__init__(in_size, out_size, hidden_size, key=key)
        ckey, _ = jrandom.split(key)
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)

    def __call__(self, data):
        def scan_fn(state, inputs):
            x, w = inputs
            input_gate = jax.nn.sigmoid(w)
            gated_x = x * input_gate
            return self.cell(gated_x, state), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, (data['xd'], data['w']))
        return (self.linear(out))


class TEALSTMCell(eqx.Module):
    """
    A Time- and Entity-Aware LSTM (EALSTM) cell .
    
    Attributes:
        hidden_size (int): The size of the hidden state in the EALSTM cell.
        input_size (int): The size of the input features.
        weight_ih (jax.Array): Input weights for input, forget, cell, and output gates.
        weight_hh (jax.Array): Hidden weights for input, forget, cell, and output gates.
        bias (jax.Array): Bias terms for forget, cell, and output gates.
        input_linear (eqx.nn.Linear): Linear transformation for static input features.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: eqx.nn.Linear
    weight_decomp: jax.Array
    bias_decomp: jax.Array
    
    def __init__(self, dynamic_input_size, static_input_size, hidden_size, out_size, *, key):
        """
        Initializes the EALSTM cell.

        Args:
            dynamic_input_size (int): The size of the dynamic input features.
            static_input_size (int): The size of the static input features.
            hidden_size (int): The size of the hidden state in the EALSTM cell.
            out_size (int): The size of the output features.
            key (jax.random.PRNGKey): A random key for initializing the cell parameters.
        """
        wkey, bkey, ikey, dkey = jrandom.split(key, 4)
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, dynamic_input_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (3 * hidden_size,))
        self.input_linear = eqx.nn.Linear(static_input_size, hidden_size, use_bias=True, key=ikey)
        self.weight_decomp = jax.nn.initializers.glorot_normal()(dkey, (hidden_size, hidden_size))
        self.bias_decomp = jax.nn.initializers.zeros(dkey, (hidden_size))
        
    def _decomp_and_decay(self, c_0, skip_count):
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
        state, x_d, x_s, skip_count = operand
        h_0, c_0 = state

        # Apply time decay if we have skipped any updates.
        c_0 = lax.cond(skip_count>0,
                       lambda _: self._decomp_and_decay(c_0, skip_count),  # Pass c_0 to the decay function
                       lambda _: c_0,  # Return c_0 as is
                       operand=None)  # The operand is not used in the functions

        # Compute the gates
        i = jax.nn.sigmoid(self.input_linear(x_s))  # Static input gate
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(h_0, self.weight_hh.T) + self.bias
        f, g, o = jnp.split(gates, 3, axis=-1)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update the state
        c_1 = f * c_0 + i * g
        h_1 = o * jnp.tanh(c_1)

        #return state and 0 for skip_count
        return (h_1, c_1), 0
        
    def _is_input_nan(self, x_d):
        # Create a dummy array with the same shape as x_d, filled with NaNs
        nan_array = jnp.full_like(x_d, jnp.nan)
        # Check if x_d is equal to the dummy NaN array
        return jnp.all(jnp.isnan(x_d) == jnp.isnan(nan_array))
        
    def __call__(self, state, x_d, x_s, skip_count):
        # is_nan = jnp.isnan(x_d).any()
        is_nan = self._is_input_nan(x_d)

        state = lax.cond(is_nan,
                         self._skip_update,
                         self._update_cell,
                         operand=(state, x_d, x_s, skip_count))
        
        return state


class TEALSTM(BaseLSTM):
    def __init__(self, dynamic_in_size, static_in_size, out_size, hidden_size, *, key):
        lkey, ckey = jrandom.split(key)
        super().__init__(dynamic_in_size, out_size, hidden_size, key=lkey)
        self.cell = TEALSTMCell(dynamic_in_size, static_in_size, hidden_size, out_size, key=ckey)

    def __call__(self, x_d, x_s):
        def scan_fn(state, x_d):
            skip_count = state[2]
            new_state, skip_count = self.cell(state[:2], x_d, x_s, skip_count)
            return (*new_state, skip_count), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), 0)
        (out, _, _), _ = jax.lax.scan(scan_fn, init_state, x_d)
        return out


class TAPLSTM(eqx.Module):
    """
    A Time-Attentitive Parallel LSTM model structure
    """
    tealstm_d: TEALSTM
    tealstm_i: TEALSTM
    attention_lambda: jnp.ndarray
    linear: eqx.nn.Linear

    def __init__(self, daily_in_size, irregular_in_size, static_in_size, out_size, hidden_size, *, key):
        dkey, ikey, lkey = jrandom.split(key, 3)
        self.tealstm_d = TEALSTM(daily_in_size, static_in_size, out_size, hidden_size, key=dkey)
        self.tealstm_i = TEALSTM(irregular_in_size, static_in_size, out_size, hidden_size, key=ikey)
        self.attention_lambda = jnp.array([0.1])
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)

    def __call__(self, data):
        d_out = self.tealstm_d(data['x_dd'], data['x_s'])
        i_out = self.tealstm_i(data['x_di'], data['x_s'])

        # Combine the outputs from both LSTM branches using attention
        a_i = jnp.exp(-self.attention_lambda * data['attn_dt'][-1])
        a_d = jnp.ones_like(a_i)
        a_normalized = jax.nn.softmax(jnp.stack([a_d, a_i], axis=-1), axis=-1)

        combined_outs = a_normalized[..., 0] * d_out + a_normalized[..., 1] * i_out

        return self.linear(combined_outs)
    
