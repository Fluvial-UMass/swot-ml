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


class TLSTM(BaseLSTM):
    def __init__(self, in_size, out_size, hidden_size, *, key):
        super().__init__(in_size, out_size, hidden_size, key=key)
        ckey, _ = jrandom.split(key)
        self.cell = MTLSTMCell(in_size, hidden_size, key=ckey)

    def __call__(self, data):
        def scan_fn(state, inputs):
            xd, dt = inputs
            return self.cell(xd, dt, state), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, (data['xd'], data['dt']))
        return self.linear(out)

class TEALSTM(BaseLSTM):
    def __init__(self, in_size, out_size, hidden_size, *, key):
        super().__init__(in_size, out_size, hidden_size, key=key)
        ckey, _ = jrandom.split(key)
        self.cell = MTLSTMCell(in_size, hidden_size, key=ckey)

    def __call__(self, data):
        def scan_fn(state, inputs):
            xd, dt = inputs
            return self.cell(xd, state, dt), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, (data['xd'], data['dt']))
        return (self.linear(out))

class TAPLSTM(eqx.Module):
    """
    A Time-Attentitive Parallel LSTM model structure
    """
    def __init__(self, d_in_size, i_in_size, out_size, d_hidden_size, i_hidden_size, *, key):
        self.d_in_size = d_in_size
        self.d_hidden_size = d_hidden_size
        dkey, dlinkey = jrandom.split(key)
        self.d_cell = TLSTMCell(d_in_size, d_hidden_size, key=dkey)
        self.d_linear = eqx.nn.Linear(d_hidden_size, out_size, use_bias=True, key=dlinkey)
    
        self.i_in_size = i_in_size
        self.i_hidden_size = i_hidden_size
        ikey, ilinkey = jrandom.split(key)
        self.i_cell = TEALSTMCell(i_in_size, i_hidden_size, key=ikey)
        self.i_linear = eqx.nn.Linear(i_hidden_size, out_size, use_bias=True, key=ilinkey)
    
        self.out_size = out_size

    def __call__(self, data):
        def scan_fn(state, inputs):
            d_state, i_state = state
            x_d, x_di, x_di_dt, x_s = inputs
    
            d_hidden, d_cell = self.d_cell(x_d, d_state)
            i_hidden, i_cell = self.i_cell(x_d, i_state, dt, x_s)
    
            d_out = self.d_linear(d_hidden)
            i_out = self.i_linear(i_hidden)
    
            return (d_hidden, d_cell), (i_hidden, i_cell), (d_out, i_out)
    
        d_init_state = (jnp.zeros(self.d_hidden_size), jnp.zeros(self.d_hidden_size))
        i_init_state = (jnp.zeros(self.i_hidden_size), jnp.zeros(self.i_hidden_size))
    
        (_, _), (_, _), (d_outs, i_outs) = jax.lax.scan(
            scan_fn, (d_init_state, i_init_state), (data['x_d'], data['x_di'], data['x_di_dt'], data['x_s'])
        )
    
        # Combine the outputs from both LSTM branches using attention
        a_d = jnp.exp(-data['lambda'] * data['dt'])
        a_s = jnp.ones_like(a_d)
        a_normalized = jax.nn.softmax(jnp.stack([a_d, a_s], axis=-1), axis=-1)
    
        combined_outs = a_normalized[..., 0] * d_outs + a_normalized[..., 1] * i_outs
    
        return combined_outs


class TLSTMCell(eqx.Module):
    """
    A Time-Aware LSTM (TLSTM) cell implemented using Equinox.

    Attributes:
        hidden_size (int): The size of the hidden state in the TLSTM cell.
        input_size (int): The size of the input features.
        input_size (int): The number of input features with different frequencies.
        weight_ih (jax.Array): Input weights for input, forget, cell, and output gates.
        weight_hh (jax.Array): Hidden weights for input, forget, cell, and output gates.
        bias (jax.Array): Bias terms for input, forget, cell, and output gates.
        weight_decomp (jax.Array): Weights for subspace decomposition of each feature.
        bias_decomp (jax.Array): Biases for subspace decomposition of each feature.
    """

    hidden_size: int
    input_size: int
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    weight_decomp: jax.Array
    bias_decomp: jax.Array

    def __init__(self, input_size, hidden_size, *, key):
        """
        Initializes the MTLSTM cell.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state in the MTLSTM cell.
            key (jax.random.PRNGKey): A random key for initializing the cell parameters.
        """
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        wkey, bkey, dkey, tkey = jrandom.split(key, 4)
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (4 * hidden_size, input_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (4 * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(key, (4 * hidden_size,))
        self.weight_decomp = jax.nn.initializers.glorot_normal()(dkey, (input_size, hidden_size, hidden_size))
        self.bias_decomp = jax.nn.initializers.zeros(dkey, (input_size, hidden_size))

    def __call__(self, x_d, dt, state):
        """
        Performs a single step of the MTLSTM cell.

        Args:
            x_d (jax.Array): The dynamic input features at the current time step.
            state (tuple): A tuple of (hidden_state, cell_state) representing the previous state.
            dt (jax.Array): The elapsed times for each feature since the last observation.

        Returns:
            tuple: A tuple of (hidden_state, cell_state) representing the updated state.
        """
        hidden, cell = state

        c_star = self._decomp_and_discount(cell, dt)
        
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(hidden, self.weight_hh.T) + self.bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        cell = f * c_star + i * g
        hidden = o * jnp.tanh(cell)

        return hidden, cell
  
    def _decomp_and_discount(self, cell, dt):
        """
        Performs subspace decomposition and elapsed time discounting for each feature.

        Args:
            cell (jax.Array): The cell state.
            dt (jax.Array): The elapsed times for each feature since the last observation.

        Returns:
            jax.Array: The updated cell state after subspace decomposition and discounting.
        """

        # Subspace decomposition and elapsed time discounting for each feature
        cs_hat_0_list = []
        ct_0_list = []
        for i in range(self.input_size):
            cs_0 = jnp.tanh(jnp.dot(cell, self.weight_decomp[i].T) + self.bias_decomp[i])
            cs_hat_0 = cs_0 * self._time_decay(dt[i]) 
            cs_hat_0_list.append(cs_hat_0)

            ct_0 = cell - cs_0
            ct_0_list.append(ct_0)
        
        ct_0 = jnp.sum(jnp.stack(ct_0_list, axis=0), axis=0)
        cs_hat_0 = jnp.sum(jnp.stack(cs_hat_0_list, axis=0), axis=0)
        c_star = ct_0 + cs_hat_0

        return c_star
    
    def _time_decay(self, dt):
        # Time decay function to discount the short-term memory based on elapsed time.
        return 1.0 / jnp.log(jnp.e + dt)
    
    
class TEALSTMCell(MTLSTMCell):
    """
    A Time- and Entity-Aware LSTM (TEALSTM) cell that extends the TLSTMCell.
    
    Attributes:
        weight_is (jax.Array): Weight matrix for the static input gate.
        bias_is (jax.Array): Bias vector for the static input gate.
    """

    weight_is: jax.Array
    bias_is: jax.Array

    def __init__(self, input_size, hidden_size, *, key):
        """
        Initializes the EA-MTLSTM cell.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state in the EA-MTLSTM cell.
            key (jax.random.PRNGKey): A random key for initializing the cell parameters.
        """
        super().__init__(input_size, hidden_size, key=key)
        
        iskey, _ = jrandom.split(key)
        self.weight_is = jax.nn.initializers.glorot_normal()(iskey, (hidden_size, input_size))
        self.bias_is = jax.nn.initializers.zeros(iskey, (hidden_size,))

    def __call__(self, x_d, state, dt, x_s):
        """
        Performs a single step of the EA-MTLSTM cell.

        Args:
            x_d (jax.Array): The dynamic input features at the current time step.
            state (tuple): A tuple of (hidden_state, cell_state) representing the previous state.
            dt (jax.Array): The elapsed times for each feature since the last observation.
            x_s (jax.Array): The static input features.

        Returns:
            tuple: A tuple of (hidden_state, cell_state) representing the updated state.
        """
        hidden, cell = state
        
        # Calculate the static input gate
        i_s = jax.nn.sigmoid(jnp.dot(x_s, self.weight_is.T) + self.bias_is)
        
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(hidden, self.weight_hh.T) + self.bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c_star = self._decomp_and_discount(cell, dt)

        # Modify the cell state update to use the static input gate
        cell = f * c_star + i * g * i_s 
        hidden = o * jnp.tanh(cell)

        return hidden, cell
