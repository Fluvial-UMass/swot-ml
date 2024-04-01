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
        return jax.nn.relu(self.linear(out))


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
        return jax.nn.relu(self.linear(out))


class MTLSTM(BaseLSTM):
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

class EAMTLSTM(BaseLSTM):
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
        return jax.nn.relu(self.linear(out))

class MTLSTMCell(eqx.Module):
    """
    A Split Time-Aware LSTM (MTLSTM) cell implemented using Equinox. This cell can 
    accomodate different observation frequencies for each data feature

    Attributes:
        hidden_size (int): The size of the hidden state in the MTLSTM cell.
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
        
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(hidden, self.weight_hh.T) + self.bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c_star = self._decomp_and_discount(cell, dt)

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
        cs_list = []
        c_list = []
        for i in range(self.input_size):
            cs = jnp.tanh(jnp.dot(cell, self.weight_decomp[i].T) + self.bias_decomp[i])
            c = cs * self._time_decay(dt[i]) 
            cs_list.append(cs)
            c_list.append(c)

        ct = cell - jnp.sum(jnp.stack(cs_list, axis=0), axis=0)
        c_star = ct + jnp.sum(jnp.stack(c_list, axis=0), axis=0)

        return c_star
    
    def _time_decay(self, dt):
        # Time decay function to discount the short-term memory based on elapsed time.
        return 1.0 / jnp.log(jnp.e + dt)
    
    
class EAMTLSTMCell(MTLSTMCell):
    """
    An Entity-Aware Multi-Timescale LSTM (EA-MTLSTM) cell that extends the MTLSTMCell.
    
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
