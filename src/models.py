import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


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

    def __call__(self, x, state, dt):
        """
        Performs a single step of the MTLSTM cell.

        Args:
            x (jax.Array): The input features at the current time step.
            state (tuple): A tuple of (hidden_state, cell_state) representing the previous state.
            dt (jax.Array): The elapsed times for each feature since the last observation.

        Returns:
            tuple: A tuple of (hidden_state, cell_state) representing the updated state.
        """
        hidden, cell = state
        gates = jnp.dot(x, self.weight_ih.T) + jnp.dot(hidden, self.weight_hh.T) + self.bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Subspace decomposition and elapsed time discounting for each feature
        cs_list = []
        c_list = []
        for i in range(self.input_size):
            cs = jnp.tanh(jnp.dot(cell, self.weight_decomp[i].T) + self.bias_decomp[i])
            c = cs * self._time_decay(dt[i]) 
            cs_list.append(cs)
            c_list.append(c)

        ct = cell - jnp.sum(jnp.stack(cs_list, axis=0), axis=0)
        c_star = ct + jnp.sum(jnp.stack(c_list, axis=0), axis=0)

        cell = f * c_star + i * g
        hidden = o * jnp.tanh(cell)

        return hidden, cell
  
    def _time_decay(self, dt):
        # Time decay function to discount the short-term memory based on elapsed time.
        return 1.0 / jnp.log(jnp.e + dt)
        

class MTLSTM(eqx.Module):
    """
    A Multi Time-Aware LSTM (MTLSTM) model implemented using Equinox.

    Attributes:
        hidden_size (int): The size of the hidden state in the MTLSTM cell.
        cell (MTLSTMCell): The MTLSTM cell used in the model.
        linear (equinox.nn.Linear): A linear layer applied to the output of the MTLSTM cell.
        bias (jax.Array): A bias term added to the output of the linear layer.
    """

    hidden_size: int
    cell: MTLSTMCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        """
        Initializes the MTLSTM model.

        Args:
            in_size (int): The size of the input features.
            out_size (int): The size of the output.
            hidden_size (int): The size of the hidden state in the MTLSTM cell.
            input_size (int): The number of input features with different frequencies.
            key (jax.random.PRNGKey): A random key for initializing the model parameters.
        """
        ckey, lkey, tkey = jrandom.split(key, 3)
        self.hidden_size = hidden_size
        self.cell = MTLSTMCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)

    def yield_params(self):
        params = [self.cell.weight_ih, 
                  self.cell.weight_hh, 
                  self.cell.bias, 
                  self.cell.weight_decomp, 
                  self.cell.bias_decomp,
                  self.linear.weight,
                  self.linear.bias]
        yield from params

    def __call__(self, data):
        """
        Forward pass through the MTLSTM model.

        Args:
            xd (jax.Array): The input sequence with shape [sequence_length, in_size].
            dt (jax.Array): The elapsed times for each feature with shape [sequence_length, input_size].

        Returns:
            jax.Array: The output of the model with shape [sequence_length, out_size].
        """
        def scan_fn(state, inputs):
            xd, dt = inputs
            return self.cell(xd, state, dt), None

        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))

        (out, _), _ = jax.lax.scan(scan_fn, init_state, (data['xd'],data['dt'])) 
        return jax.nn.relu(self.linear(out))



class LSTM(eqx.Module):
    hidden_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, data):
        def scan_fn(state, xd):
            return self.cell(xd, state), None
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, data['xd'])
        return jax.nn.relu(self.linear(out))

    def yield_params(self):
        params = [self.cell.weight_ih, 
                  self.cell.weight_hh, 
                  self.cell.bias, 
                  self.linear.weight,
                  self.linear.bias]
        yield from params


class Gated_LSTM(eqx.Module):
    hidden_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, data):
        def scan_fn(state, inputs):
            x, w = inputs
            (h, c) = state
            # Apply input gate using weights
            input_gate = jax.nn.sigmoid(w)  # Convert weights to gate values between 0 and 1
            gated_x = x * input_gate  # Apply the gate to the input features
            return self.cell(gated_x, state), None  # Use gated input in LSTM cell

        init_state = (jnp.zeros(self.cell.hidden_size), jnp.zeros(self.cell.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, (data['xd'],data['w']))
        return self.linear(out)
        
    
    
