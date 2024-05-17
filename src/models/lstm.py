import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class BaseLSTM(eqx.Module):
    """
    Base class for LSTM models in the Equinox framework.

    Attributes:
        hidden_size (int): Size of the hidden state.
        cell (eqx.Module): LSTM cell module.
        dense (eqx.nn.Linear or function): Linear layer for output transformation if `dense=True`,
                                            otherwise a lambda function that returns its input.
    """
    hidden_size: int
    cell: eqx.Module
    dropout: eqx.nn.Dropout
    dense: eqx.nn.Linear = None

    def __init__(self, in_size, out_size, hidden_size, *, key, **kwargs):
        """
        Initializes the BaseLSTM model.
    
        Args:
            in_size (int): Size of the input features.
            out_size (int): Size of the output features.
            hidden_size (int): Size of the hidden state in the LSTM cell.
            key (jax.random.PRNGKey): Random key for parameter initialization.
            dropout (float, default 0): Fraction of neurons to reset to 0 during training. 
            dense (bool, default True): If True, a linear (dense) layer is added for output transformation.
        """
        dropout = kwargs.get('dropout', 0)
        dense = kwargs.get('dense', True)
        
        ckey, lkey = jrandom.split(key,2)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.dropout = eqx.nn.Dropout(dropout)
        if dense:
            self.dense = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, data, key):
        raise NotImplementedError("Subclasses must implement this method.")


class LSTM(BaseLSTM):
    """
    Standard LSTM model built on the BaseLSTM class.
    """
    def __init__(self, in_size, out_size, hidden_size, *, key, **kwargs):
        super().__init__(in_size, out_size, hidden_size, key=key, **kwargs)

    def __call__(self, data, key):
        def scan_fn(state, xd):
            return self.cell(xd, state), None
        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, data['x_dd'])

        if self.dense is not None:
            out = self.dense(out)
        return out
        

class EALSTMCell(eqx.Module):
    """
    Am Entity-Aware LSTM (TEALSTM) cell for processing time series data with
    dynamic and static features. This cell modified the classic LSTM cell by:
    1. Controlling the input gate with an array of static entity features.

    Attributes:
        weight_ih (jax.Array): Input weights for input, forget, and output gates.
        weight_hh (jax.Array): Hidden weights for input, forget, and output gates.
        bias (jax.Array): Bias terms for input, forget, and output gates.
        input_linear (eqx.nn.Linear): Linear transformation for static input features.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: eqx.nn.Linear
    
    def __init__(self, dynamic_input_size, static_input_size, out_size, hidden_size, *, key):
        wkey, bkey, ikey, dkey = jrandom.split(key, 4)
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, dynamic_input_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (3 * hidden_size,))
        self.input_linear = eqx.nn.Linear(static_input_size, hidden_size, use_bias=True, key=ikey)
        
    def __call__(self, state, x_d, i):
        """
        Forward pass of the TEALSTMCell module.

        Args:
            state (tuple): Tuple containing the hidden and cell states.
            x_d (jax.Array): Dynamic input features.
            i (jax.Array): Static feature input gate

        Returns:
            Tuple of updated hidden and cell states.
        """
        h_0, c_0 = state

        # Compute the gates
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(h_0, self.weight_hh.T) + self.bias
        f, g, o = jnp.split(gates, 3, axis=-1)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update the state
        c_1 = f * c_0 + i * g
        h_1 = o * jnp.tanh(c_1)

        #return state and 0 for skip_count
        return (h_1, c_1)

class EALSTM(BaseLSTM):
    """
    Entity-Aware LSTM (TEALSTM) model for processing time series data with dynamic and static features.
    """
    def __init__(self, dynamic_in_size, static_in_size, out_size, hidden_size, *, key, **kwargs):
        super().__init__(dynamic_in_size, out_size, hidden_size, key=key, **kwargs)
        self.cell = EALSTMCell(dynamic_in_size, static_in_size, out_size, hidden_size, key=key)

    def __call__(self, data):
        """
        Forward pass of the TEALSTM module.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM module and final skip count.
        """
        # Input gate is based on static watershed features
        i = jax.nn.sigmoid(self.cell.input_linear(data['x_s']))
        
        def scan_fn(state, x_d, key):
            new_state = self.cell(state, x_d, i)
            return new_state, None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, data['x_dd'])

        if self.dense is not None:
            out = self.dense(out)
        return out

        
class TEALSTMCell(eqx.Module):
    """
    A Time- and Entity-Aware LSTM (TEALSTM) cell for processing time series data with
    dynamic and static features. This cell modified the classic LSTM cell by:
    1. Controlling the input gate with an array of static entity features.
    2. Skipping cell updates when the inputs are missing. When updating the cell, the
        previous cell state is then decayed according to the number of skips.

    Attributes:
        weight_ih (jax.Array): Input weights for input, forget, and output gates.
        weight_hh (jax.Array): Hidden weights for input, forget, and output gates.
        bias (jax.Array): Bias terms for input, forget, and output gates.
        input_linear (eqx.nn.Linear): Linear transformation for static input features.
        weight_decomp (jax.Array): Weights for decomposing the cell state.
        bias_decomp (jax.Array): Bias for decomposing the cell state.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: eqx.nn.Linear
    weight_decomp: jax.Array
    bias_decomp: jax.Array
    
    def __init__(self, dynamic_in_size, static_in_size, out_size, hidden_size, *, key):
        wkey, bkey, ikey, dkey = jrandom.split(key, 4)
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, dynamic_in_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (3 * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (3 * hidden_size,))
        self.input_linear = eqx.nn.Linear(static_in_size, hidden_size, use_bias=True, key=ikey)
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
        state, x_d, i, skip_count = operand
        h_0, c_0 = state

        # Apply time decay if we have skipped any updates.
        # c_0 = self._decomp_and_decay(c_0, skip_count)
        c_0 = lax.cond(skip_count>0,
                       lambda _: self._decomp_and_decay(c_0, skip_count),  # Pass c_0 to the decay function
                       lambda _: c_0,  # Return c_0 as is
                       operand=None)  # The operand is not used in the functions

        # Compute the gates
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(h_0, self.weight_hh.T) + self.bias
        f, g, o = jnp.split(gates, 3, axis=-1)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update the state
        c_1 = f * c_0 + i * g
        h_1 = o * jnp.tanh(c_1)

        #return state and 0 for skip_count
        skip_count = 0
        return (h_1, c_1), skip_count
        
    def __call__(self, state, x_d, i, skip_count):
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

        state = lax.cond(is_nan,
                         self._skip_update,
                         self._update_cell,
                         operand=(state, x_d, i, skip_count))
        
        return state


class TEALSTM(BaseLSTM):
    """
    Time- and Entity-Aware LSTM (TEALSTM) model for processing time series data with
    dynamic and static features.
    """
    
    def __init__(self, dynamic_in_size, static_in_size, out_size, hidden_size, *, key, **kwargs):
        super().__init__(dynamic_in_size, out_size, hidden_size, key=key, **kwargs)
        self.cell = TEALSTMCell(dynamic_in_size, static_in_size, out_size, hidden_size, key=key)

    def __call__(self, data, key):
        """
        Forward pass of the TEALSTM module.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM module and final skip count.
        """
        # Input gate is based on static watershed features
        i = jax.nn.sigmoid(self.cell.input_linear(data['x_s']))
        
        def scan_fn(state, x_d):
            skip_count = state[2]
            new_state, skip_count = self.cell(state[:2], x_d, i, skip_count)
            return (*new_state, skip_count), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), int(0))
        (out, _, skip_count), _ = jax.lax.scan(scan_fn, init_state, data['x_di'])

        if self.dense is not None:
            out = self.dense(out)
        return out, skip_count


class TAPLSTM(eqx.Module):
    """
    Time-Attentive Parallel LSTM (TAPLSTM) model for processing daily and irregular time series data
    with attention mechanism to combine outputs from both branches.

    Attributes:
        tealstm_d (TEALSTM): TEALSTM model for daily time series data.
        tealstm_i (TEALSTM): TEALSTM model for irregular time series data.
        attention_lambda (jnp.ndarray): Attention decay parameter.
        dense (eqx.nn.Linear): Linear layer for output transformation.
        dropout (eqx.nn.Dropout): Optional dropout layer for entire model.
    """
    ealstm_d: EALSTM
    tealstm_i: TEALSTM
    # mlp_attention: eqx.nn.MLP
    # dense: eqx.nn.Linear
    dense: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(self, *, daily_in_size, irregular_in_size, static_in_size, out_size, hidden_size, seed, dropout):
        key = jax.random.PRNGKey(seed)
        keys = jrandom.split(key, 4)
        self.ealstm_d = EALSTM(daily_in_size, static_in_size, out_size, hidden_size, 
                               key=keys[0], dropout=0, dense=False)
        self.tealstm_i = TEALSTM(irregular_in_size, static_in_size, out_size, hidden_size, 
                                 key=keys[1], dropout=0, dense=False)
        
        # self.mlp_attention = eqx.nn.MLP(in_size=static_in_size, out_size=1, width_size=hidden_size, 
        #                                 depth=3, key=keys[2])
        # self.dense = eqx.nn.Linear(2 * hidden_size, out_size, use_bias=True, key=keys[3])
        
        self.dense = eqx.nn.MLP(in_size=(2 * hidden_size) + static_in_size + 1, 
                                       out_size=out_size, 
                                       width_size=hidden_size, 
                                       depth=2, 
                                       key=keys[2])
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, data, key):
        d_out = self.ealstm_d(data)
        i_out, skip_count = self.tealstm_i(data)
        dt = skip_count + 1 # skip = 0 is still a dt of 1

        # # Compute the attention decay parameter based on static features
        # attention_lambda = self.mlp_attention(data['x_s'])

        # # Compute the raw attention weights
        # a_i = jnp.exp(-attention_lambda * dt)
        # a_d = jnp.ones_like(a_i)

        # # Normalize the attention weights
        # weight_sum = a_i + a_d
        # a_i_normalized = a_i / weight_sum
        # a_d_normalized = a_d / weight_sum
    
        # # Apply the normalized attention weights to the outputs from both LSTM branches
        # weighted_d_out = a_i_normalized * d_out
        # weighted_i_out = a_d_normalized * i_out
        
        # # jax.debug.print("dt: {a}, weight:{b}", a=dt, b=a_d_normalized)
        # # Concatenate the weighted outputs and pass through the final linear layer
        # final_out = self.dense(jnp.concatenate([weighted_d_out, weighted_i_out], axis=-1))

        # jax.debug.print("d_out:{a}\ni_out:{b}\nx_s:{c}\ndt:{d}", a=d_out.shape, b=i_out.shape, c=data['x_s'].shape, d=jnp.array([dt]).shape)
        combined_input = jnp.concatenate([d_out, i_out, data['x_s'], jnp.array([dt])])
        final_out = self.dense(combined_input)

        return final_out







