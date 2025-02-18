import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Optional


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

    def __init__(self, in_size, hidden_size, dense_size, dropout, *, key):
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
        ckey, lkey = jrandom.split(key, 2)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.dropout = eqx.nn.Dropout(dropout)
        if dense_size is not None:
            self.dense = eqx.nn.Linear(hidden_size, dense_size, key=lkey)
        else:
            self.dense = None

    def __call__(self, data, key):
        raise NotImplementedError("Subclasses must implement this method.")


class LSTM(BaseLSTM):
    """
    Standard LSTM model built on the BaseLSTM class.
    """

    def __init__(self, in_size, hidden_size, out_size, *, key, **kwargs):
        super().__init__(in_size, hidden_size, out_size, key=key, **kwargs)

    def __call__(self, x_d, key):

        def scan_fn(state, xd):
            return self.cell(xd, state), None

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, x_d)

        if self.dense is not None:
            out = self.dense(out)
        return out


class EALSTMCell(eqx.Module):
    """
    A configurable LSTM cell that can include entity-aware modifications
    based on the provided options.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: Optional[eqx.nn.Linear]
    entity_aware: bool = eqx.field(static=True)

    def __init__(self, dynamic_in_size: int, static_in_size: int, hidden_size: int, entity_aware: bool = True, *, key):

        wkey, bkey, ikey = jrandom.split(key, 3)
        self.entity_aware = entity_aware

        if self.entity_aware:
            num_gates = 3
            self.input_linear = eqx.nn.Linear(static_in_size, hidden_size, use_bias=True, key=ikey)
        else:
            num_gates = 4
            self.input_linear = None
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, dynamic_in_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (num_gates * hidden_size,))

    def __call__(self, state, x_d, i):
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
        h_0, c_0 = state

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

        return h_1, c_1


class EALSTM(BaseLSTM):
    entity_aware: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    """
    Entity-Aware LSTM (TEALSTM) model for processing time series data with
    dynamic and static features.
    """

    def __init__(self,
                 dynamic_in_size: int,
                 static_in_size: int,
                 hidden_size: int,
                 dense_size: int,
                 dropout: float,
                 return_all: bool = False,
                 *,
                 key):

        super().__init__(dynamic_in_size, hidden_size, dense_size, dropout, key=key)
        self.entity_aware = static_in_size > 0
        self.return_all = return_all

        self.cell = EALSTMCell(dynamic_in_size, static_in_size, hidden_size, self.entity_aware, key=key)

    def __call__(self, x_d, x_s, key):
        """
        Forward pass of the EALSTM.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Final state, all states, or densified final state depending on config. 
        """
        if self.entity_aware:
            # Input gate is based on static watershed features
            i = jax.nn.sigmoid(self.cell.input_linear(x_s))
        else:
            # Input gate is calculated from dynamic data per normal LSTM
            i = None

        def scan_fn(state, x):
            new_state = self.cell(state, x, i)
            return new_state, new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (final_state, _), all_states = jax.lax.scan(scan_fn, init_state, x_d)

        out = all_states if self.return_all else final_state
        out = self.dropout(out, key=key)

        if self.dense is not None:
            out = self.dense(out)

        return out


class TEALSTMCell(eqx.Module):
    """
    A configurable LSTM cell that can include time- and/or entity-aware modifications
    based on the provided options.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: Optional[eqx.nn.Linear]
    weight_decomp: Optional[jax.Array]
    bias_decomp: Optional[jax.Array]
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)

    def __init__(self,
                 dynamic_in_size: int,
                 static_in_size: int,
                 hidden_size: int,
                 time_aware: bool = True,
                 entity_aware: bool = True,
                 *,
                 key):

        wkey, bkey, ikey, dkey = jrandom.split(key, 4)
        self.time_aware = time_aware
        self.entity_aware = entity_aware

        if self.entity_aware:
            num_gates = 3
            self.input_linear = eqx.nn.Linear(static_in_size, hidden_size, use_bias=True, key=ikey)
        else:
            num_gates = 4
            self.input_linear = None
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, dynamic_in_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (num_gates * hidden_size,))

        if self.time_aware:
            self.weight_decomp = jax.nn.initializers.glorot_normal()(dkey, (hidden_size, hidden_size))
            self.bias_decomp = jax.nn.initializers.zeros(dkey, (hidden_size,))
        else:
            self.weight_decomp = None
            self.bias_decomp = None

    def _decomp_and_decay(self, c_0, skip_count):
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
            operand=None)  # The operand is not used in the functions

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

        #return state and 0 for skip_count
        skip_count = 0
        return (h_1, c_1), skip_count

    def __call__(self, state, x_d, x_s, skip_count):
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

        state = lax.cond(is_nan, self._skip_update, self._update_cell, operand=(state, x_d, x_s, skip_count))

        return state


class TEALSTM(BaseLSTM):
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    """
    Time- and Entity-Aware LSTM (TEALSTM) model for processing time series data with
    dynamic and static features.
    """

    def __init__(self,
                 dynamic_in_size: int,
                 static_in_size: int,
                 hidden_size: int,
                 dense_size: int,
                 dropout: float,
                 time_aware: bool = True,
                 return_all: bool = False,
                 *,
                 key):

        super().__init__(dynamic_in_size, hidden_size, dense_size, dropout, key=key)
        self.time_aware = time_aware
        self.entity_aware = static_in_size > 0
        self.return_all = return_all

        self.cell = TEALSTMCell(dynamic_in_size, static_in_size, hidden_size, time_aware, self.entity_aware, key=key)

    def __call__(self, x_d, x_s, dt, key):
        """
        Forward pass of the TEALSTM.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM and final skip count.
        """
        if self.entity_aware:
            # Input gate is based on static watershed features
            i = jax.nn.sigmoid(self.cell.input_linear(x_s))
        else:
            # Input gate is calculated from dynamic data per normal LSTM
            i = None

        def scan_fn(state, x):
            skip_count = state[2]
            new_state, skip_count = self.cell(state[:2], x, i, skip_count)
            return (*new_state, skip_count), new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), int(0))
        (final_state, _, skip_count), all_states = jax.lax.scan(scan_fn, init_state, x_d)

        out = all_states if self.return_all else final_state
        out = self.dropout(out, key=key)

        if self.dense is not None:
            out = self.dense(out)

        return out


class IEALSTMCell(eqx.Module):
    """
    A configurable LSTM cell that includes interval- and entity-aware modifications.
    """
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)

    def __init__(self,
                 dynamic_in_size: int,
                 static_in_size: int,
                 hidden_size: int,
                 time_aware: bool = True,
                 entity_aware: bool = True,
                 *,
                 key):

        wkey, bkey = jax.random.split(key, 2)

        self.entity_aware = entity_aware
        self.time_aware = time_aware

        num_gates = 3 if self.entity_aware else 4
        self.weight_ih = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, dynamic_in_size))
        self.weight_hh = jax.nn.initializers.glorot_normal()(wkey, (num_gates * hidden_size, hidden_size))
        self.bias = jax.nn.initializers.zeros(bkey, (num_gates * hidden_size,))

    def __call__(self, op):
        """
        Forward pass of the IEALSTMCell module.

        Args:
            state (tuple): Tuple containing the hidden and cell states.
            x_d (jax.Array): Dynamic input features.
            x_s (jax.Array): Static input features.
            dt (jax.Array): Time interval since last observations.

        Returns:
            Updated state
        """
        state, x_d, i, decay_weight = op
        h_0, c_0 = state

        # Gates calculation with added static input and time decay
        gates = jnp.dot(x_d, self.weight_ih.T) + jnp.dot(h_0, self.weight_hh.T) + self.bias
        if self.entity_aware:
            f, g, o = jnp.split(gates, 3, axis=-1)
        else:
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = jax.nn.sigmoid(i)

        f = jax.nn.sigmoid(f) * decay_weight
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update the state
        c_1 = f * c_0 + i * g
        h_1 = o * jnp.tanh(c_1)

        return h_1, c_1


class IEALSTM(BaseLSTM):
    input_linear: Optional[eqx.nn.Linear]
    decay_linear: Optional[eqx.nn.Linear]
    decay_weights: Optional[jax.Array]
    time_aware: bool = eqx.field(static=True)
    entity_aware: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)
    """
    Time- and Entity-Aware LSTM (TEALSTM) model for processing time series data with
    dynamic and static features.
    """

    def __init__(self,
                 dynamic_in_size: int,
                 static_in_size: int,
                 hidden_size: int,
                 dense_size: int,
                 dropout: float,
                 time_aware: bool = True,
                 return_all: bool = False,
                 *,
                 key):
        skey, ikey, dkey, ckey = jax.random.split(key, 4)

        super().__init__(dynamic_in_size, hidden_size, dense_size, dropout, key=skey)
        self.time_aware = time_aware
        self.entity_aware = static_in_size > 0
        self.return_all = return_all

        self.input_linear = None
        if self.entity_aware:
            self.input_linear = eqx.nn.Linear(static_in_size, hidden_size, use_bias=True, key=ikey)

        self.decay_linear = None
        self.decay_weights = None
        if self.time_aware:
            if self.entity_aware:
                # Linear layer to produce decay parameters based on static data
                self.decay_linear = eqx.nn.Linear(static_in_size, 2 * hidden_size, use_bias=True, key=dkey)
            else:
                self.decay_weights = jax.nn.initializers.glorot_normal()(dkey, (2, hidden_size))

        self.cell = IEALSTMCell(dynamic_in_size, static_in_size, hidden_size, time_aware, self.entity_aware, key=ckey)

    def get_decay_fn(self, x_s):
        if self.entity_aware:
            # Generate decay parameters a and b based on static input x_s
            params = self.decay_linear(x_s)
            a, b = jnp.split(params, 2, axis=-1)
        else:
            # Parameters are learned directly
            a = self.decay_weights[0, :]
            b = self.decay_weights[1, :]
        # Ensure positivity for stability
        a = jax.nn.relu(a)
        b = jax.nn.relu(b)
        return lambda dt: a * jnp.exp(-b * dt)

    def __call__(self, x_d, x_s, dt, key):
        """
        Forward pass of the TEALSTM.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM and final skip count.
        """
        i = jax.nn.sigmoid(self.input_linear(x_s)) if self.entity_aware else None
        decay_fn = self.get_decay_fn(x_s) if self.time_aware else lambda _: 1

        def scan_fn(state, data):
            _x_d, _dt = data
            decay_weight = decay_fn(_dt)

            # if _dt > 0:
            #     new_state = self.cell(state, _x_d, i, decay_weight)
            # else:
            #     new_state = state

            new_state = jax.lax.cond(pred=_dt > 0,
                                     true_fun=self.cell,
                                     false_fun=lambda _: state,
                                     operand=(state, _x_d, i, decay_weight))

            return new_state, new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (final_state, _), all_states = jax.lax.scan(scan_fn, init_state, (x_d, dt))

        out = all_states if self.return_all else final_state
        out = self.dropout(out, key=key)

        if self.dense is not None:
            out = self.dense(out)

        return out
