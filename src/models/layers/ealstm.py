import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class EALSTMCell(eqx.Module):
    """A configurable LSTM cell that can include entity-aware modifications
    based on the provided options.

    Parameters
    ----------
    dynamic_in_size: int
        The number of dynamic features in the input data.
    static_in_size: int
        The number of static features in the input data.
    hidden_size: int
        The number of hidden units in the LSTM cell.
    entity_aware: bool, optional
        Whether to use entity-aware modifications. Defaults to True.
    key: PRNGKeyArray
        A PRNG key used for initialization.

    Attributes
    ----------
    weight_ih: jax.Array
        The weight matrix for the input-to-hidden connections.
    weight_hh: jax.Array
        The weight matrix for the hidden-to-hidden connections.
    bias: jax.Array
        The bias vector for the LSTM cell.
    input_linear: Optional[eqx.nn.Linear]
        A linear layer used for processing static input features.
    entity_aware: bool
        Whether the cell is entity-aware.
    """

    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array
    input_linear: eqx.nn.Linear | None
    entity_aware: bool = eqx.field(static=True)

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        hidden_size: int,
        entity_aware: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        wkey, bkey, ikey = jrandom.split(key, 3)
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

    def __call__(self, state: tuple[Array, Array], x_d: Array, i: Array | None):
        """
        Forward pass of the EALSTMCell module.

        Parameters
        ----------
        state: tuple[Array, Array]
            Tuple containing the hidden and cell states.
        x_d: Array
            Dynamic input features.
        i: Optional[Array]
            Input gate values (evaluated outside cell)

        Returns
        -------
        h_1: Array
            The updated hidden state.
        c_1: Array
            The updated cell state.
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


class EALSTM(eqx.Module):
    """Entity-Aware LSTM model for dynamic and static features.

    Parameters
    ----------
    dynamic_in_size: int
        The number of dynamic features in the input data.
    static_in_size: int
        The number of static features in the input data.
    hidden_size: int
        The number of hidden units in the LSTM cell.
    dropout: float
        The dropout rate.
    return_all: bool, optional
        Return all hidden states or just the final state. Defaults to False.
    key: PRNGKeyArray
        A PRNG key used for initialization.

    Attributes
    ----------
    entity_aware: bool
        Whether the LSTM is entity-aware.
    return_all: bool
        Whether to return all hidden states or just the final state.
    cell: EALSTMCell
        The LSTM cell used by the model.
    """

    hidden_size: int
    dropout: eqx.nn.Dropout
    cell: EALSTMCell
    entity_aware: bool = eqx.field(static=True)
    return_all: bool = eqx.field(static=True)

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        hidden_size: int,
        *,
        return_all: bool = False,
        dropout: float = 0,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.entity_aware = static_in_size > 0
        self.return_all = return_all
        self.dropout = eqx.nn.Dropout(dropout)

        self.cell = EALSTMCell(
            dynamic_in_size, static_in_size, hidden_size, self.entity_aware, key=key
        )

    def __call__(self, x_d: Array, x_s: Array, key: PRNGKeyArray):
        """
        Forward pass of the EALSTM.

        Parameters
        ----------
        x_d: jax.Array
            Dynamic input features.
        x_s: jax.Array
            Static input features.
        key: PRNGKeyArray
            A PRNG key used for dropout.

        Returns
        -------
        jax.Array
            The output of the model.
            - If `return_all` is True, returns all hidden states.
            - If `return_all` is False, returns the final hidden state.
        """
        i = jax.nn.sigmoid(self.cell.input_linear(x_s)) if self.entity_aware else None

        def scan_fn(state, x):
            new_state = self.cell(state, x, i)
            return new_state, new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        (final_state, _), all_states = jax.lax.scan(scan_fn, init_state, x_d)

        out = all_states if self.return_all else final_state

        return self.dropout(out, key=key)
