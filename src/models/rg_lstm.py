import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import numpy as np


class RG_LSTM(eqx.Module):
    """
    Recurrent Graph LSTM model
    """
    graph_matrix: jnp.ndarray
    num_graph_nodes: int
    hidden_size: int
    cell: eqx.nn.LSTMCell
    q_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    dense: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, output_size: int, graph_matrix: Array, dropout: float, *,
                 key: PRNGKeyArray):
        self.graph_matrix = self._calc_adjacency(graph_matrix)
        self.num_graph_nodes = graph_matrix.shape[0]
        self.hidden_size = hidden_size

        keys = jax.random.split(key, 3)
        self.cell = eqx.nn.LSTMCell(input_size, hidden_size, key=keys[0])
        self.q_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout)
        self.dense = eqx.nn.Linear(hidden_size, output_size, key=keys[2])

    def _calc_adjacency(self, dist: Array) -> Array:
        dist = jnp.array(dist)
        dist = jnp.where(dist == 0, jnp.nan, dist)
        dist_norm = (dist - jnp.nanmean(dist)) / jnp.nanstd(dist)
        A = 1 / (1 + jnp.exp(dist_norm))
        A = jnp.where(jnp.isnan(A), 0, A)
        return A

    def _transfer(self, x: Array):
        return jax.nn.tanh(self.q_proj(x))

    def __call__(self, x_d: Array, x_s: Array, *, key: PRNGKeyArray):

        def scan_fn(state: tuple[Array, Array], x_d_t: Array):
            #Graph convolution has to happen outside cell vmap
            q = jax.vmap(self._transfer)(state[0])
            # We have to use stop_gradient to avoid updating the graph matrix.
            # I had problems using regular numpy arrays stopping gradients of q.
            c0_t = state[1] + jax.lax.stop_gradient(self.graph_matrix.T) @ q

            x = jnp.concat([x_d_t, x_s], axis=1)
            new_state = jax.vmap(self.cell)(x, (state[0], c0_t))
            return new_state, new_state

        init_state = (jnp.zeros((self.num_graph_nodes, self.hidden_size)),) * 2  # Tuple of h, c
        (h_final, c_final), all_states = jax.lax.scan(scan_fn, init_state, x_d)

        h_final = self.dropout(h_final, key)
        out = jax.vmap(self.dense)(h_final)

        return out


class Graph_LSTM(eqx.Module):
    rg_lstm: RG_LSTM
    target: list

    def __init__(self, *, target: list, dynamic_size: int, static_size: int, hidden_size: int, graph_matrix: np.array,
                 seed: int, dropout: float):

        key = jax.random.PRNGKey(seed)
        self.rg_lstm = RG_LSTM(dynamic_size + static_size, hidden_size, len(target), graph_matrix, dropout, key=key)

        self.target = target

    def __call__(self, data: dict[str:Array | dict[str:Array]], key: PRNGKeyArray) -> Array:
        return self.rg_lstm(data['dynamic']['era5'], data['static'], key=key)
