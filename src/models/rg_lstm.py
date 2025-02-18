import equinox as eqx
import jax
import jax.numpy as jnp
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
    dense: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, graph_matrix, *, key):
        self.graph_matrix = self._calc_adjacency(graph_matrix)
        self.num_graph_nodes = graph_matrix.shape[0]
        self.hidden_size = hidden_size

        keys = jax.random.split(key, 3)
        self.cell = eqx.nn.LSTMCell(input_size, hidden_size, key=keys[0])
        self.q_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.dense = eqx.nn.Linear(hidden_size, output_size, key=keys[2])

    def _calc_adjacency(self, dist):
        dist = jnp.array(dist)
        dist = jnp.where(dist == 0, jnp.nan, dist)
        dist_norm = (dist - jnp.nanmean(dist)) / jnp.nanstd(dist)
        A = 1 / (1 + jnp.exp(dist_norm))
        A = jnp.where(jnp.isnan(A), 0, A)
        return A

    def _transfer(self, x):
        return jax.nn.tanh(self.q_proj(x))

    def __call__(self, x_d, x_s):

        def scan_fn(state, x_d_t):
            #Graph convolution has to happen outside cell vmap
            q = jax.vmap(self._transfer)(state[0])
            # We have to use stop_gradient to avoid updating the graph matrix.
            # I had problems using regular numpy arrays stopping gradients of q.
            c0_t = state[1] + jax.lax.stop_gradient(self.graph_matrix.T) @ q

            x = jnp.concat([x_d_t, x_s], axis=1)
            new_state = jax.vmap(self.cell)(x, (state[0], c0_t))
            return new_state, new_state

        init_state = (jnp.zeros((self.num_graph_nodes, self.hidden_size)),) * 2  # Tuple of h, c
        final_state, all_states = jax.lax.scan(scan_fn, init_state, x_d)

        out = jax.vmap(self.dense)(final_state[0])

        return out


class Graph_LSTM(eqx.Module):
    rg_lstm: RG_LSTM
    target: list

    def __init__(self, *, target: list, dynamic_size: int, static_size: int, hidden_size: int, graph_matrix: np.array,
                 seed: int, dropout: float):

        key = jax.random.PRNGKey(seed)
        self.rg_lstm = RG_LSTM(dynamic_size + static_size, hidden_size, len(target), graph_matrix, key=key)

        self.target = target

    def __call__(self, data, keys):
        return self.rg_lstm(data['dynamic']['era5'], data['static'])
