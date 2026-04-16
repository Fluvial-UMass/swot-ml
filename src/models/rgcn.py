from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch
from .base_model import BaseModel
from .layers.static_mlp import StaticMLP


class RGrNLSTMCell(eqx.Module):
    hidden_size: int
    W_ih: eqx.nn.Linear
    W_hh: eqx.nn.Linear
    W_q: eqx.nn.Linear
    W_edge_gate: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key: PRNGKeyArray):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.hidden_size = hidden_size

        # Standard LSTM components
        self.W_ih = eqx.nn.Linear(input_size, 4 * hidden_size, key=k1)
        self.W_hh = eqx.nn.Linear(hidden_size, 4 * hidden_size, key=k2, use_bias=False)

        # Spatial Projection (Query)
        self.W_q = eqx.nn.Linear(hidden_size, hidden_size, key=k3)

        # Edge Gating: Takes (sender_h, receiver_h) to produce a per-edge sigmoid
        self.W_edge_gate = eqx.nn.Linear(2 * hidden_size, hidden_size, key=k4)

    def __call__(
        self,
        x: Array,
        h_prev: Array,
        c_prev: Array,
        senders: Array,
        receivers: Array,
        edge_weights: Array,
        num_nodes: int,
    ) -> tuple[Array, Array, Array]:
        # Raw Messages (Spatial Projection)
        q_prev = jax.vmap(self.W_q)(h_prev)  # [NumNodes, Hidden]

        # Contextualize every edge using current sender and receiver states
        h_senders = h_prev[senders]  # [NumEdges, Hidden]
        h_receivers = h_prev[receivers]  # [NumEdges, Hidden]

        edge_input = jnp.concatenate([h_senders, h_receivers], axis=-1)
        edge_gates = jax.nn.sigmoid(jax.vmap(self.W_edge_gate)(edge_input))

        # Combine: (Message) * (Static Edge Weight) * (Dynamic Edge Gate)
        messages = (q_prev[senders] * edge_weights[:, None]) * edge_gates
        upstream_q = jax.ops.segment_sum(messages, receivers, num_segments=num_nodes)

        # Standard LSTM
        gates_x = jax.vmap(self.W_ih)(x)
        gates_h = jax.vmap(self.W_hh)(h_prev)
        gates = gates_x + gates_h

        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Add spatial information directly into the long-term memory
        c_new = f * (c_prev + upstream_q) + i * g
        h_new = o * jnp.tanh(c_new)

        return h_new, c_new, q_prev


class RGCN(BaseModel):
    backbone_leaves: int
    hidden_size: int
    seq_length: int
    dense_sources: tuple
    sparse_sources: tuple
    static_size: int
    seq2seq: bool
    use_obs_memory: bool
    return_weights: bool

    head: dict
    dense_embedders: dict
    fusion_norm: eqx.nn.LayerNorm
    sparse_embedders: dict
    sparse_gain_nets: dict
    sparse_norms: dict
    rgrn_lstm: RGrNLSTMCell

    def __init__(
        self,
        target: list,
        seq_length: int,
        dense_sizes: dict,
        sparse_sizes: dict,
        static_size: int,
        hidden_size: int,
        assim_sizes: dict,
        seed: int,
        dropout: float,
        head: str,
        seq2seq: bool,
        use_obs_memory: bool,
        return_weights: bool,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        # initializes linear head and target list
        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.backbone_leaves = ["dense_embedders", "fusion_norm", "rgrn_lstm", "head"]
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.dense_sources = tuple(dense_sizes.keys())
        self.sparse_sources = tuple(sparse_sizes.keys())
        self.static_size = static_size
        self.seq2seq = seq2seq
        self.use_obs_memory = use_obs_memory
        self.return_weights = return_weights

        dense_keys = jrandom.split(keys.pop(0), len(self.dense_sources))
        self.dense_embedders = {
            name: eqx.nn.MLP(
                in_size=size, out_size=hidden_size, width_size=hidden_size, depth=1, key=k
            )
            for (name, size), k in zip(dense_sizes.items(), dense_keys)
        }
        self.fusion_norm = eqx.nn.LayerNorm(hidden_size)

        self.rgrn_lstm = RGrNLSTMCell(
            input_size=hidden_size, hidden_size=hidden_size, key=keys.pop(0)
        )

        self.sparse_embedders = {}
        self.sparse_gain_nets = {}
        self.sparse_norms = {}

        # Populate initial assimilators
        temp_model = self
        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sources))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            temp_model = temp_model.add_assimilator(name, in_size, assim_sizes.get(name, {}), key=k)

        self.sparse_sources = temp_model.sparse_sources
        self.sparse_embedders = temp_model.sparse_embedders
        self.sparse_gain_nets = temp_model.sparse_gain_nets
        self.sparse_norms = temp_model.sparse_norms

    def add_assimilator(
        self, name: str, n_features: int, size_args: dict = {}, *, key: PRNGKeyArray = None
    ) -> "RGCN":
        key = key if key is not None else jrandom.PRNGKey(0)
        k1, k2 = jrandom.split(key)

        embed_width = size_args.get("embed_width", self.hidden_size)
        embed_depth = size_args.get("embed_depth", 2)

        # Assumes StaticMLP is defined elsewhere as in the original code
        new_embedder = StaticMLP(
            dynamic_in_size=n_features,
            static_in_size=self.static_size,
            out_size=self.hidden_size,
            width_size=embed_width,
            depth=embed_depth,
            key=k1,
        )

        gain_width = size_args.get("gain_width", self.hidden_size)
        gain_depth = size_args.get("gain_depth", 2)
        gain_in_size = 2 * self.hidden_size + int(self.use_obs_memory)

        new_gain_net = eqx.nn.MLP(
            in_size=gain_in_size,
            out_size=self.hidden_size,
            width_size=gain_width,
            depth=gain_depth,
            key=k2,
        )

        final_linear = new_gain_net.layers[-1]
        new_bias = jnp.ones_like(final_linear.bias) * 2.0
        new_gain_net = eqx.tree_at(lambda m: m.layers[-1].bias, new_gain_net, new_bias)
        new_norm = eqx.nn.LayerNorm(self.hidden_size)

        new_sparse_sources = tuple(set(self.sparse_sources + (name,)))
        new_embedders = {**self.sparse_embedders, name: new_embedder}
        new_gain_nets = {**self.sparse_gain_nets, name: new_gain_net}
        new_norms = {**self.sparse_norms, name: new_norm}

        return eqx.tree_at(
            lambda m: (m.sparse_sources, m.sparse_embedders, m.sparse_gain_nets, m.sparse_norms),
            self,
            (new_sparse_sources, new_embedders, new_gain_nets, new_norms),
        )

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]

        # Extract sparse edge lists from data
        senders = data.graph_edges[0]
        receivers = data.graph_edges[1]

        # Default to uniform weighting if distance-based weights are not provided
        edge_weights = getattr(data, "edge_weights", jnp.ones_like(senders, dtype=jnp.float32))

        dense_emb_list = [
            jax.vmap(jax.vmap(self.dense_embedders[name]))(data.dynamic[name])
            for name in self.dense_sources
        ]
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)

        obs_emb = {}
        for name in self.sparse_sources:
            features = data.dynamic[name]
            obs_mask = ~jnp.any(jnp.isnan(features), axis=-1, keepdims=True)
            safe_features = jnp.nan_to_num(features)
            emb_vectors = self.sparse_embedders[name](safe_features, data.static)
            obs_emb[name] = (jnp.where(obs_mask, emb_vectors, 0.0), obs_mask)

        obs_memories = (
            self.create_observational_memory(obs_emb, num_locations)
            if self.sparse_sources
            else None
        )

        @jax.checkpoint
        def process_one_timestep(state_prev, scan_slice, senders, receivers, edge_weights):
            dense_input, step_obs_memory, step_key = scan_slice
            h_prev, c_prev = state_prev

            # RGrN forward pass utilizing sparse operations
            h_new, c_new, q_prev = self.rgrn_lstm(
                dense_input, h_prev, c_prev, senders, receivers, edge_weights, num_locations
            )

            if step_obs_memory is not None:
                h_new, assim_trace = self.assimilate(h_new, step_obs_memory)
            else:
                assim_trace = {}

            return (h_new, c_new), (h_new, q_prev, assim_trace)

        initial_state = (
            jnp.zeros((num_locations, self.hidden_size)),
            jnp.zeros((num_locations, self.hidden_size)),
        )

        partial_timestep = partial(
            process_one_timestep, senders=senders, receivers=receivers, edge_weights=edge_weights
        )

        scan_inputs = (dense_emb_norm, obs_memories, jrandom.split(key, self.seq_length))

        final_state, accumulated = jax.lax.scan(partial_timestep, initial_state, scan_inputs)
        all_h, all_q, weights = accumulated

        if self.seq2seq:
            predictions = {
                target: jax.vmap(jax.vmap(head))(all_h) for target, head in self.head.items()
            }
        else:
            predictions = {
                target: jax.vmap(head)(final_state[0]) for target, head in self.head.items()
            }

        # With jax JIT, the returns have no impact on performance unless the model is compiled with this branch.
        if self.return_weights:
            return predictions, weights
        else:
            return predictions

    def create_observational_memory(self, obs_emb, num_locations):
        def memory_step(
            prev_memory: dict[str, Array], updates_masks: dict[str, tuple[Array, Array]]
        ):
            """One step of obs memory tracking."""
            new_memory = {}
            for name, (update_emb_t, mask_t) in updates_masks.items():
                prev_obs, prev_stale, prev_seen = prev_memory[name]

                # Reset if new obs, otherwise carry forward + increment staleness
                updated_obs = jnp.where(mask_t, update_emb_t, prev_obs)
                updated_staleness = jnp.where(mask_t.squeeze(axis=-1), 0.0, prev_stale + 1.0)

                # Update seen flag. Once true, it stays true.
                has_been_seen = prev_seen | mask_t.squeeze(axis=-1)

                new_memory[name] = (updated_obs, updated_staleness, has_been_seen)
            return new_memory, new_memory  # carry + collect for whole sequence

        initial_obs_memory = {
            name: (
                jnp.zeros((num_locations, self.hidden_size)),  # last_obs
                jnp.ones((num_locations)),  # staleness
                jnp.zeros((num_locations), dtype=jnp.bool_),  # has_been_observed
            )
            for name in self.sparse_sources
        }
        _, obs_memories = jax.lax.scan(memory_step, initial_obs_memory, obs_emb)

        return obs_memories

    def assimilate(self, h: Array, obs_memory: dict[str, tuple[Array, Array]]):
        """Assimilate using observation memory with staleness tracking."""
        tracing_weights = {}
        for name, (last_obs, staleness, has_been_seen) in obs_memory.items():
            # Normalize observation embedding
            update_emb = jax.vmap(self.sparse_norms[name])(last_obs)
            nu = update_emb - h  # Innovation
            nu_normed = jax.vmap(self.sparse_norms[name])(nu)

            if self.use_obs_memory:
                gain_input = jnp.concatenate(
                    [update_emb, nu_normed, staleness[:, jnp.newaxis]], axis=1
                )
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)  # unbounded
                # Mask the gain only based on if there has been any valid observation yet.
                # (memory padding is invalid until we have an observation)
                seen_mask = has_been_seen[:, jnp.newaxis]
                masked_gain = jnp.where(seen_mask, gain, 0.0)
            else:
                gain_input = jnp.concatenate([update_emb, nu_normed], axis=1)
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)  # unbounded
                # Mask the gain where we have stale observations
                fresh_mask = staleness[:, jnp.newaxis] == 0
                masked_gain = jnp.where(fresh_mask, gain, 0.0)

            h = h + (masked_gain * nu)
            tracing_weights[name] = {"gain": masked_gain, "nu": nu}

        return h, tracing_weights
