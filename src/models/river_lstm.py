import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch

from .base_model import BaseModel
from .layers.static_mlp import StaticMLP


class RiverLSTMCellWithStatic(eqx.Module):
    W_neighbor: eqx.nn.Linear
    W_inflow_gate: eqx.nn.Linear
    W_static: eqx.nn.Linear
    W_lstm: eqx.nn.Linear
    ln_h: eqx.nn.LayerNorm
    hidden_size: int

    def __init__(
        self,
        lstm_input_size: int,
        static_size: int,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        k1, k2, k3, k4 = jrandom.split(key, 4)

        self.W_neighbor = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k1)
        self.W_inflow_gate = eqx.nn.Linear(2 * hidden_size, hidden_size, key=k2)
        self.W_static = eqx.nn.Linear(static_size, hidden_size, key=k3)

        # [x ‖ h_self ‖ neighbor_context ‖ static_proj] → 4 gates
        lstm_in = lstm_input_size + 3 * hidden_size
        self.W_lstm = eqx.nn.Linear(lstm_in, 4 * hidden_size, key=k4)

        self.ln_h = eqx.nn.LayerNorm(hidden_size)

    def __call__(
        self,
        x: Array,  # (lstm_input_size,)
        h_self: Array,  # (hidden_size,)
        c_self: Array,  # (hidden_size,)
        h_inflows: Array,  # (2, hidden_size)
        inflow_valid: Array,  # (2,)  bool
        static: Array,  # (static_size,)
    ) -> tuple[Array, Array]:
        # -- Inflow gating --
        neighbor_msgs = jax.vmap(self.W_neighbor)(h_inflows)  # (2, hidden_size)
        h_self_tiled = jnp.broadcast_to(h_self, h_inflows.shape)
        gate_input = jnp.concatenate([h_inflows, h_self_tiled], axis=-1)
        inflow_gates = jax.nn.sigmoid(jax.vmap(self.W_inflow_gate)(gate_input))
        valid_mask = inflow_valid[:, jnp.newaxis].astype(jnp.float32)
        neighbor_context = jnp.sum(inflow_gates * neighbor_msgs * valid_mask, axis=0)

        # -- Static projection --
        static_proj = self.W_static(static)

        # -- LSTM --
        combined = jnp.concatenate([x, h_self, neighbor_context, static_proj])
        i, f, o, g = jnp.split(self.W_lstm(combined), 4, axis=-1)
        c_new = jax.nn.sigmoid(f) * c_self + jax.nn.sigmoid(i) * jnp.tanh(g)
        h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)

        h_new = self.ln_h(h_new)

        return h_new, c_new


# ---------------------------------------------------------------------------
# Multi-hop propagation wrapper
# ---------------------------------------------------------------------------


def multi_hop_propagate(
    cell: RiverLSTMCellWithStatic,
    x: Array,  # (num_nodes, lstm_input_size) — same forcing for each hop
    h: Array,  # (num_nodes, hidden_size)
    c: Array,  # (num_nodes, hidden_size)
    inflow_indices: Array,  # (num_nodes, 2)  int  — index into node axis
    inflow_valid: Array,  # (num_nodes, 2)  bool
    static: Array,  # (num_nodes, static_size)
    k_hops: int,
) -> tuple[Array, Array]:
    """Apply the RiverLSTMCell k_hops times, propagating h downstream each hop.

    All nodes update in parallel per hop. After K hops a signal has travelled
    at most K reaches within one wall-clock timestep, letting you tune
    propagation speed to match your temporal resolution.
    """

    def one_hop(hc, _):
        h, c = hc
        # Gather inflow hidden states for every node: (num_nodes, 2, hidden_size)
        h_inflows = h[inflow_indices]  # fancy index into node axis
        # vmap the cell over nodes
        h_new, c_new = jax.vmap(cell)(x, h, c, h_inflows, inflow_valid, static)
        return (h_new, c_new), None

    (h_final, c_final), _ = jax.lax.scan(one_hop, (h, c), None, length=k_hops)
    return h_final, c_final


# ---------------------------------------------------------------------------
# Full RiverLSTM model
# ---------------------------------------------------------------------------


class RiverLSTM(BaseModel):
    """Spatiotemporal LSTM for river-network runoff/streamflow prediction.

    Replaces graph-attention with independently-gated inflow aggregation that
    matches the physical structure of a river network (≤2 inflows, 1 outflow).
    Multi-hop propagation within each timestep lets signals travel faster than
    one reach per clock tick.

    External call signature and assimilation interface match ST_GATransformer
    so the two models are drop-in replacements in your training loop.
    """

    hidden_size: int
    seq_length: int
    static_size: int
    k_hops: int
    seq2seq: bool
    use_obs_memory: bool
    return_weights: bool

    dense_sources: list[str]
    sparse_sources: list[str]

    dense_embedders: dict[str, eqx.nn.MLP]
    fusion_norm: eqx.nn.LayerNorm
    sparse_embedders: dict[str, StaticMLP]
    sparse_gain_nets: dict[str, eqx.nn.MLP]
    sparse_norms: dict[str, eqx.nn.LayerNorm]
    river_cell: RiverLSTMCellWithStatic

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        dense_sizes: dict[str, int],
        sparse_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        k_hops: int,
        assim_sizes: dict[str, dict[str, int]],
        seed: int,
        dropout: float,
        head: str,
        seq2seq: bool,
        use_obs_memory: bool,
        return_weights: bool,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        # BaseModel sets self.head (dict of output heads) and self.target
        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.static_size = static_size
        self.k_hops = k_hops
        self.seq2seq = seq2seq
        self.use_obs_memory = use_obs_memory
        self.return_weights = return_weights

        self.dense_sources = list(dense_sizes.keys())
        self.sparse_sources = list(sparse_sizes.keys())

        # ---- Dense (gridded / NWP) embedders ----
        dense_keys = jrandom.split(keys.pop(0), len(self.dense_sources))
        self.dense_embedders = {}
        for (name, size), k in zip(dense_sizes.items(), dense_keys):
            self.dense_embedders[name] = eqx.nn.MLP(
                in_size=size,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                key=k,
            )
        self.fusion_norm = eqx.nn.LayerNorm(hidden_size)

        # ---- Sparse (gauge) assimilators ----
        self.sparse_embedders = {}
        self.sparse_gain_nets = {}
        self.sparse_norms = {}
        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sources))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            self.add_assimilator(name, in_size, assim_sizes.get(name, {}), key=k)

        # ---- River LSTM cell ----
        # Input to cell is the fused dense embedding (hidden_size)
        self.river_cell = RiverLSTMCellWithStatic(
            lstm_input_size=hidden_size,
            static_size=static_size,
            hidden_size=hidden_size,
            key=keys.pop(0),
        )

    # ------------------------------------------------------------------
    # Assimilation interface — identical to ST_GATransformer
    # ------------------------------------------------------------------

    def add_assimilator(
        self,
        name: str,
        n_features: int,
        size_args: dict = {},
        *,
        key: PRNGKeyArray = None,
    ):
        if name not in self.sparse_sources:
            self.sparse_sources.append(name)

        key = key if key is not None else jrandom.PRNGKey(0)
        k1, k2 = jrandom.split(key)

        embed_width = size_args.get("embed_width", self.hidden_size)
        embed_depth = size_args.get("embed_depth", 2)
        self.sparse_embedders[name] = StaticMLP(
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
        self.sparse_gain_nets[name] = eqx.nn.MLP(
            in_size=gain_in_size,
            out_size=self.hidden_size,
            width_size=gain_width,
            depth=gain_depth,
            key=k2,
        )
        # Bias the final layer toward 1 so initial gain isn't near-zero
        final_linear = self.sparse_gain_nets[name].layers[-1]
        new_bias = jnp.ones_like(final_linear.bias)
        self.sparse_gain_nets[name] = eqx.tree_at(
            lambda m: m.layers[-1].bias, self.sparse_gain_nets[name], new_bias
        )

        self.sparse_norms[name] = eqx.nn.LayerNorm(self.hidden_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]
        node_mask = data.node_mask  # (num_nodes,) bool
        inflow_indices = (
            data.inflow_indices
        )  # (num_nodes, 2) int  — which nodes flow into each node
        inflow_valid = data.inflow_mask  # (num_nodes, 2) bool — False for missing/headwater slots

        # ---- 1. Embed dense (time-varying) forcings ----
        # Each embedder maps (seq_len, num_nodes, feat) → (seq_len, num_nodes, hidden)
        dense_emb_list = []
        for name in self.dense_sources:
            features = data.dynamic[name]  # (seq_len, num_nodes, feat_size)
            emb_mlp = self.dense_embedders[name]
            dense_emb_list.append(jax.vmap(jax.vmap(emb_mlp))(features))
        # Sum contributions from all dense sources, then layer-norm
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)  # (T, N, H)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)

        # ---- 2. Embed sparse (gauge) observations ----
        obs_emb = {}
        for name in self.sparse_sources:
            features = data.dynamic[name]  # (T, N, feat_size)
            obs_mask = ~jnp.any(jnp.isnan(features), axis=-1, keepdims=True)  # (T,N,1)
            safe_features = jnp.nan_to_num(features)
            embedding_mlp = self.sparse_embedders[name]
            emb_vectors = embedding_mlp(safe_features, data.static)  # (T, N, H)
            masked_obs_emb = jnp.where(obs_mask, emb_vectors, 0.0)
            obs_emb[name] = (masked_obs_emb, obs_mask)

        # ---- 3. Build observational memory (carry across time) ----
        if len(self.sparse_sources) > 0:
            obs_memories = self.create_observational_memory(obs_emb, num_locations)
        else:
            obs_memories = None

        # ---- 4. Scan over timesteps ----
        @jax.checkpoint
        def process_one_timestep(state_prev, scan_slice):
            dense_input_t, step_obs_memory, step_key = scan_slice
            # dense_input_t : (num_nodes, hidden_size)
            h_prev, c_prev = state_prev

            # Multi-hop river propagation
            h_new, c_new = multi_hop_propagate(
                self.river_cell,
                dense_input_t,
                h_prev,
                c_prev,
                inflow_indices,
                inflow_valid,
                data.static,
                self.k_hops,
            )

            # Mask padding nodes — keep their state frozen
            h_new = jnp.where(node_mask[:, jnp.newaxis], h_new, h_prev)
            c_new = jnp.where(node_mask[:, jnp.newaxis], c_new, c_prev)

            # Optional sparse data assimilation
            if step_obs_memory is not None:
                h_new, assim_trace = self.assimilate(h_new, step_obs_memory)
            else:
                assim_trace = {}

            new_state = (h_new, c_new)
            weights = {"assim": assim_trace}
            return new_state, (h_new, weights)

        initial_state = (
            jnp.zeros((num_locations, self.hidden_size)),
            jnp.zeros((num_locations, self.hidden_size)),
        )

        scan_keys = jrandom.split(key, self.seq_length)
        scan_inputs = (dense_emb_norm, obs_memories, scan_keys)

        final_state, accumulated = jax.lax.scan(process_one_timestep, initial_state, scan_inputs)

        final_h, final_c = final_state
        all_h, weights = accumulated  # all_h: (T, N, H)

        # ---- 5. Predictions ----
        if self.seq2seq:
            predictions = {
                target: jax.vmap(jax.vmap(head))(all_h) for target, head in self.head.items()
            }
        else:
            predictions = {target: jax.vmap(head)(final_h) for target, head in self.head.items()}

        if self.return_weights:
            return predictions, weights
        else:
            return predictions

    def create_observational_memory(self, obs_emb, num_locations):
        def memory_step(prev_memory, updates_masks):
            new_memory = {}
            for name, (update_emb_t, mask_t) in updates_masks.items():
                prev_obs, prev_stale, prev_seen = prev_memory[name]

                # If mask_t is True, we have a new observation; reset staleness to 0
                # Otherwise, increment the counter
                updated_obs = jnp.where(mask_t, update_emb_t, prev_obs)
                updated_staleness = jnp.where(mask_t.squeeze(axis=-1), 0.0, prev_stale + 1.0)
                has_been_seen = prev_seen | mask_t.squeeze(axis=-1)
                new_memory[name] = (updated_obs, updated_staleness, has_been_seen)
            return new_memory, new_memory

        initial_obs_memory = {
            name: (
                jnp.zeros((num_locations, self.hidden_size)),
                # Start with high staleness so initial "zeros" have no weight
                jnp.ones((num_locations,)) * 100.0,
                jnp.zeros((num_locations,), dtype=jnp.bool_),
            )
            for name in self.sparse_sources
        }
        _, obs_memories = jax.lax.scan(memory_step, initial_obs_memory, obs_emb)
        return obs_memories

    def assimilate(self, h: Array, obs_memory: dict[str, tuple[Array, Array]]):
        tracing_weights = {}
        decay_lambda = 0.5

        for name, (last_obs, staleness, has_been_seen) in obs_memory.items():
            # fall back to h (nonzero) to protect grads from layernorm of 0s
            safe_last_obs = jnp.where(has_been_seen[:, jnp.newaxis], last_obs, h)

            update_emb = jax.vmap(self.sparse_norms[name])(safe_last_obs)
            nu = update_emb - h

            if self.use_obs_memory:
                # Transform staleness count to [0, 1] range
                # staleness=0 -> 1.0 (fresh); staleness -> inf -> 0.0 (dead)
                decayed_weight = jnp.exp(-decay_lambda * staleness)[:, jnp.newaxis]

                gain_input = jnp.concatenate([update_emb, nu, decayed_weight], axis=1)
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)

                seen_mask = has_been_seen[:, jnp.newaxis]
                masked_gain = jnp.where(seen_mask, gain, 0.0)
            else:
                gain_input = jnp.concatenate([update_emb, nu], axis=1)
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)
                fresh_mask = staleness[:, jnp.newaxis] == 0
                masked_gain = jnp.where(fresh_mask, gain, 0.0)

            h = h + (masked_gain * nu)
            tracing_weights[name] = {"gain": masked_gain, "nu": nu}

        return h, tracing_weights
