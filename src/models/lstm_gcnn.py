from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch
from .base_model import BaseModel
from .layers.static_mlp import StaticMLP
from .layers.ealstm import EALSTMCell
from .layers.graph_diffusion import DiffusionRouting


class ObservationFusionModule(eqx.Module):
    """Source-agnostic observation fusion using cross-attention."""

    hidden_size: int
    obs_query_proj: eqx.nn.Linear
    obs_key_proj: eqx.nn.Linear
    obs_value_proj: eqx.nn.Linear
    null_key: jax.Array
    null_value: jax.Array
    staleness_embedding: eqx.nn.MLP
    output_gate: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray):
        keys = list(jrandom.split(key, 10))
        self.hidden_size = hidden_size

        self.obs_query_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys.pop(0))
        self.obs_key_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys.pop(0))
        self.obs_value_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys.pop(0))
        self.null_key = jax.random.normal(keys.pop(0), (hidden_size,))
        self.null_value = jax.random.normal(keys.pop(0), (hidden_size,))

        self.staleness_embedding = eqx.nn.MLP(
            in_size=1,
            out_size=hidden_size,
            width_size=hidden_size // 2,
            depth=2,
            key=keys[4],
        )
        self.output_gate = eqx.nn.Linear(hidden_size * 2, hidden_size, key=keys.pop(0))

        self.norm = eqx.nn.LayerNorm(hidden_size)

    def __call__(
        self,
        h: Array,
        obs_embeddings: list[Array],
        obs_staleness: list[Array],
        obs_valid: list[Array],
    ) -> tuple[Array, dict]:
        num_sources = len(obs_embeddings)

        if num_sources == 0:
            return h, {"weights": None, "valid_obs": None, "gate": None}

        # Stack observations
        obs_stack = jnp.stack(obs_embeddings, axis=1)
        staleness_stack = jnp.stack(obs_staleness, axis=1)
        valid_stack = jnp.stack(obs_valid, axis=1)

        # Staleness modulation
        log_staleness = jnp.log1p(staleness_stack[:, :, jnp.newaxis])
        staleness_emb = jax.vmap(jax.vmap(self.staleness_embedding))(log_staleness)
        modulated_obs = obs_stack + staleness_emb

        # Projections for sparse attention
        query = jax.vmap(self.obs_query_proj)(h)  # (L, H)
        keys = jax.vmap(jax.vmap(self.obs_key_proj))(modulated_obs)  # (L, S, H)
        values = jax.vmap(jax.vmap(self.obs_value_proj))(modulated_obs)

        # Append learned null slot
        L = h.shape[0]
        null_key = jnp.broadcast_to(self.null_key, (L, 1, self.hidden_size))
        null_value = jnp.broadcast_to(self.null_value, (L, 1, self.hidden_size))

        keys = jnp.concatenate([keys, null_key], axis=1)  # (L, S+1, H)
        values = jnp.concatenate([values, null_value], axis=1)
        # Concatenate ones since the null value is always 'valid' as key/value
        valid_stack = jnp.concatenate([valid_stack, jnp.ones((L, 1), dtype=jnp.bool_)], axis=1)

        scale = jnp.sqrt(self.hidden_size)
        scores = jnp.sum(query[:, jnp.newaxis, :] * keys, axis=-1) / scale

        # Mask by subtracting log-probability mass (gradient-safe)
        scores = scores + jnp.where(valid_stack, 0.0, -5.0)
        weights = jax.nn.softmax(scores, axis=-1)  # (L, S+1)

        fused_obs = jnp.einsum("ls,lsh->lh", weights, values)

        gate_input = jnp.concatenate([h, fused_obs], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.output_gate)(gate_input))

        # Weighted sum
        fused_obs = jnp.einsum("ls,lsh->lh", weights, values)

        # Gated residual update
        gate_input = jnp.concatenate([h, fused_obs], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.output_gate)(gate_input))

        # Scale gate by presence of *any* real observation (gradient-safe)
        any_real_valid = jnp.any(valid_stack[:, :-1], axis=-1, keepdims=True)
        gate = gate * any_real_valid.astype(gate.dtype)

        update = jax.vmap(self.norm)(fused_obs)
        h_new = h + gate * update

        return h_new, {
            "weights": weights,
            "valid_obs": valid_stack,
            "gate": gate,
        }


class SourceEmbedderRegistry(eqx.Module):
    """Registry of source-specific embedders mapping to a common hidden space."""

    embedders: dict[str, StaticMLP]
    norms: dict[str, eqx.nn.LayerNorm]
    hidden_size: int
    static_size: int

    def __init__(self, hidden_size: int, static_size: int):
        self.embedders = {}
        self.norms = {}
        self.hidden_size = hidden_size
        self.static_size = static_size

    def register_source(
        self,
        name: str,
        n_features: int,
        embed_width: int = None,
        embed_depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        embed_width = embed_width or self.hidden_size

        self.embedders[name] = StaticMLP(
            dynamic_in_size=n_features,
            static_in_size=self.static_size,
            out_size=self.hidden_size,
            width_size=embed_width,
            depth=embed_depth,
            key=key,
        )
        self.norms[name] = eqx.nn.LayerNorm(self.hidden_size)

    def embed(self, name: str, features: Array, static: Array) -> tuple[Array, Array]:
        obs_mask = ~jnp.any(jnp.isnan(features), axis=-1)
        safe_features = jnp.nan_to_num(features)

        raw_emb = self.embedders[name](safe_features, static)
        normed_emb = jax.vmap(jax.vmap(self.norms[name]))(raw_emb)

        masked_emb = jnp.where(obs_mask[:, :, jnp.newaxis], normed_emb, 0.0)

        return masked_emb, obs_mask

    @property
    def sources(self) -> list[str]:
        return list(self.embedders.keys())


class LSTMGCNN(BaseModel):
    hidden_size: int
    seq_length: int
    dense_sources: list[str]
    static_size: int
    seq2seq: bool
    supervised_attn: bool
    use_obs_memory: bool
    return_weights: bool
    staleness_threshold: int

    dense_embedders: dict[str, eqx.nn.MLP]
    fusion_norm: eqx.nn.LayerNorm
    source_registry: SourceEmbedderRegistry
    ealstm_cell: EALSTMCell
    obs_fusion: ObservationFusionModule
    routing_init_mlp: StaticMLP
    routing: DiffusionRouting

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        max_hops: int,
        dense_sizes: dict[str, int],
        sparse_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        assim_sizes: dict[str, dict[str, int]],
        seed: int,
        dropout: float,
        head: str,
        seq2seq: bool,
        supervised_attn: bool,
        use_obs_memory: bool,
        return_weights: bool,
        staleness_threshold: float = 30,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.dense_sources = list(dense_sizes.keys())
        self.static_size = static_size
        self.seq2seq = seq2seq
        self.supervised_attn = supervised_attn
        self.return_weights = return_weights
        self.use_obs_memory = use_obs_memory
        self.staleness_threshold = staleness_threshold

        # Dense embedders
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

        # Sparse embedders
        self.source_registry = SourceEmbedderRegistry(hidden_size, static_size)
        sparse_keys = jrandom.split(keys.pop(0), len(sparse_sizes))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            size_args = assim_sizes.get(name, {})
            self.source_registry.register_source(
                name,
                in_size,
                embed_width=size_args.get("embed_width"),
                embed_depth=size_args.get("embed_depth", 2),
                key=k,
            )

        # LSTM cell
        self.ealstm_cell = EALSTMCell(
            dynamic_in_size=hidden_size,
            static_in_size=static_size,
            hidden_size=hidden_size,
            entity_aware=static_size > 0,
            key=keys.pop(0),
        )

        # Routing
        self.routing_init_mlp = eqx.nn.MLP(
            in_size=static_size,
            out_size=hidden_size,
            width_size=2 * hidden_size,
            depth=2,
            key=keys.pop(0),
        )
        self.routing = DiffusionRouting(static_size, hidden_size, max_hops, key=keys.pop(0))

        # Assimilation
        self.obs_fusion = ObservationFusionModule(hidden_size, key=keys.pop(0))

    def add_assimilator(
        self,
        name: str,
        n_features: int,
        size_args: dict = {},
        *,
        key: PRNGKeyArray = None,
    ):
        """Add a new observation source."""
        key = key if key is not None else jax.random.PRNGKey(0)
        self.source_registry.register_source(
            name,
            n_features,
            embed_width=size_args.get("embed_width", 2 * self.hidden_size),
            embed_depth=size_args.get("embed_depth", 2),
            key=key,
        )

    @property
    def sparse_sources(self) -> list[str]:
        return self.source_registry.sources

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]

        # 1. Dense embedding
        dense_emb_list = []
        for name in self.dense_sources:
            features = data.dynamic[name]
            emb_mlp = self.dense_embedders[name]
            dense_emb_list.append(jax.vmap(jax.vmap(emb_mlp))(features))
        # Sum and Norm
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)

        # 2. Sparse observation embedding
        obs_emb_seq = {}
        for name in self.source_registry.sources:
            features = data.dynamic[name]
            emb, mask = self.source_registry.embed(name, features, data.static)
            obs_emb_seq[name] = (emb, mask)

        lstm_init = jnp.zeros((num_locations, self.hidden_size))

        routing_init = jax.vmap(self.routing_init_mlp)(data.static)
        obs_memory_init = {
            name: (
                jnp.zeros((num_locations, self.hidden_size)),  # emb
                jnp.full((num_locations,), 1e6),  # staleness
                jnp.zeros((num_locations,), dtype=jnp.bool_),  # seen
            )
            for name in obs_emb_seq.keys()
        }

        initial_state = (lstm_init, lstm_init, routing_init, obs_memory_init)

        @jax.checkpoint
        def one_timestep(carry, scan_slice, static, edges, node_mask):
            h_prev, c_prev, H_prev, current_obs_mem_state = carry
            dense_emb_t_slice, obs_emb_t_slice = scan_slice

            if self.ealstm_cell.entity_aware:
                i_gates = jax.nn.sigmoid(jax.vmap(self.ealstm_cell.input_linear)(static))
                in_axes = (0, 0, 0)
            else:
                i_gates = None
                in_axes = (0, 0, None)

            h_new, c_new = jax.vmap(self.ealstm_cell, in_axes=in_axes)(
                (h_prev, c_prev), dense_emb_t_slice, i_gates
            )

            # Routing
            H_new, routing_trace = self.routing(static, h_new, H_prev, edges, node_mask)

            # Update Observation Memory (Done inside the checkpoint!)
            new_obs_mem_state = {}
            for name, (emb_t, mask_t) in obs_emb_t_slice.items():
                prev_emb, prev_stale, prev_seen = current_obs_mem_state[name]

                # Update logic
                new_emb = jnp.where(mask_t[:, jnp.newaxis], emb_t, prev_emb)
                new_stale = jnp.where(mask_t, 0.0, prev_stale + 1.0)
                new_seen = prev_seen | mask_t
                new_obs_mem_state[name] = (new_emb, new_stale, new_seen)

            # Fusion (using the fresh memory state)
            H_new, fusion_trace = self._fuse_observations(H_new, new_obs_mem_state)

            # Pack carry tuple
            new_state = (h_new, c_new, H_new, new_obs_mem_state)

            step_trace = {"routing": routing_trace, "fusion": fusion_trace}
            return new_state, (H_new, step_trace)

        closed_timestep = partial(
            one_timestep,
            static=data.static,
            edges=data.graph_edges,
            node_mask=data.node_mask,
        )

        scan_seq = (dense_emb_norm, obs_emb_seq)
        _, (H_final_seq, weight_trace) = jax.lax.scan(closed_timestep, initial_state, scan_seq)

        # 3. Finally project to Discharge
        # This head now learns to map the "assimilated latent state" to Q
        Q = jax.vmap(jax.vmap(self.head["discharge"]))(H_final_seq)

        if not self.seq2seq:
            # original shape (time, locs, 1)
            Q = Q[-1, ...]

        predictions = {self.target[0]: Q}
        if self.supervised_attn:
            predictions["attention_weights"] = weight_trace["fusion"]["weights"]
            predictions["valid_obs"] = weight_trace["fusion"]["valid_obs"]

        if self.return_weights:
            return predictions, weight_trace
        else:
            return predictions

    def _create_observational_memory(
        self,
        obs_emb: dict[str, tuple[Array, Array]],
        num_locations: int,
    ):
        if len(self.source_registry.sources) == 0:
            return None

        def memory_step(prev_memory, current_obs):
            new_memory = {}
            for name, (emb_t, mask_t) in current_obs.items():
                prev_emb, prev_stale, prev_seen = prev_memory[name]

                new_emb = jnp.where(mask_t[:, jnp.newaxis], emb_t, prev_emb)
                new_stale = jnp.where(mask_t, 0.0, prev_stale + 1.0)
                new_seen = prev_seen | mask_t

                new_memory[name] = (new_emb, new_stale, new_seen)

            return new_memory, new_memory

        init_memory = {
            name: (
                jnp.zeros((num_locations, self.hidden_size)),
                jnp.full((num_locations,), 1e6),
                jnp.zeros((num_locations,), dtype=jnp.bool_),
            )
            for name in obs_emb.keys()
        }

        _, memories = jax.lax.scan(memory_step, init_memory, obs_emb)
        return memories

    def _fuse_observations(
        self,
        h: Array,
        obs_memory: dict[str, tuple[Array, Array, Array]],
    ) -> tuple[Array, dict]:
        obs_embeddings = []
        obs_staleness = []
        obs_valid = []

        if obs_memory is None:
            return h, {}

        for name, (last_emb, staleness, has_been_seen) in obs_memory.items():
            obs_embeddings.append(last_emb)
            obs_staleness.append(staleness)

            if self.use_obs_memory:
                valid = has_been_seen & (staleness < self.staleness_threshold)
            else:
                valid = staleness == 0

            obs_valid.append(valid)

        h_new, fusion_info = self.obs_fusion(h, obs_embeddings, obs_staleness, obs_valid)

        return h_new, fusion_info
