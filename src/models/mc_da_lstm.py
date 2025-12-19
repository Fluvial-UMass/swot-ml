from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch
from .base_model import BaseModel
from .layers.static_mlp import StaticMLP
from .layers.ealstm import EALSTMCell
from .layers.muskingum_cunge import MuskingumCunge


class ObservationFusionModule(eqx.Module):
    """Source-agnostic observation fusion using cross-attention."""

    hidden_size: int
    null_obs_emb: Array
    obs_query_proj: eqx.nn.Linear
    obs_key_proj: eqx.nn.Linear
    obs_value_proj: eqx.nn.Linear
    staleness_embedding: eqx.nn.MLP
    output_gate: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 6)
        self.hidden_size = hidden_size

        self.null_obs_emb = jax.random.normal(keys[0], (hidden_size,))
        self.obs_query_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.obs_key_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.obs_value_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])

        self.staleness_embedding = eqx.nn.MLP(
            in_size=1,
            out_size=hidden_size,
            width_size=hidden_size // 2,
            depth=2,
            key=keys[4],
        )

        self.output_gate = eqx.nn.Linear(hidden_size * 2, hidden_size, key=keys[5])
        # A bias of +2.0 makes sigmoid(x) ≈ 0.88.
        # This forces the model to use the fusion update early in training.
        new_bias = self.output_gate.bias + 2.0
        self.output_gate = eqx.tree_at(lambda g: g.bias, self.output_gate, new_bias)

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

        # Real observational data
        obs_stack = jnp.stack(obs_embeddings, axis=1)
        staleness_stack = jnp.stack(obs_staleness, axis=1)
        valid_stack = jnp.stack(obs_valid, axis=1)

        log_staleness = jnp.log1p(staleness_stack[:, :, jnp.newaxis])
        staleness_emb = jax.vmap(jax.vmap(self.staleness_embedding))(log_staleness)
        modulated_obs = obs_stack + staleness_emb

        # Null token
        batch_size = h.shape[0]
        null_emb = jnp.tile(self.null_obs_emb[None, None, :], (batch_size, 1, 1))
        null_valid = jnp.ones((batch_size, 1), dtype=jnp.bool_)  # null is always valid

        # concat obs with null
        full_obs = jnp.concatenate([modulated_obs, null_emb], axis=1)
        full_valid = jnp.concatenate([valid_stack, null_valid], axis=1)

        # sparse attention
        query = jax.vmap(self.obs_query_proj)(h)
        keys = jax.vmap(jax.vmap(self.obs_key_proj))(full_obs)
        values = jax.vmap(jax.vmap(self.obs_value_proj))(full_obs)

        scale = jnp.sqrt(self.hidden_size)
        scores = jnp.einsum("lh,lsh->ls", query, keys) / scale

        # Mask invalid, leaving Null as the fallback
        scores = jnp.where(full_valid, scores, -1e9)

        # Never collapses to -inf because of the null obs
        weights = jax.nn.softmax(scores, axis=-1)

        fused_obs = jnp.einsum("ls,lsh->lh", weights, values)

        gate_input = jnp.concatenate([h, fused_obs], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.output_gate)(gate_input))

        # Only apply gate if we actually had REAL data 
        # If we only attended to Null, we effectively want update=0.
        any_real_valid = jnp.any(valid_stack, axis=-1, keepdims=True)
        effective_gate = jnp.where(any_real_valid, gate, 0.0)

        # residual update
        update = jax.vmap(self.norm)(fused_obs)
        h_new = h + (effective_gate * update)

        return h_new, {"weights": weights, "valid_obs": full_valid, "gate": effective_gate}


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


class MCDALSTM(BaseModel):
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
    lstm_cell: EALSTMCell
    obs_fusion: ObservationFusionModule
    routing: MuskingumCunge

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
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

        super().__init__(hidden_size, ['runoff'], head, key=keys.pop(0))

        self.target = [target[0]]
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

        # Sparse observation handling
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

        self.lstm_cell = EALSTMCell(
            dynamic_in_size=hidden_size,
            static_in_size=static_size,
            hidden_size=hidden_size,
            entity_aware=static_size > 0,
            key=keys.pop(0),
        )
        self.obs_fusion = ObservationFusionModule(hidden_size, key=keys.pop(0))

        self.routing = MuskingumCunge(static_size, hidden_size, key=keys.pop(0))

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
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)

        # 2. Sparse observation embedding
        obs_emb = {}
        for name in self.source_registry.sources:
            features = data.dynamic[name]
            emb, mask = self.source_registry.embed(name, features, data.static)
            obs_emb[name] = (emb, mask)
        obs_memories = self._create_observational_memory(obs_emb, num_locations)

        # 4. Recurrent loop
        @jax.checkpoint
        def process_one_timestep(state_prev, scan_slice, input_gate):
            dense_input, step_obs_memory, step_key = scan_slice
            
            h_new, c_new = self.lstm_cell(state_prev, dense_input, input_gate)

            h_new, fusion_trace = self._fuse_observations(h_new, step_obs_memory)

            new_state = (h_new, c_new)
            accumulated_outputs = (h_new, fusion_trace)
            return new_state, accumulated_outputs

        zeros = jnp.zeros((num_locations, self.hidden_size))
        initial_state = (zeros, zeros)

        if self.lstm_cell.entity_aware:
            input_gate = jax.nn.sigmoid(jax.vmap(self.lstm_cell.input_linear)(data.static))
        else:
            input_gate = None
        partial_timestep = partial(
            process_one_timestep,
            input_gate=input_gate,
        )

        scan_keys = jrandom.split(key, self.seq_length)
        scan_inputs = (dense_emb_norm, obs_memories, scan_keys)
        _, accumulated = jax.lax.scan(partial_timestep, initial_state, scan_inputs)

        all_h, fusion_trace = accumulated

        # 5. Predictions
        runoff = jax.vmap(jax.vmap(self.head['runoff']))(all_h)
        Q = self.routing(data.static, runoff, data.graph_edges, data.node_mask, data.edge_mask)

        if not self.seq2seq:
            # original shape (time, locs, 1)
            Q = Q[0,...]
        
        predictions = {self.target[0]: Q}
        if self.supervised_attn:
            predictions["attention_weights"] = fusion_trace["weights"]
            predictions["valid_obs"] = fusion_trace["valid_obs"]

        if self.return_weights:
            weights = {"fusion": fusion_trace}
            return predictions, weights
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
