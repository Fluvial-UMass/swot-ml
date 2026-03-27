from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch

from .base_model import BaseModel
from .layers.graph_attn import SpatioTemporalLSTMCell

# from .layers.sinusoidal_encoding import sinusoidal_encoding
from .layers.static_mlp import StaticMLP


class ST_GATransformer(BaseModel):
    hidden_size: int
    seq_length: int
    dense_sources: list[str]
    sparse_sources: list[str]
    seq2seq: bool
    use_obs_memory: bool
    return_weights: bool

    static_embedder: eqx.nn.MLP
    dense_embedder: eqx.nn.MLP
    sparse_embedders: dict[str, StaticMLP]
    sparse_gain_nets: dict[str, eqx.nn.MLP]
    spat_temp_lstm: SpatioTemporalLSTMCell

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
        num_heads: int,
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

        # initializes linear head and target list
        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.dense_sources = list(dense_sizes.keys())
        self.sparse_sources = list(sparse_sizes.keys())
        self.seq2seq = seq2seq
        self.return_weights = return_weights
        self.use_obs_memory = use_obs_memory

        # Embedding network for static features
        self.static_embedder = eqx.nn.MLP(
            in_size = static_size,
            out_size = hidden_size, 
            width_size = 2 * hidden_size,
            depth = 2,
            key = keys.pop(0)
        )

        # Embedding network for concatenated dense dynamic features
        total_dense_size = sum(list(dense_sizes.values()))
        self.dense_embedder = eqx.nn.MLP(
            in_size = total_dense_size,
            out_size = hidden_size, 
            width_size = 2 * hidden_size,
            depth = 2,
            key = keys.pop(0)
        )

        # Embedding and assimilation networks for the sparse sources
        self.sparse_embedders = {}
        self.sparse_gain_nets = {}
        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sources))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            self.add_assimilator(name, in_size, assim_sizes.get(name, {}), key=k)

        # The combined graph attention LSTM cell
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            static_feature_size=hidden_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            dropout_p=dropout,
            key=keys.pop(0),
        )

    def get_backbone(self) -> list[str]:
        return ['static_embedder','dense_embedder','spat_temp_lstm','head']

    def add_assimilator(
        self, name: str, n_features: int, size_args: dict = {}, *, key: PRNGKeyArray = None
    ):
        if name not in self.sparse_sources:
            self.sparse_sources.append(name)

        # Initialize with a 0 seeded key if not passed.
        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2 = jrandom.split(key)

        embed_width = size_args.get("embed_width", self.hidden_size)
        embed_depth = size_args.get("embed_depth", 2)
        self.sparse_embedders[name] = StaticMLP(
            dynamic_in_size=n_features,
            static_in_size=self.hidden_size,
            out_size=self.hidden_size,
            width_size=embed_width,
            depth=embed_depth,
            key=k1,
        )

        gain_width = size_args.get("gain_width", self.hidden_size)
        gain_depth = size_args.get("gain_depth", 2)
        # Gain network size depends on obs_memory flag
        gain_in_size = 2 * self.hidden_size + int(self.use_obs_memory)
        self.sparse_gain_nets[name] = eqx.nn.MLP(
            in_size=gain_in_size,  # observation + innovation + staleness
            out_size=self.hidden_size,
            width_size=gain_width,
            depth=gain_depth,
            key=k2,
        )


    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]  # including padding
        node_mask = data.node_mask
        edge_mask = data.edge_mask
        graph_edges = data.graph_edges

        # Data Embedding
        static_emb = jax.vmap(self.static_embedder)(data.static)

        dense_feat = jnp.concat([data.dynamic[name] for name in self.dense_sources], axis=-1)
        dense_emb = jax.vmap(jax.vmap(self.dense_embedder))(dense_feat)
        
        # Embed all sparse sources across the full time dimension up front.
        # The memory tracking (staleness, has_been_seen) is handled inside
        # the timestep scan via obs_emb_seq, so the carry is correct.
        obs_emb = {}
        for name in self.sparse_sources:
            features = data.dynamic[name]
            obs_mask = ~jnp.any(jnp.isnan(features), axis=-1, keepdims=True)
            safe_features = jnp.nan_to_num(features)
            embedding_mlp = self.sparse_embedders[name]
            emb_vectors = embedding_mlp(safe_features, static_emb)
            # Shape: (T, num_locations, hidden_size), (T, num_locations, 1)
            obs_emb[name] = (emb_vectors, obs_mask)


        # --- 2. Recurrent Processing Loop ---
        @jax.checkpoint
        def process_one_timestep(carry, scan_slice, static_data, graph_edges, node_mask, edge_mask):
            lstm_state, obs_memory = carry
            dense_input, step_obs_emb, step_key = scan_slice

            # Temporal + spatial LSTM update
            (h_new, c_new), spatial_trace = self.spat_temp_lstm(
                dense_input,
                lstm_state,
                static_data,
                graph_edges,
                node_mask,
                edge_mask,
                key=step_key,
            )

            # Update observational memory inside the scan carry
            # so staleness and has_been_seen accumulate correctly over time.
            if step_obs_emb is not None:
                obs_memory, h_new, assim_trace = self.step_obs_memory_and_assimilate(
                    h_new, obs_memory, step_obs_emb
                )
            else:
                assim_trace = {}

            new_carry = ((h_new, c_new), obs_memory)
            weights = {"spatial": spatial_trace, "assim": assim_trace}
            accumulated_outputs = (h_new, weights)

            return new_carry, accumulated_outputs

        # Initial states
        initial_lstm = (
            jnp.zeros((num_locations, self.hidden_size)),
            jnp.zeros((num_locations, self.hidden_size)),
        )
        # obs_memory is now part of the scan carry, not pre-computed
        initial_obs_memory = {
            name: (
                jnp.zeros((num_locations, self.hidden_size)),   # last_obs
                jnp.ones((num_locations,)),                     # staleness
                jnp.zeros((num_locations,), dtype=jnp.bool_),  # has_been_seen
            )
            for name in self.sparse_sources
        }
        initial_carry = (initial_lstm, initial_obs_memory)

        partial_timestep = partial(
            process_one_timestep,
            static_data=static_emb,
            graph_edges=graph_edges,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        scan_keys = jrandom.split(key, self.seq_length)
        # obs_emb_seq has shape (time, nodes, features) 
        scan_inputs = (dense_emb, obs_emb if self.sparse_sources else None, scan_keys)
        final_carry, accumulated = jax.lax.scan(partial_timestep, initial_carry, scan_inputs)

        (final_h, final_c), _ = final_carry
        all_h, weights = accumulated

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

    def step_obs_memory_and_assimilate(
        self,
        h: Array,
        obs_memory: dict[str, tuple[Array, Array, Array]],
        step_obs_emb: dict[str, tuple[Array, Array]],
    ) -> tuple[dict, Array, dict]:
        """
        Updates staleness and has_been_seen as a carry, then assimilates the updated memory into h.
        """
        new_memory = {}
        tracing_weights = {}

        for name in self.sparse_sources:
            update_emb_t, mask_t = step_obs_emb[name]   # (num_loc, hidden), (num_loc, 1)
            prev_obs, prev_stale, prev_seen = obs_memory[name]

            # Update memory state
            updated_obs = jnp.where(mask_t, update_emb_t, prev_obs)
            updated_staleness = jnp.where(
                mask_t.squeeze(axis=-1), 0.0, prev_stale + 1.0
            )
            has_been_seen = prev_seen | mask_t.squeeze(axis=-1)
            new_memory[name] = (updated_obs, updated_staleness, has_been_seen)

            # Assimilation
            nu = updated_obs - h 

            if self.use_obs_memory:
                gain_input = jnp.concatenate(
                    [updated_obs, nu, updated_staleness[:, jnp.newaxis]], axis=1
                )
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)
                seen_mask = has_been_seen[:, jnp.newaxis]
                masked_gain = jnp.where(seen_mask, gain, 0.0)
            else:
                gain_input = jnp.concatenate([updated_obs, nu], axis=1)
                gain = jax.vmap(self.sparse_gain_nets[name])(gain_input)
                fresh_mask = updated_staleness[:, jnp.newaxis] == 0
                masked_gain = jnp.where(fresh_mask, gain, 0.0)

            h = h + (masked_gain * nu)
            tracing_weights[name] = {"gain": masked_gain, "nu": nu}

        return new_memory, h, tracing_weights
