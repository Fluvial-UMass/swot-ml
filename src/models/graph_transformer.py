import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphData
from .base_model import BaseModel

# from .layers.sinusoidal_encoding import sinusoidal_encoding
from .layers.static_mlp import StaticMLP
from .layers.graph_attn import SpatioTemporalLSTMCell


class ST_GATransformer(BaseModel):
    hidden_size: int
    seq_length: int
    # time_embedding: Array
    dense_sources: list[str]
    sparse_sources: list[str]
    seq2seq: bool
    return_weights: bool

    dense_embedders: dict[str, eqx.nn.MLP]
    fusion_norm: eqx.nn.LayerNorm
    sparse_embedders: dict[str, StaticMLP]
    sparse_gain_nets: dict[str, eqx.nn.MLP]
    sparse_norms: dict[str, eqx.nn.LayerNorm]
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
        seed: int,
        dropout: float,
        edge_feature_size: int,
        seq2seq: bool,
        return_weights: bool,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        # The model's total output size depends on the hidden_size
        super().__init__(hidden_size, target, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        # self.time_embedding = sinusoidal_encoding(hidden_size, seq_length)
        self.dense_sources = list(dense_sizes.keys())
        self.sparse_sources = list(sparse_sizes.keys())
        self.seq2seq = seq2seq
        self.return_weights = return_weights

        # Create embedding MLPs for dynamic sources
        dense_keys = jrandom.split(keys.pop(0), len(self.dense_sources))
        self.dense_embedders = {}
        for (name, size), k in zip(dense_sizes.items(), dense_keys):
            self.dense_embedders[name] = eqx.nn.MLP(
                in_size=size, out_size=hidden_size, width_size=hidden_size, depth=1, key=k
            )

        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sources))
        self.sparse_embedders = {}
        self.sparse_gain_nets = {}
        self.sparse_norms = {}
        for (name, size), k in zip(sparse_sizes.items(), sparse_keys):
            k1, k2 = jrandom.split(k)
            self.sparse_embedders[name] = StaticMLP(
                dynamic_in_size=size,
                static_in_size=static_size,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                key=k1
            )
            self.sparse_gain_nets[name] = eqx.nn.MLP(
                in_size=2 * hidden_size + 1,  # observation + innovation + staleness
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                key=k2,
            )
            self.sparse_norms[name] = eqx.nn.LayerNorm(hidden_size)

        self.fusion_norm = eqx.nn.LayerNorm(hidden_size)

        # The combined GAT-LSTM cell
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            node_hidden_size=hidden_size,
            static_feature_size=static_size,
            edge_feature_size=edge_feature_size,
            k_hops=k_hops,
            num_heads=num_heads,
            dropout_p=dropout,
            key=keys.pop(0),
        )

    def __call__(
        self, data: dict[str, Array | GraphData | dict[str, Array]], key: PRNGKeyArray
    ) -> Array:
        num_locations = data["graph"].node_features.shape[0]

        # --- 1. Data Embedding ---
        dense_emb_list = []
        for name in self.dense_sources:
            features = data["dynamic"][name]
            # vmap over first two dimensions (time and location)
            emb_mlp = self.dense_embedders[name]
            dense_emb_list.append(jax.vmap(jax.vmap(emb_mlp))(features))
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)

        obs_emb = {}
        for name in self.sparse_sources:
            # Clean up nans in the dynamic data
            features = data["dynamic"][name]
            obs_mask = ~jnp.any(jnp.isnan(features), axis=-1, keepdims=True) 
            safe_features = jnp.nan_to_num(features)

            embedding_mlp = self.sparse_embedders[name]
            emb_vectors = embedding_mlp(safe_features, data['static'])

            # Mask the update: if data were NaN, the update vector becomes zero
            masked_obs_emb = jnp.where(obs_mask, emb_vectors, 0.0)
            obs_emb[name] = (masked_obs_emb, obs_mask)

        # --- 2. Preprocess the observational memory ---
        if len(self.sparse_sources)>0:
            def memory_step(prev_memory: dict[str, Array], updates_masks: dict[str, tuple[Array, Array]]):
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
                    jnp.zeros((num_locations, self.hidden_size)),   # last_obs
                    jnp.ones((num_locations)) * 1e6,                # staleness
                    jnp.zeros((num_locations), dtype=jnp.bool_)     # has_been_observed
                )
                for name in self.sparse_sources
            }
            _, obs_memories = jax.lax.scan(memory_step, initial_obs_memory, obs_emb)
        else:
            obs_memories = None

        # --- 2. Recurrent Processing Loop ---
        def assimilate(h: Array, obs_memory: dict[str, tuple[Array, Array]]):
            tracing_weights = {}
            for name, (last_obs, staleness, has_been_seen) in obs_memory.items():
                update_emb = jax.vmap(self.sparse_norms[name])(last_obs)
                nu = update_emb - h  # Innovation
                gain_input = jnp.concatenate([update_emb, nu, staleness[:, jnp.newaxis]], axis=1)
                gain = jax.nn.sigmoid(jax.vmap(self.sparse_gain_nets[name])(gain_input))

                # Reshape has_been_seen to broadcast correctly with h
                mask = has_been_seen[:, jnp.newaxis]
                masked_gain = jnp.where(mask, gain, 0.0)

                h = h + (masked_gain * nu)
                tracing_weights[name] = {'gain': masked_gain, 'nu': nu}

            return h, tracing_weights

        @jax.checkpoint
        def process_one_timestep(state_prev, scan_slice):
            dense_input, step_obs_memory, step_key = scan_slice

            # Recurrent update
            (h_new, c_new), (trace_data) = self.spat_temp_lstm(
                dense_input,
                state_prev,
                data["graph"].node_features,
                data["graph"].edge_index,
                data["graph"].edge_features,
                key=step_key,
            )

            if step_obs_memory is not None:
                h_new, assim_trace = assimilate(h_new, step_obs_memory)
            else:
                assim_trace = {}
           
            new_state = (h_new, c_new)
            accumulated_outputs = (h_new, *trace_data, assim_trace)
            return new_state, accumulated_outputs

        # Initial states
        initial_h = jnp.zeros((num_locations, self.hidden_size))
        initial_c = jnp.zeros((num_locations, self.hidden_size))
        initial_state = (initial_h, initial_c)

        # Scan over the time sequence
        scan_keys = jrandom.split(key, self.seq_length)
        scan_inputs = (dense_emb_norm, obs_memories, scan_keys)
        final_state, accumulated_outputs = jax.lax.scan(
            process_one_timestep, initial_state, scan_inputs
        )
        final_h, final_c = final_state
        all_h, fwd_w, rev_w, z, r, assim = accumulated_outputs

        # --- 3. Aggregation and Prediction ---
        # The final hidden state at each location is used for prediction
        if self.seq2seq:
            predictions = jax.vmap(jax.vmap(self.head))(all_h)
        else:
            predictions = jax.vmap(self.head)(final_h)

        if self.return_weights:
            weights = {"fwd": fwd_w, "rev": rev_w, "z": z, "r": r, "assim": assim}
            return predictions, weights
        else:
            return predictions
