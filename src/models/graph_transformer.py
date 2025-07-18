import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphData
from .base_model import BaseModel
from .layers.sinusoidal_encoding import sinusoidal_encoding
from .layers.graph_attn import SpatioTemporalLSTMCell


class ST_GATransformer(BaseModel):
    hidden_size: int
    time_embedding: Array
    dense_sensors: list[str]
    sparse_sensors: list[str]
    return_weights: bool

    # Embedders
    dense_embed: dict[str, eqx.nn.Linear]
    dense_projector: eqx.nn.Linear
    sparse_embed: dict[str, eqx.nn.MLP]

    # Core processor for the continuous data stream
    fusion_norm: eqx.nn.LayerNorm | None
    dense_processor: SpatioTemporalLSTMCell

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
        seed: int,
        dropout: float,
        edge_feature_size: int,
        return_weights: bool,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        # The model's total output size depends on the hidden_size
        super().__init__(hidden_size, target, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.time_embedding = sinusoidal_encoding(hidden_size, seq_length)
        self.dense_sensors = list(dense_sizes.keys())
        self.sparse_sensors = list(sparse_sizes.keys())
        self.return_weights = return_weights

        # Create embedding layers for DENSE sensors
        dense_keys = jrandom.split(keys.pop(0), len(self.dense_sensors))
        self.dense_embed = {
            s_name: eqx.nn.Linear(s_size, hidden_size, key=s_key)
            for (s_name, s_size), s_key in zip(dense_sizes.items(), dense_keys)
        }
        # Add a linear layer to project concatenated dense features down to hidden_size
        dense_input_size = hidden_size * len(self.dense_sensors)
        self.dense_projector = eqx.nn.Linear(dense_input_size, hidden_size, key=keys.pop(0))

        # Create embedding MLPs for SPARSE sensors to generate update vectors
        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sensors))
        self.sparse_embed = {
            s_name: eqx.nn.MLP(
                in_size=s_size, out_size=hidden_size, width_size=hidden_size, depth=1, key=s_key
            )
            for (s_name, s_size), s_key in zip(sparse_sizes.items(), sparse_keys)
        }

        if len(self.sparse_sensors):
            self.fusion_norm = eqx.nn.LayerNorm(hidden_size)
        else:
            self.fusion_norm = None

        # This processor works on the dense stream, which has a known input size
        self.dense_processor = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            node_hidden_size=hidden_size,
            static_feature_size=static_size,
            edge_feature_size=edge_feature_size,
            k_hops=k_hops,
            dropout_p=dropout,
            key=keys.pop(0),
        )

    def __call__(
        self, data: dict[str, Array | GraphData | dict[str, Array]], key: PRNGKeyArray
    ) -> Array:
        num_locations = data["graph"].node_features.shape[0]
        seq_length = self.time_embedding.shape[0]

        # --- 1. Prepare DENSE Data Stream ---
        # This stream must be spatially and temporally complete (no NaNs)
        dense_feature_list = []
        for s_name in self.dense_sensors:
            # Assumes no NaNs in dense data, so no masking needed here
            features = data["dynamic"][s_name]
            s_emb = self.dense_embed[s_name]
            embedded_features = jax.vmap(jax.vmap(s_emb))(features)
            dense_feature_list.append(embedded_features)

        # Concatenate all dense features and add time embedding
        dense_concat = jnp.concatenate(dense_feature_list, axis=-1)
        # Project the concatenated vector down to hidden_size
        dense_projected = jax.vmap(jax.vmap(self.dense_projector))(dense_concat)
        # add the time embedding
        dense_x = dense_projected + self.time_embedding[:, None, :]

        # --- 2. Prepare SPARSE Data Stream ---
        # This stream has missing values that we turn into update vectors
        if len(self.sparse_sensors) > 0:
            sparse_update_list = []
            for s_name in self.sparse_sensors:
                features = data["dynamic"][s_name]
                nan_mask = jnp.any(jnp.isnan(features), axis=-1)
                safe_features = jnp.nan_to_num(features)

                s_emb_mlp = self.sparse_embed[s_name]
                # Create the update vector
                update_vectors = jax.vmap(jax.vmap(s_emb_mlp))(safe_features)
                # Mask the update: if data was NaN, the update vector becomes zero
                masked_updates = jnp.where(nan_mask[..., None], 0.0, update_vectors)
                sparse_update_list.append(masked_updates)
            # Sum all update vectors to get the final sparse update for each node/step
            sparse_v_update = jnp.sum(jnp.stack(sparse_update_list), axis=0)

        # --- 3. Recurrent Processing Loop ---
        def process_one_timestep(state_prev, scan_slice):
            if len(self.sparse_sensors) > 0:
                # Fusion between dense and sparse before LSTM GAT
                x_dense, x_sparse, step_key = scan_slice
                input = jax.vmap(self.fusion_norm)(x_dense + x_sparse)
            else:
                input, step_key = scan_slice

            (h_new, c_new), (trace_data) = self.dense_processor(
                input,
                state_prev,
                data["graph"].node_features,
                data["graph"].edge_index,
                data["graph"].edge_features,
                key=step_key,
            )

            new_state = (h_new, c_new)
            accumulated_outputs = (h_new, *trace_data)
            return new_state, accumulated_outputs

        # Initial states are zeros
        initial_h = jnp.zeros((num_locations, self.hidden_size))
        initial_c = jnp.zeros((num_locations, self.hidden_size))
        initial_state = (initial_h, initial_c)

        scan_keys = jrandom.split(key, seq_length)
        if len(self.sparse_sensors) > 0:
            scan_inputs = (dense_x, sparse_v_update, scan_keys)
        else:
            scan_inputs = (dense_x, scan_keys)

        # Scan over the time sequence
        final_state, accumulated_outputs = jax.lax.scan(
            process_one_timestep, initial_state, scan_inputs
        )
        final_h, final_c = final_state
        all_h, fwd_w, rev_w, z, r = accumulated_outputs

        # --- 4. Aggregation and Prediction ---
        # The final hidden state at each location is used for prediction
        predictions = jax.vmap(self.head)(final_h)

        if self.return_weights:
            weights = {"fwd": fwd_w, "rev": rev_w, "z": z, "r": r}
            return predictions, weights
        else:
            return predictions
