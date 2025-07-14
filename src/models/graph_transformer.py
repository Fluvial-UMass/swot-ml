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
    num_layers: int
    time_embedding: Array
    sensors: list[str]
    edge_feature_size: int

    static_encoder: eqx.nn.MLP
    sensor_embed: dict[str, eqx.nn.Linear]
    # The model now has a list of recurrent cells instead of simple GAT layers.
    recurrent_layers: list[SpatioTemporalLSTMCell]

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        dynamic_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        num_layers: int,
        seed: int,
        dropout: float,
        edge_feature_size: int,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        self.sensors = list(dynamic_sizes.keys())
        num_sensors = len(self.sensors)

        super().__init__(hidden_size * num_sensors, target, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_embedding = sinusoidal_encoding(hidden_size, seq_length)
        self.edge_feature_size = edge_feature_size

        self.static_encoder = eqx.nn.MLP(
            in_size=static_size,
            out_size=hidden_size,
            width_size=hidden_size * 2,
            depth=2,
            key=keys.pop(0),
        )

        embed_keys = jrandom.split(keys.pop(0), num_sensors)
        self.sensor_embed = {
            e_name: eqx.nn.Linear(e_size, hidden_size, key=e_key)
            for (e_name, e_size), e_key in zip(dynamic_sizes.items(), embed_keys)
        }

        # --- REFACTORED: Create a list of SpatioTemporalLSTMCells ---
        layer_keys = jrandom.split(keys.pop(0), num_layers)
        self.recurrent_layers = [
            SpatioTemporalLSTMCell(
                node_hidden_size=hidden_size,
                static_feature_size=static_size,
                edge_feature_size=self.edge_feature_size,
                dropout_p=dropout,
                key=l_key,
            )
            for l_key in layer_keys
        ]

    def __call__(
        self, data: dict[str, Array | GraphData | dict[str, Array]], key: PRNGKeyArray
    ) -> Array:
        num_locations = data["graph"].node_features.shape[0]
        num_sensors = len(self.sensors)
        num_nodes = num_locations * num_sensors
        seq_length = self.time_embedding.shape[0]

        # --- Directional Edge Preparation ---
        # Assumes data['graph'].edge_index represents downstream flow (source -> dest)
        down_edge_index_base = data["graph"].edge_index
        down_edge_features_base = data["graph"].edge_features
        # Upstream edges are the reverse of downstream edges
        up_edge_index_base = jnp.stack([down_edge_index_base[1], down_edge_index_base[0]])
        up_edge_features_base = down_edge_features_base  # Assuming edge features are symmetric

        # Expand edge indices for the multi-sensor node representation
        up_node_edge_index, up_node_edge_features = [], []
        down_node_edge_index, down_node_edge_features = [], []
        for s_offset_i in range(num_sensors):
            for s_offset_j in range(num_sensors):
                # Upstream
                up_source = up_edge_index_base[0] * num_sensors + s_offset_i
                up_dest = up_edge_index_base[1] * num_sensors + s_offset_j
                up_node_edge_index.append(jnp.stack([up_source, up_dest]))
                up_node_edge_features.append(up_edge_features_base)
                # Downstream
                down_source = down_edge_index_base[0] * num_sensors + s_offset_i
                down_dest = down_edge_index_base[1] * num_sensors + s_offset_j
                down_node_edge_index.append(jnp.stack([down_source, down_dest]))
                down_node_edge_features.append(down_edge_features_base)

        up_edge_index = jnp.concatenate(up_node_edge_index, axis=1)
        up_edge_features = jnp.concatenate(up_node_edge_features, axis=0)
        down_edge_index = jnp.concatenate(down_node_edge_index, axis=1)
        down_edge_features = jnp.concatenate(down_node_edge_features, axis=0)

        # --- Input Feature Preparation (same as before) ---
        sensor_embeddings, padding_masks = [], []
        for s_name, s_emb in self.sensor_embed.items():
            features = data["dynamic"][s_name]
            nan_mask = jnp.any(jnp.isnan(features), axis=-1)
            padding_masks.append(nan_mask)
            safe_features = jnp.nan_to_num(features)
            embedded_features = jax.vmap(jax.vmap(s_emb))(safe_features)
            sensor_embeddings.append(embedded_features)

        padding_mask = jnp.stack(padding_masks).transpose(1, 2, 0).reshape(seq_length, num_nodes)
        x = (
            jnp.stack(sensor_embeddings)
            .transpose(1, 2, 0, 3)
            .reshape(seq_length, num_nodes, self.hidden_size)
        )
        x = x + self.time_embedding[:, None, :]
        static_loc_embeddings = jax.vmap(self.static_encoder)(data["static"])
        static_node_bias = jnp.repeat(static_loc_embeddings, len(self.sensors), axis=0)
        x = x + static_node_bias[None, :, :]

        # --- Recurrent Processing Loop ---
        layer_keys = jrandom.split(key, self.num_layers)
        h_sequence = x  # Use input features as the initial "hidden state" for the first layer

        for layer, lkey in zip(self.recurrent_layers, layer_keys):
            # Initial states for the LSTM are zeros
            initial_h = jnp.zeros((num_nodes, self.hidden_size))
            initial_c = jnp.zeros((num_nodes, self.hidden_size))
            initial_state = (initial_h, initial_c)

            scan_keys = jrandom.split(lkey, seq_length)
            scan_inputs = (h_sequence, padding_mask, scan_keys)

            def process_one_timestep(state_prev, scan_slice):
                h_prev, c_prev = state_prev
                x_t, mask_t, step_key = scan_slice

                # Perform one step of the spatio-temporal update
                h_candidate, c_candidate = layer(
                    x_t,
                    (h_prev, c_prev),
                    data["graph"].node_features,
                    mask_t,
                    up_edge_index,
                    up_edge_features,
                    down_edge_index,
                    down_edge_features,
                    key=step_key,
                )

                # --- CRITICAL: Handle Missing Data ---
                # If the input at this timestep was masked (NaN), do not update the state.
                # Instead, carry forward the previous state.
                mask_t_expanded = mask_t[:, None]
                h_final = jnp.where(mask_t_expanded, h_prev, h_candidate)
                c_final = jnp.where(mask_t_expanded, c_prev, c_candidate)

                state_final = (h_final, c_final)
                return state_final, h_final  # Output the hidden state for the next layer

            # Scan over the time sequence
            _, h_sequence = jax.lax.scan(process_one_timestep, initial_state, scan_inputs)

        # --- Aggregation and Prediction (Unchanged) ---
        final_representation = h_sequence[-1, :, :]
        aggregated_states = final_representation.reshape(
            (num_locations, len(self.sensors) * self.hidden_size)
        )
        return jax.vmap(self.head)(aggregated_states)
