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
    return_weights: bool

    sensor_embed: dict[str, eqx.nn.Linear]
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
        return_weights: bool,
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
        self.return_weights = return_weights

        embed_keys = jrandom.split(keys.pop(0), num_sensors)
        self.sensor_embed = {
            e_name: eqx.nn.Linear(e_size, hidden_size, key=e_key)
            for (e_name, e_size), e_key in zip(dynamic_sizes.items(), embed_keys)
        }

        # Create a list of SpatioTemporalLSTMCells
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

        # --- Input Feature Preparation ---
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

        # --- Recurrent Processing Loop ---
        all_layer_weights = []  # store the weights from each layer
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
                h_0, c_0 = state_prev
                x_t, mask_t, step_key = scan_slice

                # Perform one step of the spatio-temporal update
                layer_out = layer(
                    x_t,
                    (h_0, c_0),
                    data["graph"].node_features,
                    mask_t,
                    up_edge_index,
                    up_edge_features,
                    down_edge_index,
                    down_edge_features,
                    key=step_key,
                )
                (h_1, c_1), (up_w, down_w) = layer_out

                # # If the input at this timestep was masked (NaN), do not update the state.
                # # Instead, carry forward the previous state.
                # mask_t_expanded = mask_t[:, None]
                # h_1 = jnp.where(mask_t_expanded, h_0, h_1)
                # c_1 = jnp.where(mask_t_expanded, c_0, c_1)

                # Return the new states for the next layer and states/weights for accumulation
                return (h_1, c_1), (h_1, up_w, down_w)

            # Scan over the time sequence
            _, (scan_accum) = jax.lax.scan(process_one_timestep, initial_state, scan_inputs)
            h_sequence, up_w, down_w = scan_accum

            # Store the collected weights and gates for this layer
            all_layer_weights.append({"up_w": up_w, "down_w": down_w})

        # --- Aggregation and Prediction (Unchanged) ---
        final_representation = h_sequence[-1, :, :]

        out_shape = (num_locations, len(self.sensors) * self.hidden_size)
        aggregated_states = final_representation.reshape(out_shape)
        predictions = jax.vmap(self.head)(aggregated_states)

        if self.return_weights:
            return predictions, all_layer_weights
        else:
            return predictions
