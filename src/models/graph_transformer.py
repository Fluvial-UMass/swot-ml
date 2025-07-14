import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphData
from .base_model import BaseModel
from .layers.sinusoidal_encoding import sinusoidal_encoding
from .layers.graph_attn import GATLayer


class ST_GATransformer(BaseModel):
    """Spatio Temporal Graph Transformer with self-contained GAT layers"""
    hidden_size: int
    num_layers: int
    time_embedding: Array
    sensors: list[str]
    edge_feature_size: int

    static_encoder: eqx.nn.MLP
    sensor_embed: dict[str, eqx.nn.Linear]
    graph_layers: list[GATLayer] 

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

        # --- REFACTORED: Create a list of self-contained GATLayers ---
        graph_keys = jrandom.split(keys.pop(0), num_layers)
        self.graph_layers = [
            GATLayer(
                node_hidden_size=hidden_size,
                static_feature_size=static_size,
                edge_feature_size=self.edge_feature_size,
                dropout_p=dropout,
                key=g_key,
            ) for g_key in graph_keys
        ]
    
    def __call__(self, data: dict[str, Array | GraphData | dict[str, Array]], key: PRNGKeyArray) -> Array:
        num_locations = data['graph'].node_features.shape[0]
        num_sensors = len(self.sensors)
        num_nodes = num_locations * num_sensors
        seq_length = self.time_embedding.shape[0]

        node_edge_index = []
        node_edge_features = []
        for s_offset_i in range(num_sensors):
            for s_offset_j in range(num_sensors):
                source_nodes = data['graph'].edge_index[0] * num_sensors + s_offset_i
                dest_nodes = data['graph'].edge_index[1] * num_sensors + s_offset_j
                node_edge_index.append(jnp.stack([source_nodes, dest_nodes]))
                node_edge_features.append(data['graph'].edge_features)

        edge_index = jnp.concatenate(node_edge_index, axis=1)
        edge_features = jnp.concatenate(node_edge_features, axis=0)

        sensor_embeddings = []
        padding_masks = []
        for s_name, s_emb in self.sensor_embed.items():
            features = data["dynamic"][s_name]
            nan_mask = jnp.any(jnp.isnan(features), axis=-1)
            padding_masks.append(nan_mask)
            safe_features = jnp.nan_to_num(features)
            embedded_features = jax.vmap(jax.vmap(s_emb))(safe_features)
            sensor_embeddings.append(embedded_features)

        padding_mask = jnp.stack(padding_masks).transpose(1, 2, 0).reshape(seq_length, num_nodes)
        x = jnp.stack(sensor_embeddings).transpose(1, 2, 0, 3).reshape(seq_length, num_nodes, self.hidden_size)
        x = x + self.time_embedding[:, None, :]
        static_loc_embeddings = jax.vmap(self.static_encoder)(data["static"])
        static_node_bias = jnp.repeat(static_loc_embeddings, len(self.sensors), axis=0)
        x = x + static_node_bias[None, :, :]
        # --- End of data preparation ---

        # --- UPDATED: Simplified processing loop with GATLayer ---
        layer_keys = jrandom.split(key, self.num_layers)
        layer_input = x

        # We now iterate over a single list of layers
        for graph_layer, lkey in zip(self.graph_layers, layer_keys):
            initial_h = jnp.zeros((num_nodes, self.hidden_size))
            scan_inputs = (layer_input, padding_mask, jrandom.split(lkey, seq_length))

            def process_one_timestep_with_masking(h_prev, scan_slice):
                timestep_x, timestep_mask, step_key = scan_slice
                
                # Call the self-contained GATLayer
                h_new = graph_layer(
                    timestep_x, 
                    data['graph'].node_features, 
                    edge_index, 
                    edge_features, 
                    key=step_key
                )
                
                # Apply padding mask
                h_final = jnp.where(timestep_mask[:, None], h_prev, h_new)
                return h_final, h_final

            _, layer_output = jax.lax.scan(process_one_timestep_with_masking, initial_h, scan_inputs)
            layer_input = layer_output

        # --- Aggregation and Prediction part is unchanged ---
        final_representation = layer_input[-1, :, :]
        aggregated_states = final_representation.reshape((num_locations, len(self.sensors) * self.hidden_size))
        return jax.vmap(self.head)(aggregated_states)