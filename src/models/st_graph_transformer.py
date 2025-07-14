import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

# Assuming sinusoidal_encoding and BaseModel are defined elsewhere as in the original problem
from .base_model import BaseModel
from .layers.sinusoidal_encoding import sinusoidal_encoding
from .layers.biased_attn import BiasedTransformerLayer


class ST_Graph_Transformer(BaseModel):
    """Spatio Temporal Graph Transformer"""

    hidden_size: int
    num_heads: int
    num_layers: int
    time_embedding: Array
    sensors: list[str]

    static_encoder: eqx.nn.MLP
    sensor_embed: dict[str, eqx.nn.Linear]
    transformer_layers: list[BiasedTransformerLayer]

    # This learned graph is now optional, as all info could be in the distance matrix.
    # However, learning sensor-type interactions can still be powerful.
    sensor_to_sensor_graph: Array

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        dynamic_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        seed: int,
        dropout: float,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        self.sensors = list(dynamic_sizes.keys())
        num_sensors = len(self.sensors)

        super().__init__(hidden_size * num_sensors, target, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding = sinusoidal_encoding(hidden_size, seq_length)

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

        self.sensor_to_sensor_graph = jrandom.uniform(keys.pop(0), (num_sensors, num_sensors))

        transformer_keys = jrandom.split(keys.pop(0), num_layers)
        self.transformer_layers = [
            BiasedTransformerLayer(
                input_size=hidden_size, num_heads=num_heads, dropout_p=dropout, key=t_key
            )
            for t_key in transformer_keys
        ]

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray) -> Array:
        dynamic_data = data["dynamic"]
        static_data = data["static"]
        distance_matrix = data["graph_matrix"]  # [num_locations, num_locations]

        num_locations = distance_matrix.shape[0]
        num_nodes = num_locations * len(self.sensors)
        seq_length = self.time_embedding.shape[0]

        # 1. Create Spatial Bias (No changes here)
        max_dist = jnp.max(jnp.where(jnp.isinf(distance_matrix), 0, distance_matrix))
        normalized_distance = distance_matrix / (max_dist + 1e-6)
        spatial_adjacency = jnp.exp(-(normalized_distance**2))

        spatial_adjacency = jnp.where(
            (distance_matrix == 0) & ~jnp.eye(num_locations, dtype=bool), -1e9, spatial_adjacency
        )

        spatial_bias = jnp.kron(spatial_adjacency, self.sensor_to_sensor_graph)

        # 2. Embed Inputs and Create Padding Mask (No changes here)
        sensor_embeddings = []
        padding_masks = []
        for s_name, s_emb in self.sensor_embed.items():
            features = dynamic_data[s_name]
            nan_mask = jnp.any(jnp.isnan(features), axis=-1)
            padding_masks.append(nan_mask)
            safe_features = jnp.nan_to_num(features)
            embedded_features = jax.vmap(jax.vmap(s_emb))(safe_features)
            sensor_embeddings.append(embedded_features)

        padding_mask = jnp.stack(padding_masks).transpose(1, 2, 0).reshape(seq_length, num_nodes)

        x = jnp.stack(sensor_embeddings).transpose(1, 2, 0, 3)
        x = x.reshape(seq_length, num_nodes, self.hidden_size)

        # 3. Add Positional and Static Embeddings
        # --- CHANGE 1: Corrected Time Embedding Reshape ---
        # Reshape time_embedding to (seq_length, 1, hidden_size) for correct broadcasting.
        x = x + self.time_embedding[:, None, :]

        static_loc_embeddings = jax.vmap(self.static_encoder)(static_data)
        static_node_bias = jnp.repeat(static_loc_embeddings, len(self.sensors), axis=0)
        x = x + static_node_bias[None, :, :]

        # 4. Apply Fused Spatio-Temporal Transformer Layers
        layer_keys = jrandom.split(key, self.num_layers)

        # The input to the first layer's scan is the initial embedded data `x`.
        # For subsequent layers, the input will be the output of the previous layer.
        layer_input = x

        for layer, lkey in zip(self.transformer_layers, layer_keys):
            # Initial hidden state for the scan is a zero vector.
            initial_h = jnp.zeros((num_nodes, self.hidden_size))

            # We now scan over the input data AND the padding mask for each timestep.
            scan_inputs = (layer_input, padding_mask)

            # --- CHANGE 2: Stateful Scan with Masking ---
            def process_one_timestep_with_masking(h_prev, scan_slice):
                timestep_x, timestep_mask = scan_slice  # shapes: (nodes, hidden), (nodes,)

                # Perform the graph attention on the input data for this step
                norm_x = jax.vmap(layer.norm1)(timestep_x)
                attn_output = layer.attention(
                    norm_x,
                    spatial_bias=spatial_bias,
                    mask=None,  # Causal/temporal masks are not used for node attention
                    key=lkey,
                )
                x_res = timestep_x + layer.dropout(attn_output, key=lkey)

                norm_res = jax.vmap(layer.norm2)(x_res)
                ff_output = jax.vmap(layer.feed_forward)(norm_res)
                h_new = x_res + layer.dropout(ff_output, key=lkey)

                # If mask is True (missing), use previous state h_prev.
                # Otherwise, use the newly computed state h_new.
                # The mask must be broadcast from (nodes,) to (nodes, hidden_size).
                h_final = jnp.where(timestep_mask[:, None], h_prev, h_new)

                return h_final, h_final

            # Run the scan. The second return value is the collected sequence of states.
            _, layer_output = jax.lax.scan(
                process_one_timestep_with_masking, initial_h, scan_inputs
            )

            # The output of this layer becomes the input for the next one.
            layer_input = layer_output

        # 5. Aggregate and Predict
        # Use the representation from the final timestep of the final layer's output.
        final_representation = layer_input[-1, :, :]

        aggregated_states = final_representation.reshape(
            (num_locations, len(self.sensors) * self.hidden_size)
        )

        return jax.vmap(self.head)(aggregated_states)
