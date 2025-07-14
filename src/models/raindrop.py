import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from .base_model import BaseModel
from .layers.sinusoidal_encoding import sinusoidal_encoding


class RAINDROP(BaseModel):
    # Sub-modules and Parameters
    hidden_size: int
    time_embedding: Array
    sensors: list[str]

    # Static and learnable components
    static_encoder: eqx.nn.MLP
    sigma_encoder: eqx.nn.MLP
    sensor_to_sensor_graph: Array  # [num_sensors, num_sensors]
    sensor_embed: dict[str, eqx.nn.Linear]
    gru_cell: eqx.nn.GRUCell
    gcn_kernel: eqx.nn.Linear  # The learned convolution kernel

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        num_locations: int,
        dynamic_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        time_embedding_size: int,
        seed: int,
        dropout: float,
    ):
        """
        Args:
            num_locations: Number of spatial locations (e.g., sub-basins).
            sensor_features: A dictionary mapping sensor names to their feature dimension.
            hidden_size: The dimension of the hidden state for each node.
            time_embedding_size: The dimension for the timestamp embeddings.
            key: A JAX random key for initialization.
        """
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        self.sensors = list(dynamic_sizes.keys())
        num_sensors = len(self.sensors)

        super().__init__(hidden_size * num_sensors, target, key=keys.pop())

        self.hidden_size = hidden_size
        self.time_embedding = sinusoidal_encoding(time_embedding_size, seq_length)

        # A small MLP to process static features (e.g., elevation, slope)
        # It learns to map static properties to a vector that will modulate the node's behavior.
        self.static_encoder = eqx.nn.MLP(
            in_size=static_size,
            out_size=hidden_size,
            width_size=hidden_size * 2,
            depth=2,
            key=keys.pop(),
        )

        # MLP to predict gaussian kernel sigma from static features.
        # It outputs a single positive value per location.
        self.sigma_encoder = eqx.nn.MLP(
            in_size=static_size,
            out_size=1,
            width_size=16,
            depth=1,
            final_activation=jax.nn.softplus,
            key=keys.pop(),
        )

        # Initialize a separate embedding layer for each sensor type
        embed_keys = jax.random.split(keys.pop(), num_sensors)
        self.sensor_embed = {
            e_name: eqx.nn.Linear(e_size, hidden_size, key=e_key)
            for (e_name, e_size), e_key in zip(dynamic_sizes.items(), embed_keys)
        }

        # This graph learns the general relationship between sensor *types*,
        # independent of their location.
        self.sensor_to_sensor_graph = jrandom.uniform(keys.pop(), (num_sensors, num_sensors))

        self.gru_cell = eqx.nn.GRUCell(hidden_size, hidden_size, key=keys.pop())

        # This is the learned convolution kernel, a linear transformation applied to node features.
        self.gcn_kernel = eqx.nn.Linear(hidden_size, hidden_size, key=keys.pop())

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray) -> Array:
        """
        Processes a full sequence of observations using GCN-style message passing.
        """
        dynamic_data = data["dynamic"]
        static_data = data["static"]
        distance_matrix = data["graph_matrix"]

        num_locations = distance_matrix.shape[0]
        num_nodes = num_locations * len(self.sensors)

        max_distance = jnp.max(distance_matrix)
        distance_matrix = distance_matrix / (max_distance + 1e-6)

        # 1. Boolean mask: 1 if connected and not self, 0 otherwise
        # Valid connections are: nonzero distances OR diagonal entries (self-connections)
        identity = jnp.eye(distance_matrix.shape[0], dtype=bool)
        is_connected = (distance_matrix != 0.0) | identity

        # 2. Fill diagonal with 0 (self-connections)
        distance_matrix = jnp.where(
            jnp.eye(distance_matrix.shape[0], dtype=bool), 0.0, distance_matrix
        )

        # 3. Predict sigmas
        sigmas_per_location = jax.vmap(self.sigma_encoder)(static_data) + 1e-4
        sigmas_per_location = jnp.clip(sigmas_per_location, 1e-2, 1e2)
        sigma_i = jnp.tile(sigmas_per_location, (1, num_locations))
        sigma_j = jnp.tile(sigmas_per_location.T, (num_locations, 1))
        avg_sigma_matrix = (sigma_i + sigma_j) / 2

        # 4. Gaussian kernel ONLY for valid connections
        gaussian = jnp.exp(-(distance_matrix**2) / (2 * avg_sigma_matrix**2 + 1e-8))
        spatial_adjacency = gaussian * is_connected.astype(jnp.float32)

        # --- Dynamic Graph Construction & Normalization ---
        dependency_graph = jnp.kron(spatial_adjacency, self.sensor_to_sensor_graph)
        # Add self-loops to the graph to include the node's own features in the aggregation
        graph_with_self_loops = dependency_graph + jnp.eye(num_nodes)

        # Compute the degree matrix D
        row_sum = jnp.sum(graph_with_self_loops, axis=1)
        # Compute D^-0.5, padding with small epsilon to avoid division by 0.
        d_inv_sqrt = jnp.power(row_sum + 1e-6, -0.5)
        D_inv_sqrt = jnp.diag(d_inv_sqrt)
        # Symmetrically normalized adjacency matrix: D^-0.5 * A * D^-0.5
        normalized_graph = D_inv_sqrt @ graph_with_self_loops @ D_inv_sqrt

        # Create biases for each location's GRU updates.
        static_loc_embeddings = jax.vmap(self.static_encoder)(static_data)
        static_node_bias = jnp.repeat(static_loc_embeddings, len(self.sensors), axis=0)

        def _process_one_timestep(states, dyn_data_step):
            # 1. Compute missing mask FOR THIS TIMESTEP

            # 2. Compute initial embeddings.
            sensor_embeddings = []
            nan_masks = []
            for i, (s_name, s_emb) in enumerate(self.sensor_embed.items()):
                features = dyn_data_step[s_name]
                nan_mask = jnp.any(jnp.isnan(features), axis=-1)
                nan_masks.append(nan_mask)
                # Use `where` to select 0 for missing data, otherwise use the original features.
                # The mask needs to be broadcast to the feature dimension.
                # This has to be done otherwise the later jnp.where condition will poison the backpropagation
                # with NaN values from these embeddings even though they are not used in the output.
                safe_features = jnp.where(nan_mask[:, None], 0.0, features)
                sensor_embeddings.append(jax.vmap(s_emb)(safe_features))

            is_missing_mask = jnp.stack(nan_masks).T.reshape(num_nodes)

            stacked_by_sensor = jnp.stack(sensor_embeddings, axis=0)
            stacked_by_loc = stacked_by_sensor.transpose(1, 0, 2)
            initial_embeddings_with_nan = stacked_by_loc.reshape(num_nodes, self.hidden_size)
            initial_embeddings = jnp.nan_to_num(initial_embeddings_with_nan)

            # 3. GCN-based Message Passing
            # TODO: What is the importance of the GCN kernel if we have the other graph?
            transformed_embeddings = jax.vmap(self.gcn_kernel)(initial_embeddings)
            propagated_messages = normalized_graph @ transformed_embeddings
            gru_inputs = propagated_messages + static_node_bias

            # 4. Update state with masking
            candidate_states = jax.vmap(self.gru_cell)(gru_inputs, states)
            updated_states = jnp.where(
                is_missing_mask[:, None],  # Broadcast mask
                states,  # If True (missing), keep old state
                candidate_states,  # If False (not missing), use new state
            )

            return updated_states, updated_states

        initial_states = jnp.zeros((num_nodes, self.hidden_size))
        # Note: we don't need time_embedding for the GCN version, but scan expects a tuple
        final_states, _ = jax.lax.scan(_process_one_timestep, initial_states, (dynamic_data))

        # return jax.vmap(self.head)(final_states)

        # --- Changed ---: Aggregate sensor states for each location before predicting
        # 1. Reshape from (num_nodes, hidden) to (num_locations, num_sensors, hidden)
        reshaped_states = final_states.reshape((num_locations, len(self.sensors), self.hidden_size))

        # 2. Concatenate the sensor states for each location into a single vector
        # New shape: (num_locations, num_sensors * hidden_size)
        aggregated_states = reshaped_states.reshape(
            (num_locations, len(self.sensors) * self.hidden_size)
        )

        # 3. Apply the head to each location's aggregated state
        # The output shape will now be (num_locations, output_dim), which is correct.
        return jax.vmap(self.head)(aggregated_states)
