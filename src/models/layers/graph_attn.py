import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def segment_softmax(scores: Array, segment_ids: Array, num_segments: int) -> Array:
    """Applies a numerically stable softmax to scores over segments."""
    max_scores = jax.ops.segment_max(scores, segment_ids, num_segments=num_segments)
    stable_scores = scores - max_scores[segment_ids]
    exp_scores = jnp.exp(stable_scores)
    sum_exp_scores = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)
    return exp_scores / (sum_exp_scores[segment_ids] + 1e-9)


class DirectionalGAT(eqx.Module):
    """
    Directionally-aware Graph Attention message passing.
    It computes and aggregates messages from upstream and downstream neighbors separately,
    allowing the model to learn different functions for these distinct relationships.
    """

    up_attention_mlp: eqx.nn.MLP
    down_attention_mlp: eqx.nn.MLP
    update_mlp: eqx.nn.MLP
    activation: callable

    def __init__(
        self,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = list(jax.random.split(key, 10))

        input_size = (node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size
        # Attention mechanism for messages coming from upstream nodes
        self.up_attention_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=keys.pop(),
        )

        # Attention mechanism for messages going to downstream nodes
        self.down_attention_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=keys.pop(),
        )

        # MLP to update the node state after aggregating messages from both directions
        self.update_mlp = eqx.nn.MLP(
            in_size=node_hidden_size * 3,  # Current state + upstream messages + downstream messages
            out_size=node_hidden_size,
            width_size=node_hidden_size * 3,
            depth=1,
            key=keys.pop(),
        )
        self.activation = jax.nn.relu

    def _compute_messages(
        self,
        x: Array,
        x_s: Array,
        node_mask: Array,
        edge_index: tuple[Array, Array],
        edge_features: Array,
        attention_mlp: eqx.nn.MLP,
    ):
        """Helper to compute attention-weighted messages for a given direction."""
        num_nodes = x.shape[0]
        source_nodes, dest_nodes = edge_index

        # Get features for source and destination nodes of each edge
        source_hidden_feats = x[source_nodes]
        dest_hidden_feats = x[dest_nodes]
        source_static_feats = x_s[source_nodes]
        dest_static_feats = x_s[dest_nodes]

        # Concatenate all features to create input for the attention MLP
        attention_input = jnp.concatenate(
            [
                source_hidden_feats,
                dest_hidden_feats,
                source_static_feats,
                dest_static_feats,
                edge_features,
            ],
            axis=-1,
        )

        # Compute raw attention scores and apply leaky ReLU
        raw_scores = jax.vmap(attention_mlp)(attention_input).squeeze(-1)
        raw_scores = jax.nn.leaky_relu(raw_scores)

        # Temperature scaling for numerical stability
        temperature = jnp.sqrt(x.shape[-1])
        scaled_scores = raw_scores / temperature

        # An edge is invalid if its source node has invalid data for this timestep.
        # By setting the score to -inf, it becomes zero after the softmax.
        # source_mask = node_mask[source_nodes]
        # masked_scores = jnp.where(source_mask, jnp.finfo(scaled_scores.dtype).min, scaled_scores)
        # attention_weights = segment_softmax(masked_scores, dest_nodes, num_nodes)

        attention_weights = segment_softmax(scaled_scores, dest_nodes, num_nodes)

        # Weight the source node features by attention and aggregate at destination nodes
        weighted_messages = source_hidden_feats * attention_weights[:, None]
        aggregated_feats = jax.ops.segment_sum(
            weighted_messages, dest_nodes, num_segments=num_nodes
        )
        return aggregated_feats, attention_weights

    def __call__(
        self,
        x: Array,
        x_s: Array,
        node_mask: Array,
        up_edge_index: Array,
        up_edge_features: Array,
        down_edge_index: Array,
        down_edge_features: Array,
    ) -> Array:
        """
        Args:
            x: Current node hidden states.
            x_s: Static node features.
            up_edge_index: Edges representing upstream flow (source -> dest).
            up_edge_features: Features for upstream edges.
            down_edge_index: Edges representing downstream influence (dest -> source).
            down_edge_features: Features for downstream edges.
        """
        # Compute aggregated messages from upstream neighbors
        up_messages, up_w = self._compute_messages(
            x,
            x_s,
            node_mask,
            up_edge_index,
            up_edge_features,
            self.up_attention_mlp,
        )

        # Compute aggregated messages from downstream neighbors (e.g., backwater effects)
        down_messages, down_w = self._compute_messages(
            x,
            x_s,
            node_mask,
            down_edge_index,
            down_edge_features,
            self.down_attention_mlp,
        )

        # Combine current state with messages from both directions and update
        update_input = jnp.concatenate([x, up_messages, down_messages], axis=-1)
        update = self.activation(jax.vmap(self.update_mlp)(update_input))
        return update, up_w, down_w


class StackedGAT(eqx.Module):
    """
    Applies the DirectionalGAT layer k_hops times sequentially to expand the spatial
    receptive field. Residual connections and layer normalization are used for stability.
    """

    gats: list[DirectionalGAT]
    norm: eqx.nn.LayerNorm
    k_hops: int

    def __init__(
        self,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        k_hops: int,
        *,
        key: PRNGKeyArray,
    ):
        self.k_hops = k_hops
        keys = jax.random.split(key, k_hops)

        self.gats = [
            DirectionalGAT(
                node_hidden_size=node_hidden_size,
                static_feature_size=static_feature_size,
                edge_feature_size=edge_feature_size,
                key=k,
            )
            for k in keys
        ]
        self.norm = eqx.nn.LayerNorm(node_hidden_size)

    def __call__(
        self,
        x: Array,
        x_s: Array,
        node_mask: Array,
        up_edge_index: Array,
        up_edge_features: Array,
        down_edge_index: Array,
        down_edge_features: Array,
    ) -> tuple[Array, Array, Array]:
        up_ws = []
        down_ws = []

        for gat in self.gats:
            # Normalize input and apply GAT
            gat_input = jax.vmap(self.norm)(x)
            gat_out, up_w, down_w = gat(
                gat_input,
                x_s,
                node_mask,
                up_edge_index,
                up_edge_features,
                down_edge_index,
                down_edge_features,
            )
            
            x = x + gat_out # Residual connection for training stability
            up_ws.append(up_w)
            down_ws.append(down_w)

        # Stack and mean over hops
        up_w_agg = jnp.mean(jnp.stack(up_ws), axis=0)
        down_w_agg = jnp.mean(jnp.stack(down_ws), axis=0)

        # Return the output and weights
        return x, up_w_agg, down_w_agg


class SpatioTemporalLSTMCell(eqx.Module):
    """
    A recurrent cell that first uses a DirectionalGAT to aggregate spatial information
    and then uses an LSTMCell to update the temporal hidden state.
    """

    norm: eqx.nn.LayerNorm
    gat: StackedGAT
    lstm_cell: eqx.nn.LSTMCell
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        k_hops: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gat_key, lstm_key = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(node_hidden_size)
        self.gat = StackedGAT(
            node_hidden_size=node_hidden_size,
            static_feature_size=static_feature_size,
            edge_feature_size=edge_feature_size,
            k_hops=k_hops,
            key=gat_key,
        )
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=node_hidden_size,  # GAT output size
            hidden_size=node_hidden_size,
            key=lstm_key,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        x_t: Array,
        state: tuple[Array, Array],
        x_s: Array,
        node_mask: Array,
        up_edge_index: Array,
        up_edge_features: Array,
        down_edge_index: Array,
        down_edge_features: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """
        Performs one recurrent step.

        Args:
            x_t: Node features for the current timestep.
            state: Tuple of (hidden_state, cell_state) from the previous timestep.
            x_s: Static node features.
            (rest): Graph structure information.
            key: JAX random key.
        """
        h_prev, c_prev = state

        # --- Spatial Update ---
        # Get a spatial context vector by aggregating neighbor information.
        gat_out = self.gat(
            x_t,
            x_s,
            node_mask,
            up_edge_index,
            up_edge_features,
            down_edge_index,
            down_edge_features,
        )
        spatial_context, up_w, down_w = gat_out
        spatial_context = self.dropout(spatial_context, key=key)

        # --- Temporal Update ---
        # Vmap'd over each location
        h_new, c_new = jax.vmap(self.lstm_cell)(spatial_context, (h_prev, c_prev))

        return (h_new, c_new), (up_w, down_w)
