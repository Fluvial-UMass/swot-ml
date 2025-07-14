import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class GATMessagePassing(eqx.Module):
    """
    Graph Attention Network (GAT) message passing.
    Computes and aggregates attention-weighted messages based on node and edge features.
    This is the core spatial aggregation mechanism.
    """

    attention_mlp: eqx.nn.MLP
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
        keys = jax.random.split(key, 2)

        self.attention_mlp = eqx.nn.MLP(
            in_size=(node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size,
            out_size=1,
            width_size=(node_hidden_size * 2 + edge_feature_size),
            depth=1,
            key=keys[0],
        )

        self.update_mlp = eqx.nn.MLP(
            in_size=node_hidden_size * 2,
            out_size=node_hidden_size,
            width_size=node_hidden_size * 2,
            depth=1,
            key=keys[1],
        )
        self.activation = jax.nn.relu

    def __call__(self, x: Array, x_s: Array, edge_index: Array, edge_features: Array) -> Array:
        num_nodes = x.shape[0]
        source_nodes, dest_nodes = edge_index

        source_hidden_feats = x[source_nodes]
        dest_hidden_feats = x[dest_nodes]
        source_static_feats = x_s[source_nodes]
        dest_static_feats = x_s[dest_nodes]

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

        raw_attention_scores = jax.vmap(self.attention_mlp)(attention_input).squeeze(-1)
        raw_attention_scores = jax.nn.leaky_relu(raw_attention_scores)

        attention_weights = segment_softmax(raw_attention_scores, dest_nodes, num_nodes)

        weighted_messages = source_hidden_feats * attention_weights[:, None]
        aggregated_feats = jax.ops.segment_sum(
            weighted_messages, dest_nodes, num_segments=num_nodes
        )

        update_input = jnp.concatenate([x, aggregated_feats], axis=-1)
        return self.activation(jax.vmap(self.update_mlp)(update_input))


def segment_softmax(scores: Array, segment_ids: Array, num_segments: int) -> Array:
    """
    Applies a numerically stable softmax to scores over segments.
    Args:
        scores: The raw scores to be softmaxed (e.g., attention scores).
        segment_ids: An array mapping each score to a segment (e.g., destination nodes).
        num_segments: The total number of segments (e.g., total number of nodes).
    Returns:
        The normalized attention weights for each score.
    """
    # Find the maximum score in each segment for numerical stability
    max_scores = jax.ops.segment_max(scores, segment_ids, num_segments=num_segments)

    # Subtract the max score from all scores in their respective segments
    stable_scores = scores - max_scores[segment_ids]

    # Exponentiate and sum up scores within each segment
    exp_scores = jnp.exp(stable_scores)
    sum_exp_scores = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)

    # Normalize by dividing by the sum of exponentiated scores in the segment
    # Add a small epsilon to avoid division by zero for isolated nodes
    return exp_scores / (sum_exp_scores[segment_ids] + 1e-9)


class GATLayer(eqx.Module):
    """
    A complete Spatio-Temporal processing block.
    It uses GAT for spatial aggregation within a standard "pre-norm"
    Transformer-style architecture (Norm -> Sublayer -> Dropout -> Residual).
    """

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    message_passing: GATMessagePassing
    feed_forward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gkey, fkey = jax.random.split(key)

        self.norm1 = eqx.nn.LayerNorm(node_hidden_size)
        self.norm2 = eqx.nn.LayerNorm(node_hidden_size)
        self.message_passing = GATMessagePassing(
            node_hidden_size=node_hidden_size,
            static_feature_size=static_feature_size,
            edge_feature_size=edge_feature_size,
            key=gkey,
        )
        self.feed_forward = eqx.nn.MLP(
            in_size=node_hidden_size,
            out_size=node_hidden_size,
            width_size=4 * node_hidden_size,
            depth=2,
            key=fkey,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self, x: Array, x_s: Array, edge_index: Array, edge_features: Array, *, key: PRNGKeyArray
    ) -> Array:
        mp_key, ff_key = jax.random.split(key, 2)

        # --- First sub-layer: Spatial Message Passing ---
        norm_x = jax.vmap(self.norm1)(x)
        mp_output = self.message_passing(norm_x, x_s, edge_index, edge_features)
        x = x + self.dropout(mp_output, key=mp_key)

        # --- Second sub-layer: Position-wise Feed-Forward ---
        norm_x_2 = jax.vmap(self.norm2)(x)
        ff_output = jax.vmap(self.feed_forward)(norm_x_2)
        x = x + self.dropout(ff_output, key=ff_key)

        return x
