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


class MultiHeadFwdAttentionLayer(eqx.Module):
    """Computes forward attention using a fused MLP for all heads"""

    num_heads: int
    head_size: int
    attention_mlp: eqx.nn.MLP
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        assert node_hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        mlp_key, proj_key = jax.random.split(key)

        self.num_heads = num_heads
        self.head_size = node_hidden_size // num_heads

        # MLP input is based on full hidden dimensions, not per-head dimensions.
        mlp_input_size = (node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size
        self.attention_mlp = eqx.nn.MLP(
            in_size=mlp_input_size,
            out_size=self.num_heads,  # MLP directly outputs scores for all heads.
            width_size=node_hidden_size * 2,
            depth=1,
            use_bias=False,
            key=mlp_key,
        )
        self.output_proj = eqx.nn.Linear(
            node_hidden_size, node_hidden_size, use_bias=False, key=proj_key
        )

    def __call__(
        self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_features: Array
    ) -> tuple[Array, Array]:
        num_nodes, hidden_size = h.shape
        src, dest = edge_index

        # 1. Prepare inputs for the fused MLP
        h_src = h[src]
        h_dest = h[dest]
        mlp_input = jnp.concatenate([h_src, h_dest, x_s[src], x_s[dest], edge_features], axis=-1)

        # 2. Compute scores for all heads in a single pass
        # vmap is over the edge dimension. The MLP handles the heads.
        # raw_scores has shape (num_edges, num_heads)
        raw_scores = jax.vmap(self.attention_mlp)(mlp_input)

        raw_scores = jax.nn.leaky_relu(raw_scores)
        temperature = jnp.sqrt(self.head_size)
        scaled_scores = raw_scores / temperature

        # 3. Compute per-head softmax
        # vmap is over the head dimension of the scores.
        weights = jax.vmap(segment_softmax, in_axes=(1, None, None), out_axes=1)(
            scaled_scores, dest, num_nodes
        )

        # 4. Apply weights to per-head messages
        # Now we reshape the hidden states to apply per-head weights.
        h_reshaped = h.reshape(num_nodes, self.num_heads, self.head_size)
        weighted_messages = h_reshaped[src] * weights[:, :, None]

        aggregated_messages = jax.vmap(
            lambda msg: jax.ops.segment_sum(msg, dest, num_nodes), in_axes=1, out_axes=1
        )(weighted_messages)

        # 5. Combine heads and project
        aggregated_messages = aggregated_messages.reshape(num_nodes, hidden_size)
        projected_messages = jax.vmap(self.output_proj)(aggregated_messages)

        return projected_messages, weights


class MultiHeadRevGatingLayer(eqx.Module):
    """Computes reverse gating using a fused MLP for all heads."""

    num_heads: int
    head_size: int
    gate_mlp: eqx.nn.MLP
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        assert node_hidden_size % num_heads == 0, "node_hidden_size must be divisible by num_heads"
        mlp_key, proj_key = jax.random.split(key)

        self.num_heads = num_heads
        self.head_size = node_hidden_size // num_heads

        mlp_input_size = (node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size
        self.gate_mlp = eqx.nn.MLP(
            in_size=mlp_input_size,
            out_size=self.num_heads,  # MLP directly outputs gates for all heads.
            width_size=node_hidden_size * 2,
            depth=1,
            use_bias=False,
            key=mlp_key,
        )
        self.output_proj = eqx.nn.Linear(
            node_hidden_size, node_hidden_size, use_bias=False, key=proj_key
        )

    def __call__(
        self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_features: Array
    ) -> tuple[Array, Array]:
        num_nodes, hidden_size = h.shape
        src, dest = edge_index[::-1]  # Reversed for downstream-to-upstream

        h_src = h[src]
        h_dest = h[dest]
        mlp_input = jnp.concatenate([h_src, h_dest, x_s[src], x_s[dest], edge_features], axis=-1)

        raw_scores = jax.vmap(self.gate_mlp)(mlp_input)
        gates = jax.nn.sigmoid(raw_scores)

        h_reshaped = h.reshape(num_nodes, self.num_heads, self.head_size)
        gated_messages = h_reshaped[src] * gates[:, :, None]

        aggregated_messages = jax.vmap(
            lambda msg: jax.ops.segment_sum(msg, dest, num_nodes), in_axes=1, out_axes=1
        )(gated_messages)

        aggregated_messages = aggregated_messages.reshape(num_nodes, hidden_size)
        projected_messages = jax.vmap(self.output_proj)(aggregated_messages)

        return projected_messages, gates


class MultiHeadStackedGAT(eqx.Module):
    """
    Applies k_hops of optimized multi-head FwdAttention and RevGating, then
    integrates the results with a GRU-style update.
    """

    k_hops: int
    fwd_layers: list[MultiHeadFwdAttentionLayer]
    rev_layers: list[MultiHeadRevGatingLayer]
    norm: eqx.nn.LayerNorm

    reset_gate_mlp: eqx.nn.MLP
    update_gate_mlp: eqx.nn.MLP
    candidate_mlp: eqx.nn.MLP

    def __init__(
        self,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        k_hops: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ):
        self.k_hops = k_hops
        fwd_keys, rev_keys, gate_keys = jax.random.split(key, 3)
        fwd_keys = jax.random.split(fwd_keys, k_hops)
        rev_keys = jax.random.split(rev_keys, k_hops)
        r_key, z_key, c_key = jax.random.split(gate_keys, 3)

        self.fwd_layers = [
            MultiHeadFwdAttentionLayer(
                num_heads, node_hidden_size, static_feature_size, edge_feature_size, key=k
            )
            for k in fwd_keys
        ]
        self.rev_layers = [
            MultiHeadRevGatingLayer(
                num_heads, node_hidden_size, static_feature_size, edge_feature_size, key=k
            )
            for k in rev_keys
        ]
        self.norm = eqx.nn.LayerNorm(node_hidden_size)

        gate_input_size = node_hidden_size * 3
        self.reset_gate_mlp = eqx.nn.MLP(
            gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=r_key
        )
        self.update_gate_mlp = eqx.nn.MLP(
            gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=z_key
        )
        self.candidate_mlp = eqx.nn.MLP(
            gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=c_key
        )

    def __call__(
        self, x: Array, x_s: Array, edge_index: Array, edge_features: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        # This __call__ method remains unchanged, as the optimizations were
        # encapsulated within the attention/gating layers.
        fwd_ws, rev_ws = [], []

        h_fwd = x
        for layer in self.fwd_layers:
            h_norm = jax.vmap(self.norm)(h_fwd)
            fwd_messages, fwd_w = layer(h_norm, x_s, edge_index, edge_features)
            h_fwd = h_fwd + fwd_messages
            fwd_ws.append(fwd_w)
        fwd_ws = jnp.stack(fwd_ws, axis=-1)

        h_rev = x
        for layer in self.rev_layers:
            h_norm = jax.vmap(self.norm)(h_rev)
            rev_messages, rev_w = layer(h_norm, x_s, edge_index, edge_features)
            h_rev = h_rev + rev_messages
            rev_ws.append(rev_w)
        rev_ws = jnp.stack(rev_ws, axis=-1)

        fwd_aggregated = h_fwd - x
        rev_aggregated = h_rev - x

        messages = jnp.concatenate([fwd_aggregated, rev_aggregated], axis=-1)
        gate_input = jnp.concatenate([x, messages], axis=-1)

        r_gate = jax.nn.sigmoid(jax.vmap(self.reset_gate_mlp)(gate_input))
        z_gate = jax.nn.sigmoid(jax.vmap(self.update_gate_mlp)(gate_input))

        candidate_input = jnp.concatenate([r_gate * x, messages], axis=-1)
        candidate_state = jnp.tanh(jax.vmap(self.candidate_mlp)(candidate_input))

        final_h = (1 - z_gate) * x + z_gate * candidate_state

        return final_h, fwd_ws, rev_ws, z_gate, r_gate


class SpatioTemporalLSTMCell(eqx.Module):
    """
    A recurrent cell that first uses an LSTMCell to update the temporal state locally,
    and then uses a StackedGAT to propagate that new state spatially.
    (Note: This implements the temporal-first, spatial-second logic).
    """

    gat: MultiHeadStackedGAT
    lstm_cell: eqx.nn.LSTMCell
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        lstm_input_size: int,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        k_hops: int,
        num_heads: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gat_key, lstm_key = jax.random.split(key)
        self.gat = MultiHeadStackedGAT(
            node_hidden_size=node_hidden_size,
            static_feature_size=static_feature_size,
            edge_feature_size=edge_feature_size,
            k_hops=k_hops,
            num_heads=num_heads,
            key=gat_key,
        )
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=lstm_input_size,
            hidden_size=node_hidden_size,
            key=lstm_key,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        x_t: Array,
        state: tuple[Array, Array],
        node_features: Array,
        edge_index: Array,
        edge_features: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """
        Performs one recurrent step on the dense data stream.
        """
        h_prev, c_prev = state

        # Update states with new observations. Vmap'd over each location.
        h_candidate, c_new = jax.vmap(self.lstm_cell)(x_t, (h_prev, c_prev))

        # Propagate the freshly updated hidden states spatially.
        gat_out, fwd_w, rev_w, z, r = self.gat(
            h_candidate,
            node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        h_new = self.dropout(gat_out, key=key)

        # The new state and a tuple of all traceable weights/gates
        new_state = (h_new, c_new)
        trace_data = (fwd_w, rev_w, z, r)

        return new_state, trace_data
