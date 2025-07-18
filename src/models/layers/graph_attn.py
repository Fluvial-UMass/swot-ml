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

    fwd_attention_mlp: eqx.nn.MLP
    rev_gate_mlp: eqx.nn.MLP

    # Gating MLPs for a GRU-style update
    reset_gate_mlp: eqx.nn.MLP
    update_gate_mlp: eqx.nn.MLP
    candidate_mlp: eqx.nn.MLP

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
        self.fwd_attention_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=keys.pop(),
        )

        # Attention mechanism for messages going to downstream nodes
        self.rev_gate_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=keys.pop(),
        )

        # MLPs for GRU-style update gates and candidate state
        gate_input_size = node_hidden_size * 3  # Current state (x) + fwd_messages + rev_messages
        self.reset_gate_mlp = eqx.nn.MLP(
            in_size=gate_input_size,
            out_size=node_hidden_size,
            width_size=node_hidden_size * 3,
            depth=1,
            key=keys.pop(),
        )
        self.update_gate_mlp = eqx.nn.MLP(
            in_size=gate_input_size,
            out_size=node_hidden_size,
            width_size=node_hidden_size * 3,
            depth=1,
            key=keys.pop(),
        )
        self.candidate_mlp = eqx.nn.MLP(
            in_size=gate_input_size,  # Takes reset state + messages
            out_size=node_hidden_size,
            width_size=node_hidden_size * 3,
            depth=1,
            key=keys.pop(),
        )

    def _compute_messages(
        self,
        x: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        edge_features: Array,
        mlp: eqx.nn.MLP,
        use_softmax: bool,
    ):
        """Helper to compute attention-weighted messages for a given direction."""
        num_nodes = x.shape[0]
        src, dest = edge_index

        # Concatenate all features to create input for the attention MLP
        all_features = [x[src], x[dest], x_s[src], x_s[dest], edge_features]
        mlp_input = jnp.concatenate(all_features, axis=-1)
        raw_scores = jax.vmap(mlp)(mlp_input).squeeze(-1)

        if use_softmax:
            raw_scores = jax.nn.leaky_relu(raw_scores)
            # Temperature scaling for numerical stability
            temperature = jnp.sqrt(x.shape[-1])
            scaled_scores = raw_scores / temperature
            # Apply softmax over each of source nodes (relative contrbution weights)
            weights = segment_softmax(scaled_scores, dest, num_nodes)
        else:
            # Use sigmoid for gating instead of softmax for attention
            weights = jax.nn.sigmoid(raw_scores)

        # Weight the source node features by attention and aggregate at destination nodes
        weighted_messages = x[src] * weights[:, None]
        aggregated_feats = jax.ops.segment_sum(weighted_messages, dest, num_nodes)
        return aggregated_feats, weights

    def __call__(
        self,
        x: Array,
        x_s: Array,
        edge_index: Array,
        edge_features: Array,
    ) -> Array:
        """
        Args:
            x: Current node hidden states.
            x_s: Static node features.
            edge_index: Edges representing upstream flow (source -> dest).
            edge_features: Features for upstream edges.
        """
        # Compute aggregated messages from upstream neighbors (e.g., runoff)
        fwd_messages, fwd_w = self._compute_messages(
            x, x_s, edge_index, edge_features, self.fwd_attention_mlp, use_softmax=True
        )

        # Compute messages for downstream neighbor. Note singular `neighbor` as all nodes in a (tributary) river network
        # only have 1 downstream neighbor. THese are in `out-degrees` in the networkx graph. So for the reverse pass, we
        # use the MLP to estimate a gate value that allows or blocks info propagation. Have to reverse the node
        # source/dest columns
        rev_messages, rev_w = self._compute_messages(
            x, x_s, edge_index[::-1], edge_features, self.rev_gate_mlp, use_softmax=False
        )

        # --- GRU-style Gated Update ---
        messages = jnp.concatenate([fwd_messages, rev_messages], axis=-1)
        gate_input = jnp.concatenate([x, messages], axis=-1)

        # Reset (r) and update (z) gates
        r_gate = jax.nn.sigmoid(jax.vmap(self.reset_gate_mlp)(gate_input))
        z_gate = jax.nn.sigmoid(jax.vmap(self.update_gate_mlp)(gate_input))

        # Candidate state
        candidate_input = jnp.concatenate([r_gate * x, messages], axis=-1)
        candidate_state = jnp.tanh(jax.vmap(self.candidate_mlp)(candidate_input))

        # Final update
        # update_gate interpolates between old state (x) and new candidate state
        update = (1 - z_gate) * x + z_gate * candidate_state

        return update, fwd_w, rev_w, z_gate, r_gate


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
        edge_index: Array,
        edge_features: Array,
    ) -> tuple[Array, Array, Array]:
        fwd_ws, rev_ws, z_gates, r_gates = [], [], [], []
        h = x

        for gat in self.gats:
            # Normalize input and apply GAT
            h_norm = jax.vmap(self.norm)(h)
            gat_out, fwd_w, rev_w, z, r = gat(
                h_norm,
                x_s,
                edge_index,
                edge_features,
            )
            h = h + gat_out  # Residual

            fwd_ws.append(fwd_w)
            rev_ws.append(rev_w)
            z_gates.append(z)
            r_gates.append(r)

        # Stack and mean over hops
        fwd_ws = jnp.stack(fwd_ws, axis=-1)
        rev_ws = jnp.stack(rev_ws, axis=-1)
        z_gates = jnp.stack(z_gates, axis=-1)
        r_gates = jnp.stack(r_gates, axis=-1)

        # Return the output and weights
        return h, fwd_ws, rev_ws, z_gates, r_gates


class SpatioTemporalLSTMCell(eqx.Module):
    """
    A recurrent cell that first uses an LSTMCell to update the temporal state locally,
    and then uses a StackedGAT to propagate that new state spatially.
    (Note: This implements the temporal-first, spatial-second logic).
    """

    gat: StackedGAT
    lstm_cell: eqx.nn.LSTMCell
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        lstm_input_size: int,
        node_hidden_size: int,
        static_feature_size: int,
        edge_feature_size: int,
        k_hops: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gat_key, lstm_key = jax.random.split(key)
        self.gat = StackedGAT(
            node_hidden_size=node_hidden_size,
            static_feature_size=static_feature_size,
            edge_feature_size=edge_feature_size,
            k_hops=k_hops,
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
