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


class FwdAttentionLayer(eqx.Module):
    """
    Computes a single hop of forward (upstream-to-downstream) attention-weighted messages.
    """
    attention_mlp: eqx.nn.MLP

    def __init__(self, node_hidden_size: int, static_feature_size: int, edge_feature_size: int, *, key: PRNGKeyArray):
        input_size = (node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size
        self.attention_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=key,
        )

    def __call__(self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_features: Array) -> tuple[Array, Array]:
        num_nodes = h.shape[0]
        src, dest = edge_index

        mlp_input = jnp.concatenate([h[src], h[dest], x_s[src], x_s[dest], edge_features], axis=-1)
        raw_scores = jax.vmap(self.attention_mlp)(mlp_input).squeeze(-1)

        raw_scores = jax.nn.leaky_relu(raw_scores)
        temperature = jnp.sqrt(h.shape[-1])
        scaled_scores = raw_scores / temperature
        weights = segment_softmax(scaled_scores, dest, num_nodes)

        weighted_messages = h[src] * weights[:, None]
        aggregated_messages = jax.ops.segment_sum(weighted_messages, dest, num_nodes)
        return aggregated_messages, weights
    

class RevGatingLayer(eqx.Module):
    """
    Computes a single hop of reverse (downstream-to-upstream) gated messages.
    """
    gate_mlp: eqx.nn.MLP

    def __init__(self, node_hidden_size: int, static_feature_size: int, edge_feature_size: int, *, key: PRNGKeyArray):
        input_size = (node_hidden_size * 2) + (static_feature_size * 2) + edge_feature_size
        self.gate_mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=node_hidden_size * 2,
            depth=1,
            key=key,
        )

    def __call__(self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_features: Array) -> tuple[Array, Array]:
        num_nodes = h.shape[0]
        # Note: edge_index is reversed for downstream-to-upstream flow
        src, dest = edge_index[::-1]

        mlp_input = jnp.concatenate([h[src], h[dest], x_s[src], x_s[dest], edge_features], axis=-1)
        raw_scores = jax.vmap(self.gate_mlp)(mlp_input).squeeze(-1)
        gates = jax.nn.sigmoid(raw_scores)

        gated_messages = h[src] * gates[:, None]
        aggregated_messages = jax.ops.segment_sum(gated_messages, dest, num_nodes)
        return aggregated_messages, gates


class StackedGAT(eqx.Module):
    """
    Applies k_hops of FwdAttention and RevGating independently, then integrates
    the results with a GRU-style update. This prevents information from cycling
    between forward and reverse passes at each hop.
    """
    k_hops: int
    fwd_layers: list[FwdAttentionLayer]
    rev_layers: list[RevGatingLayer]
    norm: eqx.nn.LayerNorm

    # GRU-style integration MLPs
    reset_gate_mlp: eqx.nn.MLP
    update_gate_mlp: eqx.nn.MLP
    candidate_mlp: eqx.nn.MLP

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
        fwd_keys, rev_keys, gate_keys = jax.random.split(key, 3)
        fwd_keys = jax.random.split(fwd_keys, k_hops)
        rev_keys = jax.random.split(rev_keys, k_hops)
        r_key, z_key, c_key = jax.random.split(gate_keys, 3)

        self.fwd_layers = [
            FwdAttentionLayer(node_hidden_size, static_feature_size, edge_feature_size, key=k)
            for k in fwd_keys
        ]
        self.rev_layers = [
            RevGatingLayer(node_hidden_size, static_feature_size, edge_feature_size, key=k)
            for k in rev_keys
        ]
        self.norm = eqx.nn.LayerNorm(node_hidden_size)

        # Integration MLPs are now here
        gate_input_size = node_hidden_size * 3  # Initial state (x) + fwd_aggr + rev_aggr
        self.reset_gate_mlp = eqx.nn.MLP(gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=r_key)
        self.update_gate_mlp = eqx.nn.MLP(gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=z_key)
        self.candidate_mlp = eqx.nn.MLP(gate_input_size, node_hidden_size, node_hidden_size * 3, 1, key=c_key)


    def __call__(self, x: Array, x_s: Array, edge_index: Array, edge_features: Array) -> tuple[Array, Array, Array, Array, Array]:
        fwd_ws, rev_ws = [], []

        # --- Forward Propagation ---
        h_fwd = x
        for layer in self.fwd_layers:
            h_norm = jax.vmap(self.norm)(h_fwd)
            fwd_messages, fwd_w = layer(h_norm, x_s, edge_index, edge_features)
            h_fwd = h_fwd + fwd_messages  # Residual connection
            fwd_ws.append(fwd_w)
        fwd_ws = jnp.stack(fwd_ws, axis=-1)

        # --- Reverse Propagation ---
        h_rev = x
        for layer in self.rev_layers:
            h_norm = jax.vmap(self.norm)(h_rev)
            rev_messages, rev_w = layer(h_norm, x_s, edge_index, edge_features)
            h_rev = h_rev + rev_messages  # Residual connection
            rev_ws.append(rev_w)
        rev_ws = jnp.stack(rev_ws, axis=-1)

        # --- Final Gated Integration ---
        # The "message" is the total change aggregated over k_hops for each path
        fwd_aggregated = h_fwd - x
        rev_aggregated = h_rev - x

        messages = jnp.concatenate([fwd_aggregated, rev_aggregated], axis=-1)
        gate_input = jnp.concatenate([x, messages], axis=-1)

        r_gate = jax.nn.sigmoid(jax.vmap(self.reset_gate_mlp)(gate_input))
        z_gate = jax.nn.sigmoid(jax.vmap(self.update_gate_mlp)(gate_input))

        candidate_input = jnp.concatenate([r_gate * x, messages], axis=-1)
        candidate_state = jnp.tanh(jax.vmap(self.candidate_mlp)(candidate_input))

        # final_h is interpolation of previous and candidate based on gate. 
        final_h = (1 - z_gate) * x + z_gate * candidate_state

        return final_h, fwd_ws, rev_ws, z_gate, r_gate



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
