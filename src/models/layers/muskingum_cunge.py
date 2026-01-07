import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class LatentMuskingumCunge(eqx.Module):
    """Propagates hidden states through the graph using MC logic."""

    hidden_size: int
    parameter_generator: eqx.nn.MLP  # Maps static/state to C1, C2, C3
    num_substeps: int

    def __init__(self, static_size, hidden_size, num_substeps, key):
        self.hidden_size = hidden_size
        self.num_substeps = num_substeps
        # Outputs 3 coefficients per hidden dimension, or 3 shared across dims
        self.parameter_generator = eqx.nn.MLP(
            in_size=static_size,
            out_size=3,
            width_size=16,
            depth=1,
            final_activation=jax.nn.sigmoid,  # Ensure unity of params
            key=key,
        )

    def __call__(self, static, H_runoff, H_prev_out, H_prev_in, edges, node_mask, edge_mask):
        # 1. Generate coefficients based on physical properties
        # Shape: (num_locations, 3)
        raw_coeffs = jax.vmap(self.parameter_generator)(static)
        coeffs = jax.nn.softmax(raw_coeffs, axis=-1)
        C1, C2, C3 = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]

        # Distribute daily runoff across sub-steps
        # We assume runoff is added incrementally at each sub-step
        H_runoff_sub = H_runoff / self.num_substeps

        # 2. Aggregate upstream inputs (H_t, i-1)
        senders, receivers = edges

        @jax.checkpoint 
        def sub_step_fn(i, carry):
            curr_out_prev, curr_in_prev = carry  # These are from step i-1

            # 1. Current Inflow (I_t) from upstream neighbors
            upstream_h = curr_out_prev[senders]
            H_upstream_t = jnp.zeros_like(H_runoff).at[receivers].add(upstream_h)

            # 2. Add local runoff (already divided by num_substeps)
            I_t = H_upstream_t + H_runoff_sub

            # 3. Routing Equation
            # At i=0, curr_in_prev is H_prev_in (from yesterday)
            # At i>0, curr_in_prev is I_t from the previous sub-step
            H_out_t = C1 * I_t + C2 * curr_in_prev + C3 * curr_out_prev

            # Masking
            H_out_t = jnp.where(node_mask[:, jnp.newaxis], H_out_t, 0.0)

            return (H_out_t, I_t)

        # Initial carry for the first sub-step is the state from the previous DAY
        init_carry = (H_prev_out, H_prev_in)
        final_out, final_in = jax.lax.fori_loop(0, self.num_substeps, sub_step_fn, init_carry)

        return final_out, final_in


class MuskingumCunge(eqx.Module):
    mlp: eqx.nn.MLP
    hidden_size: int

    def __init__(self, static_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.hidden_size = hidden_size
        # Input: [static_features + Q_prev + runoff_curr]
        self.mlp = eqx.nn.MLP(
            in_size=static_size + 2, out_size=2, width_size=hidden_size, depth=2, key=key
        )

    def __call__(
        self,
        static: Array,
        runoff: Array,
        Q_prev: Array,
        I_prev: Array,
        edges: Array,
        node_mask: Array,
        edge_mask: Array,
    ) -> Array:
        """
        static: [nodes, static_feat]
        runoff: [time, nodes]
        edges: [2, num_edges]
        node_mask: [nodes]
        edge_mask: [num_edges]
        """
        num_nodes = runoff.shape[0]

        def calculate_inflow(Q_prev: Array, r_curr: Array) -> Array:
            # Sum upstream discharge reaching each node
            # Using edge_mask to zero out padded edges
            src, dst = edges
            upstream_flow = jnp.zeros(num_nodes).at[dst].add(Q_prev[src] * edge_mask)
            # Total inflow = upstream flows + local runoff
            return upstream_flow + r_curr

        # Safer than squeeze to make it shape (num_nodes,)
        r_t_sq = runoff.reshape(num_nodes)

        # 1. Calculate current inflow
        I_curr = calculate_inflow(Q_prev, r_t_sq)

        # 2. Predict routing parameters K and X per node
        # Input features: static attributes, previous discharge, and local runoff
        mlp_in = jnp.concatenate(
            [
                static,
                Q_prev[:, None],
                r_t_sq[:, None],
            ],
            axis=-1,
        )

        # Map MLP across nodes (vmap)
        params = jax.vmap(self.mlp)(mlp_in)
        K = jax.nn.softplus(params[:, 0])  # Travel time (must be positive)
        X = jax.nn.sigmoid(params[:, 1]) * 0.5  # Weighting (typically 0 to 0.5)

        # 3. Muskingum-Cunge Coefficients
        dt = 1.0
        denom = 2 * K * (1 - X) + dt
        c1 = (dt - 2 * K * X) / denom
        c2 = (dt + 2 * K * X) / denom
        c3 = (2 * K * (1 - X) - dt) / denom
        c4 = (2 * dt) / denom

        # 4. Update Q
        Q_curr = (c1 * I_curr) + (c2 * I_prev) + (c3 * Q_prev) + (c4 * r_t_sq)

        # Apply node mask to prevent padding nodes from accumulating
        Q_curr = jnp.where(node_mask, Q_curr, 0.0)
        I_curr = jnp.where(node_mask, I_curr, 0.0)

        return Q_curr, I_curr
