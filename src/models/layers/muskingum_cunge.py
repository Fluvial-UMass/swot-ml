import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Tuple

class MuskingumCunge(eqx.Module):
    mlp: eqx.nn.MLP
    hidden_size: int

    def __init__(self, static_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.hidden_size = hidden_size
        # Input: [static_features + Q_prev + runoff_curr]
        self.mlp = eqx.nn.MLP(
            in_size=static_size + 2, 
            out_size=2, 
            width_size=hidden_size, 
            depth=2, 
            key=key
        )

    def __call__(
        self, 
        static: Array, 
        runoff: Array, 
        edges: Array, 
        node_mask: Array, 
        edge_mask: Array
    ) -> Array:
        """
        static: [nodes, static_feat]
        runoff: [time, nodes]
        edges: [2, num_edges]
        node_mask: [nodes]
        edge_mask: [num_edges]
        """
        num_timesteps = runoff.shape[0]
        num_nodes = runoff.shape[1]

        # Initialize discharge Q and inflow I
        init_Q = jnp.zeros(num_nodes)
        init_I = jnp.zeros(num_nodes)
        
        def calculate_inflow(Q_prev: Array, r_curr: Array) -> Array:
            # Sum upstream discharge reaching each node
            # Using edge_mask to zero out padded edges
            src, dst = edges
            upstream_flow = jnp.zeros(num_nodes).at[dst].add(Q_prev[src] * edge_mask)
            # Total inflow = upstream flows + local runoff
            return upstream_flow + r_curr

        def step(carry: Tuple[Array, Array], r_t: Array):
            Q_prev, I_prev = carry
            
            # Safer than squeeze to make it shape (num_nodes,)
            r_t_sq = r_t.reshape(num_nodes)

            # 1. Calculate current inflow
            I_curr = calculate_inflow(Q_prev, r_t_sq)
            
            # 2. Predict routing parameters K and X per node
            # Input features: static attributes, previous discharge, and local runoff
            mlp_in = jnp.concatenate([
                static, 
                Q_prev[:, None], 
                r_t_sq[:, None],
            ], axis=-1)
            
            # Map MLP across nodes (vmap)
            params = jax.vmap(self.mlp)(mlp_in)
            K = jax.nn.softplus(params[:, 0]) # Travel time (must be positive)
            X = jax.nn.sigmoid(params[:, 1]) * 0.5 # Weighting (typically 0 to 0.5)

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
            
            return (Q_curr, I_curr), Q_curr

        # Iterate through time
        _, Q_history = jax.lax.scan(step, (init_Q, init_I), runoff)
        
        return Q_history[...,jnp.newaxis]