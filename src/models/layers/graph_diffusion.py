import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class DiffusionRouting(eqx.Module):
    hidden_size: int
    max_hops: int
    hop_mlp: eqx.nn.MLP
    rms_norm: eqx.nn.RMSNorm

    def __init__(self, static_size: int, hidden_size: int, max_hops: int, *, key: PRNGKeyArray):
        self.hidden_size = hidden_size
        self.max_hops = max_hops

        self.hop_mlp = eqx.nn.MLP(
            in_size=static_size,
            out_size=max_hops + 1,
            width_size=32,
            depth=2,
            key=key,
        )

        self.rms_norm = eqx.nn.RMSNorm(hidden_size)

    def __call__(
        self,
        static: Array,  # (num_nodes, static_size)
        H_runoff: Array,  # (num_nodes, hidden_size)
        H_prev: Array,  # (num_nodes, hidden_size)
        edges: tuple[Array, Array],  # (senders, receivers)
        node_mask: Array,  # (num_nodes,)
    ):
        senders, receivers = edges
        num_nodes = H_runoff.shape[0]

        # 1. Hop weights (node-specific)
        # Shape: (num_nodes, K+1)
        raw_alphas = jax.vmap(self.hop_mlp)(static)
        alphas = jax.nn.softmax(raw_alphas, axis=-1)

        # 2. Diffusion
        Hk = H_runoff
        routed = alphas[:, 0:1] * Hk

        for k in range(1, self.max_hops + 1):
            Hk = upstream_aggregate(Hk, senders, receivers, num_nodes)
            routed = routed + alphas[:, k : k + 1] * Hk

        # 3. Residual update + normalization
        delta = jax.vmap(self.rms_norm)(routed)
        H_out = H_prev + delta

        # 4. Mask
        H_out = jnp.where(node_mask[:, None], H_out, 0.0)

        return H_out, alphas


def upstream_aggregate(H, senders, receivers, num_nodes):
    out = jnp.zeros((num_nodes, H.shape[-1]))
    out = out.at[receivers].add(H[senders])
    return out
