import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class StaticMLP(eqx.Module):
    """MLP model that concatenates static features at each time step."""

    append_static: bool
    mlp: eqx.nn.MLP

    def __init__(
        self,
        dynamic_in_size: int,
        static_in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        self.append_static = static_in_size > 0
        self.mlp = eqx.nn.MLP(
            in_size=dynamic_in_size + static_in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x_d: Array, x_s: Array, key: PRNGKeyArray):
        """Apply MLP to inputs with optional static features."""

        # vmap function that optionally adds static data
        def mlp_apply(x):
            input = jnp.concatenate([x, x_s], axis=-1) if self.append_static else x
            return self.mlp(input, key=key)

        # Use vmap to apply mlp_apply to each row in x_d
        return jax.vmap(mlp_apply)(x_d)
