import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class StaticMLP(eqx.Module):
    """
    An MLP that concatenates static features to dynamic features before processing.

    This module is designed to handle inputs with any number of leading batch
    dimensions (e.g., time, location).
    """

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
        """
        Initializes the StaticMLP.

        Args:
            dynamic_in_size: The feature size of the dynamic input array.
            static_in_size: The feature size of the static input array. If 0,
                            no static data is appended.
            out_size: The output feature size of the MLP.
            width_size: The width of the hidden layers in the MLP.
            depth: The number of hidden layers in the MLP.
            key: A JAX random key for initializing the MLP weights.
        """
        self.append_static = static_in_size > 0
        mlp_in_size = dynamic_in_size + static_in_size if self.append_static else dynamic_in_size

        self.mlp = eqx.nn.MLP(
            in_size=mlp_in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self, x_d: Array, x_s: Array | None = None, *, key: PRNGKeyArray | None = None
    ) -> Array:
        """
        Apply the MLP to inputs, optionally concatenating static features first.

        Args:
            x_d: The dynamic features array. Expected shape is (..., dynamic_in_size).
            x_s: The static features array. Its batch dimensions must be a suffix of
                 `x_d`'s batch dimensions. For example, if `x_d` has shape
                 (time, location, feats_d), `x_s` could have shape (location, feats_s).
                 Required if `static_in_size` > 0 during initialization.
            key: An optional JAX random key, used if the MLP contains dropout layers.

        Returns:
            The output of the MLP, with the same batch dimensions as `x_d`.
        """
        if self.append_static:
            if x_s is None:
                raise ValueError(
                    "Static features `x_s` must be provided when `static_in_size` > 0."
                )

            # Reshape x_s to be broadcastable with x_d's leading dimensions.
            # E.g., if x_d is (T, N, D_dyn) and x_s is (N, D_stat),
            # this makes x_s compatible for broadcasting to (T, N, D_stat).
            num_extra_dims = x_d.ndim - x_s.ndim
            if num_extra_dims < 0:
                raise ValueError(
                    "Static features `x_s` cannot have more dimensions than dynamic features `x_d`."
                )

            new_shape = (1,) * num_extra_dims + x_s.shape
            x_s_reshaped = jnp.reshape(x_s, new_shape)

            # Broadcast x_s to match the batch dimensions of x_d
            broadcast_shape = x_d.shape[:-1] + (x_s.shape[-1],)
            x_s_broadcasted = jnp.broadcast_to(x_s_reshaped, broadcast_shape)

            mlp_input = jnp.concatenate([x_d, x_s_broadcasted], axis=-1)
        else:
            mlp_input = x_d

        # Dynamically vmap the adapter over all batch dimensions of the input.
        vmapped_apply = self.mlp
        for _ in range(mlp_input.ndim - 1):
            vmapped_apply = jax.vmap(vmapped_apply)

        return vmapped_apply(mlp_input)
