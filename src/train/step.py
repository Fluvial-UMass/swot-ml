import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree
from typing import Callable

from data import GraphBatch
from .loss import get_loss_fn


def compute_loss(
    diff_model: PyTree,
    static_model: PyTree,
    data: GraphBatch,
    key: PRNGKeyArray,
    *,
    target_weights: dict,
    loss_name: str = "mse",
    **kwargs,
) -> float:
    """Compute the loss between the predicted and true values.

    Includes options for per-target weighting and physics-based regularization for specific variable combinations.

    Parameters
    ----------
    diff_model: PyTree
        Differential components of the model.
    static_model: PyTree
        Static components of the model.
    data: GraphBatch
        GraphBatch of data to use for training.
    keys: list[PRNGKeyArray]
        Batch of keys to use for the random dropout in the model.
    loss_name: str, optional
        Name of the loss function to use.
    target_weights: dict[str: float]
        Weights to use for the target variables.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    loss : float
    """
    model = eqx.combine(diff_model, static_model)
    y_hat = model(data, key)

    if loss_name in ["nse", "spin_up_nse"]:
        # Expand the mask with a time dimension for broadcasting.
        node_mask = data.node_mask[jnp.newaxis, :]
    else:
        node_mask = data.node_mask

    # Loop through the targets and calculate loss
    losses = []
    loss_fn = get_loss_fn(loss_name)
    for target_name in y_hat.keys():
        y_target = data.y[target_name][..., 0]
        y_hat_target = y_hat[target_name]

        valid_mask = ~jnp.isnan(y_target) & node_mask
        masked_y = jnp.where(valid_mask, y_target, 0)

        if isinstance(y_hat_target, dict):
            expanded_mask = jnp.expand_dims(valid_mask, axis=-1)
            masked_y_hat = {
                p_name: jnp.where(expanded_mask, p_arr, 0) for p_name, p_arr in y_hat_target.items()
            }
        else:
            masked_y_hat = jnp.where(valid_mask, y_hat_target[..., 0], 0)

        # Calculate masked loss, safeguard against NaN's, and apply target weights
        loss = loss_fn(masked_y, masked_y_hat, valid_mask)
        loss = jnp.nan_to_num(loss, 0) * target_weights[target_name]
        losses.append(loss)

    loss = jnp.mean(jnp.stack(losses))

    return loss


def clip_gradients(grads: PyTree, max_norm: float) -> PyTree:
    """
    Clips gradients to a maximum norm.

    Parameters
    ----------
    grads: PyTree
        The gradients to clip.
    max_norm: float
        The maximum norm for clipping.

    Returns
    -------
    grads: PyTree
        The clipped gradients.
    """
    total_norm = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jtu.tree_map(lambda g: scale * g, grads)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    data: GraphBatch,
    key: PRNGKeyArray,
    opt_state: PyTree,
    optim: Callable,
    filter_spec: PyTree,
    **kwargs,
) -> tuple[float, PyTree, eqx.Module, PyTree]:
    """Performs a single optimization step, updating the model parameters.

    Parameters
    ----------
    model: eqx.Module
        Equinox model to train. Must take in a dict of data and PRNGKey.
    data: GraphBatch
        The batch of training data.
    keys: list[PRNGKeyArray]
        The PRNG keys for this batch, used by the model for dropout regularization.
    opt_state: PyTree
    optim: Callable
    filter_spec: PyTree
        The filter specification. True values indicate which parameters will be updated.
    max_grad_norm: float, optional
        The maximum gradient norm.

    Returns
    -------
    loss: float
        The loss value.
    grads: PyTree
        The update gradients.
    model: eqx.Module
        The updated model.
    opt_state: PyTree
        The updated optimizer state.
    """
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss_fn_with_grad = eqx.filter_value_and_grad(compute_loss)
    loss, grads = loss_fn_with_grad(diff_model, static_model, data, key, **kwargs)
    if kwargs.get("max_grad_norm"):
        grads = clip_gradients(grads, kwargs.get("max_grad_norm"))
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state
