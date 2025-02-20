import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable


def mse_loss(y: Array, y_pred: Array, mask: Array):
    mse = jnp.mean(jnp.square(y - y_pred), where=mask)
    return mse


def mae_loss(y: Array, y_pred: Array, mask: Array):
    mae = jnp.mean(jnp.abs(y - y_pred), where=mask)
    return mae


def huber_loss(y: Array, y_pred: Array, mask: Array, *, huber_delta: float = 1.0):
    # Passing delta through make_step is not yet implemented.
    residual = y - y_pred
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def flux_agreement(y_pred: Array, target_list: list):
    ssc = y_pred[:, target_list.index('ssc')] / 1E6  # mg/l -> kg/l
    flux = y_pred[:, target_list.index('flux')] / 1.102 / 1E3  # short ton/day -> kg/d
    q = y_pred[:, target_list.index('usgs_q')] * 24 * 3600 * 1000  # m^3/s -> l/d

    rel_error = ((ssc * q) - flux) / ((ssc * q + flux) / 2)
    return jnp.mean(jnp.square(rel_error))


@eqx.filter_value_and_grad
def compute_loss(diff_model: PyTree, static_model: PyTree, data: dict[str:Array | dict[str:Array]],
                 keys: list[PRNGKeyArray], denormalize_fn: Callable, loss_name: str, target_weights: list[float],
                 agreement_weight: float) -> float:
    """
    Computes the loss between the model predictions and the targets using the specified loss function.

    Args:
        diff_model (PyTree): Differentiable components of model.
        static_model (PyTree): Static components of model.
        data (dict): Dictionary Containing all input data.
        keys (list[PRNGKeyArray]): List of rng keys for batch.
        denormalize_fn (Callable): Function to denormalize data back to real units. Useful for some types of regularization.
        loss_name (str): The name of the loss function to use.
        target_weights (list[float]): List of weights for each target.
        agreement_weight (float): Agreement regularization weight.

    Returns:
        float: The computed loss.
    """
    model = eqx.combine(diff_model, static_model)
    y_pred = jax.vmap(model)(data, keys)

    y = data['y'][:, -1, ...]  # End of time dimension
    valid_mask = ~jnp.isnan(y)
    masked_y = jnp.where(valid_mask, y, 0)
    masked_y_pred = jnp.where(valid_mask, y_pred, 0)

    if loss_name == "mse":
        loss_fn = mse_loss
    elif loss_name == "mae":
        loss_fn = mae_loss
    elif loss_name == "huber":
        loss_fn = huber_loss
    else:
        raise ValueError("Invalid loss function name.")

    vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(-1, -1, -1))
    raw_losses = vectorized_loss_fn(masked_y, masked_y_pred, valid_mask)

    # Exclude any nan target losses from average.
    valid_loss = ~jnp.isnan(raw_losses)
    target_losses = jnp.where(valid_loss, raw_losses, 0)
    target_weights = valid_loss * jnp.array(target_weights)

    loss = jnp.average(target_losses, weights=target_weights)

    # Debugging
    def print_func(operand):
        y, y_pred = operand
        jax.debug.print("y: {a}\ny_pred: {b}\nraw losses: {c}\nweights: {d}\nweighted losses: {e}",
                        a=y,
                        b=y_pred,
                        c=raw_losses,
                        d=target_weights,
                        e=loss)
        return None

    jax.lax.cond(jnp.isnan(loss), print_func, lambda *args: None, operand=(y, y_pred))
    # Debugging

    if agreement_weight > 0:
        y_pred_denorm = denormalize_fn(y_pred)
        loss += agreement_weight * flux_agreement(y_pred_denorm, model.target)

    return loss


def clip_gradients(grads: PyTree, max_norm: float) -> PyTree:
    """
    Clip gradients to prevent them from exceeding a maximum norm.

    Args:
        grads (PyTree): The gradients to be clipped.
        max_norm (float): The maximum norm for clipping.
        
    Returns:
        PyTree: The clipped gradients.
    """
    total_norm = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jax.tree_map(lambda g: scale * g, grads)


@eqx.filter_jit
def make_step(model: eqx.Module, data: dict[str:Array | dict[str:Array]], keys: list[PRNGKeyArray], opt_state: PyTree,
              optim: Callable, filter_spec: PyTree, denormalize_fn: Callable,
              **kwargs) -> tuple[float, PyTree, eqx.Module, PyTree]:
    """
    Performs a single optimization step, updating the model parameters.

    Args:
        model (equinox.Module): Equinox model that takes in single dict of data arrays
        data (dict): Dictionary of batched data arrays for model input
        opt_state: The state of the optimizer.
        optim: The optimizer.
        filter_spec (pytree): Pytree filter spec following structure of model. True elements will be updated.
        kwargs:
            loss (str): name of loss fn
            target_weights (list, optional): relative weights of targets for loss calc
            max_grad_norm (float, optional): The maximum norm for clipping gradients.

    Returns:
        tuple: A tuple containing the loss, gradients, updated model, and updated optimizer state.
    """
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = compute_loss(diff_model, static_model, data, keys, denormalize_fn, kwargs.get('loss'),
                               kwargs.get('target_weights', 1), kwargs.get('agreement_weight', 0))

    if kwargs.get('max_grad_norm'):
        grads = clip_gradients(grads, kwargs.get('max_grad_norm'))

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state
