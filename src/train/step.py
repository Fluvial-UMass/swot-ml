import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable


def mse_loss(y: Array, y_pred: Array, mask: Array):
    """Calculates Mean Squared Error (MSE) loss."""
    mse = jnp.mean(jnp.square(y - y_pred), where=mask)
    return mse


def mae_loss(y: Array, y_pred: Array, mask: Array):
    """Calculates Mean Absolute Error (MAE) loss."""
    mae = jnp.mean(jnp.abs(y - y_pred), where=mask)
    return mae


def huber_loss(y: Array, y_pred: Array, mask: Array, *, huber_delta: float = 1.0):
    """Calculates Huber loss."""
    # Passing delta through make_step is not yet implemented.
    residual = y - y_pred
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def flux_agreement(y_pred: Array, target_list: list):
    """Calculates the normalized difference between direct SSF and SSC*Q flux estimates"""
    ssc = y_pred[:, target_list.index('ssc')] / 1E6  # mg/l -> kg/l
    flux = y_pred[:, target_list.index('flux')] / 1.102 / 1E3  # short ton/day -> kg/d
    q = y_pred[:, target_list.index('usgs_q')] * 24 * 3600 * 1000  # m^3/s -> l/d

    rel_error = ((ssc * q) - flux) / ((ssc * q + flux) / 2)
    return jnp.mean(jnp.square(rel_error))


@eqx.filter_value_and_grad
def compute_loss(diff_model: PyTree,
                 static_model: PyTree,
                 data: dict[str:Array | dict[str:Array]],
                 keys: list[PRNGKeyArray],
                 denormalize_fn: Callable,
                 *,
                 loss_name: str = 'mse',
                 target_weights: float | list[float] = 1,
                 agreement_weight: float = 0,
                 **kwargs) -> float:
    """Compute the loss between the predicted and true values.

    Includes options for per-target weighting and physics-based regularization for 
    specific variable combinations.

    Parameters
    ----------
    diff_model: PyTree
        Differential components of the model.
    static_model: PyTree
        Static components of the model.
    data: dict[str: Array | dict[str: Array]]
        Batch of data to use for training.
    keys: list[PRNGKeyArray]
        Batch of keys to use for the random dropout in the model. 
    denormalize_fn: Callable
        Function to use to denormalize the predictions. This is useful for some 
        types of regularization.
    loss_name: str, optional
        Name of the loss function to use.
    target_weights: float | list[float], optional
        Weights to use for the target variables.
    agreement_weight: float, optional
        Agreement regularization weight. Currently, only applicable when predicting 
        SSC, SSF, and Q.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    loss : float
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

    vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(-1, -1, -1))  # Features dimension
    raw_losses = vectorized_loss_fn(masked_y, masked_y_pred, valid_mask)

    # Exclude any nan target losses from average.
    valid_loss = ~jnp.isnan(raw_losses)
    target_losses = jnp.where(valid_loss, raw_losses, 0)
    target_weights = valid_loss * jnp.array(target_weights)

    loss = jnp.average(target_losses, weights=target_weights)

    # Debugging
    def print_func(operand):
        y, y_pred = operand
        jax.debug.print(
            "y: {a}\ny_pred: {b}\nraw losses: {c}\nweights: {d}\nweighted losses: {e}",
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

    total_norm = jtu.tree_reduce(lambda x, y: x + y,
                                 jtu.tree_map(lambda x: jnp.sum(x**2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jax.tree_map(lambda g: scale * g, grads)


"""
    Clip gradients to prevent them from exceeding a maximum norm.

    Args:
        grads (PyTree): The gradients to be clipped.
        max_norm (float): The maximum norm for clipping.
        
    Returns:
        PyTree: The clipped gradients.
    """


@eqx.filter_jit
def make_step(model: eqx.Module, data: dict[str:Array | dict[str:Array]],
              keys: list[PRNGKeyArray], opt_state: PyTree, optim: Callable,
              filter_spec: PyTree, denormalize_fn: Callable,
              **kwargs) -> tuple[float, PyTree, eqx.Module, PyTree]:
    """Performs a single optimization step, updating the model parameters.

    Parameters
    ----------
    model: eqx.Module
        Equinox model to train. Must take in a dict of data and PRNGKey. 
    data: dict[str: Array | dict[str: Array]]
        The batch of training data.
    keys: list[PRNGKeyArray]
        The PRNG keys for this batch, used by the model for dropout regularization.
    opt_state: PyTree
    optim: Callable
    filter_spec: PyTree
        The filter specification. True values indicate which parameters will be updated. 
    denormalize_fn: Callable
        The denormalization function. Useful for physical regularization, when we can 
        enforce some relationship between the real quantites, rather than their normalized or encoded values. 
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
    loss, grads = compute_loss(diff_model, static_model, data, keys, denormalize_fn,
                               **kwargs)

    if kwargs.get('max_grad_norm'):
        grads = clip_gradients(grads, kwargs.get('max_grad_norm'))

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state
