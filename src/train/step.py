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


def nse_loss(y: Array, y_pred: Array, mask: Array):
    """
    Calculates the smooth-joint NSE from Kratzert et al., 2019
    https://doi.org/10.5194/hess-23-5089-2019
    """
    # Arrays can be either [batch, basins, time] or [basins, time] depending on model config.
    # Graph models use the former because they need to predict over multiple basins for each step.
    sq_error = jnp.square(y - y_pred) * mask
    std_y = jnp.std(y, axis=-1, where=mask.astype(bool))  # Per-basin standard deviation
    se_sum = jnp.sum(sq_error, axis=-1)  # Sum of squared errors per basin
    denom = jnp.square(std_y + 0.1)  # Denominator with smoothing and epsilon for stability
    return jnp.mean(se_sum / denom)


def flux_agreement(y_pred: Array, target_list: list):
    """Calculates the normalized difference between direct SSF and SSC*Q flux estimates"""
    ssc = y_pred[:, target_list.index("ssc")] / 1e6  # mg/l -> kg/l
    flux = y_pred[:, target_list.index("flux")] / 1.102 / 1e3  # short ton/day -> kg/d
    q = y_pred[:, target_list.index("usgs_q")] * 24 * 3600 * 1000  # m^3/s -> l/d
    rel_error = ((ssc * q) - flux) / ((ssc * q + flux) / 2)
    return jnp.mean(jnp.square(rel_error))


LOSS_FN_MAP = {"mse": mse_loss, "mae": mae_loss, "huber": huber_loss, "nse": nse_loss}


def compute_loss_fn(
    diff_model: PyTree,
    static_model: PyTree,
    data: dict[str : Array | dict[str:Array]],
    keys: list[PRNGKeyArray],
    denormalize_fn: Callable,
    *,
    loss_name: str = "mse",
    target_weights: float | list[float] = 1,
    agreement_weight: float = 0,
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
    data: dict[str: Array | dict[str: Array]]
        Batch of data to use for training.
    keys: list[PRNGKeyArray]
        Batch of keys to use for the random dropout in the model.
    denormalize_fn: Callable
        Function to use to denormalize the predictions. This is useful for some types of regularization.
    loss_name: str, optional
        Name of the loss function to use.
    target_weights: float | list[float], optional
        Weights to use for the target variables.
    agreement_weight: float, optional
        Agreement regularization weight. Currently, only applicable when predicting SSC, SSF, and Q.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    loss : float
    """
    model = eqx.combine(diff_model, static_model)

    # TODO: If we start training with mixes of different basins we will need to fix this.
    static_keys = ["graph"]
    in_axes_data = {k: (None if k in static_keys and k in data else 0) for k in data}
    in_axes_keys = 0
    y_pred = jax.vmap(model, in_axes=(in_axes_data, in_axes_keys))(data, keys)

    # y_pred = jax.vmap(model)(data, keys)

    # NSE calc requires the full time series, not just the final value.
    if loss_name == "nse":
        y = data["y"]
    else:
        y = data["y"][:, -1, ...]  # End of time dimension

    valid_mask = ~jnp.isnan(y)
    masked_y = jnp.where(valid_mask, y, 0)
    masked_y_pred = jnp.where(valid_mask, y_pred, 0)

    try:
        loss_fn = LOSS_FN_MAP[loss_name]
    except KeyError:
        raise ValueError("loss name not recognized by LOSS_FN_MAP in step.py")

    vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(-1, -1, -1))  # Features dimension
    raw_losses = vectorized_loss_fn(masked_y, masked_y_pred, valid_mask)
    # Exclude any nan target losses from average.
    valid_loss = ~jnp.isnan(raw_losses)
    target_losses = jnp.where(valid_loss, raw_losses, 0)
    target_weights = valid_loss * jnp.array(target_weights)
    loss = jnp.average(target_losses, weights=target_weights)
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
    total_norm = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jtu.tree_map(lambda g: scale * g, grads)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    data: dict[str : Array | dict[str:Array]],
    keys: list[PRNGKeyArray],
    opt_state: PyTree,
    optim: Callable,
    filter_spec: PyTree,
    denormalize_fn: Callable,
    **kwargs,
) -> tuple[float, PyTree, eqx.Module, PyTree]:
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
        The denormalization function. Useful for physical regularization, when we can enforce some relationship between the real quantites, rather than their normalized or encoded values.
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
    loss_fn_with_grad = eqx.filter_value_and_grad(compute_loss_fn)
    loss, grads = loss_fn_with_grad(diff_model, static_model, data, keys, denormalize_fn, **kwargs)
    if kwargs.get("max_grad_norm"):
        grads = clip_gradients(grads, kwargs.get("max_grad_norm"))
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state
