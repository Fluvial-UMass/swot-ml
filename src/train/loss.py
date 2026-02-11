from typing import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from data import GraphBatch


def compute_loss(
    diff_model: PyTree,
    static_model: PyTree | None,
    data: GraphBatch,
    denorm_fn: Callable,
    key: PRNGKeyArray,
    *,
    target_weights: dict[str, int],
    loss_name: str = "mse",
) -> float:
    """Compute the loss between the predicted and true values.

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
    model = eqx.combine(diff_model, static_model) if static_model is not None else diff_model
    y_hat = model(data, key)

    if loss_name in ["nse", "spin_up_nse"]:
        # Expand the mask with a time dimension for broadcasting.
        node_mask = data.node_mask[jnp.newaxis, :]
    else:
        node_mask = data.node_mask

    # Loop through the targets and calculate loss
    losses = []
    loss_fn = get_loss_fn(loss_name)
    for target_name, y_target in data.y.items():
        # Skip targets the model didn't predict (or aren't in target_weights)
        if target_name not in y_hat:
            continue
        if target_weights.get(target_name, 0.0) == 0.0:
            continue

        y_hat_target = y_hat[target_name]
        y_true = y_target[..., 0]

        valid_mask = ~jnp.isnan(y_true) & node_mask
        masked_y = jnp.where(valid_mask, y_true, 0)

        # Handle GMM (dict output) vs Standard (Array output)
        if isinstance(y_hat_target, dict):
            expanded_mask = jnp.expand_dims(valid_mask, axis=-1)
            masked_y_hat = {k: jnp.where(expanded_mask, v, 0) for k, v in y_hat_target.items()}
        else:
            masked_y_hat = jnp.where(valid_mask, y_hat_target[..., 0], 0)

        target_denorm_fn = partial(denorm_fn, name=target_name)
        loss = loss_fn(masked_y, masked_y_hat, valid_mask, target_denorm_fn)
        loss = jnp.nan_to_num(loss, 0) * target_weights[target_name]
        losses.append(loss)

    loss = jnp.mean(jnp.stack(losses))

    # Check for Attention Supervision
    if "attention_weights" in y_hat:
        # Re-use the node_mask for attention
        attn_loss_val = attention_leakage_loss(
            y_hat["attention_weights"],
            y_hat["valid_obs"],
            node_mask,
        )
        loss += attn_loss_val

    return loss


def get_loss_fn(loss_name):
    match loss_name:
        case "mse":
            return mse_loss
        case "mae":
            return mae_loss
        case "huber":
            return huber_loss
        case "nse":
            return nse_loss
        case "spin_up_nse":
            return spin_up_nse_loss
        case "gmm_nll":
            return gmm_nll
        case _:
            raise ValueError(f"Config'd loss ({loss_name}) not found.")


def mse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Calculates Mean Squared Error (MSE) loss."""
    mse = jnp.mean(jnp.square(y - y_hat), where=mask)
    return mse


def mae_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Calculates Mean Absolute Error (MAE) loss."""
    mae = jnp.mean(jnp.abs(y - y_hat), where=mask)
    return mae


def huber_loss(
    y: Array, y_hat: Array, mask: Array, denorm_fn: Callable, *, huber_delta: float = 1.0
):
    """Calculates Huber loss."""
    # Passing delta through make_step is not yet implemented.
    residual = y - y_hat
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def nse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """
    Calculates the smooth-joint NSE from Kratzert et al., 2019
    https://doi.org/10.5194/hess-23-5089-2019
    """
    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.mean(sq_error, axis=0)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    jnp.mean(node_nse, where=node_nse != 0)


def spin_up_nse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Equivalent to nse_loss but with weights that favor the final ~1/3 of the sequence"""
    y = denorm_fn(y)
    y_hat = denorm_fn(y_hat)

    seq_len = y.shape[0]
    weights = jax.nn.sigmoid(jnp.linspace(-10, 10, seq_len))

    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.average(sq_error, axis=0, weights=weights)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    return jnp.mean(node_nse, where=node_nse != 0)


def gmm_nll(y: Array, y_hat: dict[str, Array], mask: Array, denorm_fn: Callable) -> Array:
    """
    Calculates the average negative log-likelihood for a Gaussian Mixture Model (GMM).
    """
    y_obs = jnp.expand_dims(y, axis=-1)  # Allow broadcasting to each error model
    mu_hat = y_hat["mu"]
    sigma_hat = y_hat["sigma"]

    # Log of the normal PDF for each component:
    # log(pdf) = -log(sigma) - 0.5 * log(2*pi) - 0.5 * ((y - mu) / sigma)^2
    log_2pi = jnp.log(2.0 * jnp.pi)
    log_pdfs = -jnp.log(sigma_hat) - 0.5 * log_2pi - 0.5 * jnp.square((y_obs - mu_hat) / sigma_hat)

    # Combine with log(pi) and use logsumexp over components
    # log(sum(pi * pdf)) = logsumexp(log(pi) + log(pdf))
    log_pi = jnp.log(y_hat["pi"] + 1e-9)  # eps for stability
    log_weighted_likelihoods = jax.scipy.special.logsumexp(log_pi + log_pdfs, axis=-1)

    return -jnp.mean(log_weighted_likelihoods, where=mask)


def attention_leakage_loss(attn_weights: Array, obs_mask: Array, node_mask: Array) -> Array:
    """
    Penalizes any attention weight assigned to masked (invalid) time steps.
    Does NOT penalize the distribution over valid time steps our sources.
    """
    real_data_exists = jnp.any(obs_mask[..., :-1], axis=-1)
    null_attn_weights = attn_weights[..., -1]
    penalty = null_attn_weights * real_data_exists

    return jnp.mean(penalty, where=real_data_exists)
