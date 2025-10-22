import jax
import jax.numpy as jnp
from jaxtyping import Array


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


def mse_loss(y: Array, y_hat: Array, mask: Array):
    """Calculates Mean Squared Error (MSE) loss."""
    mse = jnp.mean(jnp.square(y - y_hat), where=mask)
    return mse


def mae_loss(y: Array, y_hat: Array, mask: Array):
    """Calculates Mean Absolute Error (MAE) loss."""
    mae = jnp.mean(jnp.abs(y - y_hat), where=mask)
    return mae


def huber_loss(y: Array, y_hat: Array, mask: Array, *, huber_delta: float = 1.0):
    """Calculates Huber loss."""
    # Passing delta through make_step is not yet implemented.
    residual = y - y_hat
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def nse_loss(y: Array, y_hat: Array, mask: Array):
    """
    Calculates the smooth-joint NSE from Kratzert et al., 2019
    https://doi.org/10.5194/hess-23-5089-2019
    """
    # Arrays can be either [batch, time, basins] or [basins, time] depending on model config.
    # Graph models use the former because they need to predict over multiple basins for each step.
    sq_error = jnp.square(y - y_hat) * mask
    std_y = jnp.std(y, axis=1, where=mask.astype(bool))  # Per-basin standard deviation
    mse = jnp.mean(sq_error, axis=1)  # mean squared errors per basin
    denom = jnp.square(
        jnp.nan_to_num(std_y) + 0.1
    )  # Denominator with smoothing and epsilon for stability
    return jnp.mean(mse / denom)


def spin_up_nse_loss(y: Array, y_hat: Array, mask: Array):
    """Equivalent to nse_loss but with weights that favor the final ~1/3 of the sequence"""
    seq_len = y.shape[0]
    weights = jax.nn.sigmoid(jnp.linspace(-10, 10, seq_len))

    sq_error = jnp.square(y - y_hat) * mask * weights[:, None]
    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    mse = jnp.mean(sq_error, axis=0)  # Sum of squared errors per basin
    denom = jnp.square(
        jnp.nan_to_num(std_y) + 0.1
    )  # Denominator with smoothing and epsilon for stability
    return jnp.mean(mse / denom)


def gmm_nll(y: Array, y_hat: dict[str, Array], mask: Array) -> Array:
    """
    Calculates the average negative log-likelihood for a Gaussian Mixture Model (GMM).
    """    
    y_unsqueezed = jnp.expand_dims(y, axis=-1) # Allow broadcasting to each error model

    # Calculate the probability density of y for each Gaussian component.
    one_over_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)
    exponent = -0.5 * jnp.square((y_unsqueezed - y_hat['mu']) / y_hat['sigma'])
    pdf_values = one_over_sqrt_2pi * jnp.exp(exponent) / y_hat['sigma']

    # Weight the PDFs by the mixture weights (pi) and sum them up
    weighted_likelihoods = jnp.sum(y_hat['pi'] * pdf_values, axis=-1)

    # negative log likelihood
    nll = -jnp.log(weighted_likelihoods + 1E-6)
    
    return jnp.mean(nll, where=mask)