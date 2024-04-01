import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

def mse_loss(y, y_pred):
    mse = jnp.mean(jnp.square(y[:,-1] - y_pred[:,-1]))
    return mse

# Intermittent flow modified MSE
def if_mse_loss(y, y_pred, unscaled_q):
    mse = mse_loss(y, y_pred)
    if_err = jnp.where(unscaled_q==0,
                       y_pred-0,
                       jnp.nan)
    if_mse = jnp.nanmean(jnp.square(if_err))
    
    loss = jnp.nansum(jnp.array([mse, if_mse*0.1]))
    return loss

@eqx.filter_value_and_grad
def compute_loss(model, data, loss_name, **kwargs):
    """
    Computes the loss between the model predictions and the targets using the specified loss function.

    Args:
        model (LSTM): The LSTM model.
        data (dict): Dictionary containing all input data.
        loss_name (str): The name of the loss function to use.
        **kwargs: Additional keyword arguments to pass to the loss function.

    Returns:
        float: The computed loss.
    """
    y_pred = jax.vmap(model)(data)
    if loss_name == "mse":
        return mse_loss(data['y'], y_pred)
    elif loss_name == "if_mse":
        return if_mse_loss(data['y'], y_pred, data['unscaled_q'])
    else:
        raise ValueError("Invalid loss function name.")

@eqx.filter_jit
def make_step(model, data, opt_state, optim, loss_name="mse", max_grad_norm=None, l2_weight=None):
    """
    Performs a single optimization step, updating the model parameters.

    Args:
        model (LSTM): The LSTM model.
        x tuple(jax.Array): The input data.
        y (jax.Array): The target data with shape [batch_size, out_size].
        opt_state: The state of the optimizer.
        optim: The optimizer.
        max_grad_norm (float, optional): The maximum norm for clipping gradients. Defaults to None.
        l2_reg (float, optional): The L2 regularization strength. Defaults to None.

    Returns:
        tuple: A tuple containing the loss, updated model, and updated optimizer state.
    """
    loss, grads = compute_loss(model, data, loss_name)
    
    if max_grad_norm is not None:
        grads = clip_gradients(grads, max_grad_norm)
    if l2_weight is not None:
        loss += l2_regularization(model, l2_weight)
        
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def lr_dict_scheduler(epoch, lr_dict):
    """
    Returns the learning rate for a given epoch based on a multi-step schedule.

    Args:
        epoch (int): The current epoch.
        lr_dict (dict): A dictionary mapping epochs to learning rates.

    Returns:
        float: The learning rate for the given epoch.
    """
    current_lr = None
    for milestone_epoch in sorted(lr_dict.keys(), reverse=True):
        if epoch >= milestone_epoch:
            current_lr = lr_dict[milestone_epoch]
            break
    return current_lr if current_lr is not None else lr_dict[0]

def clip_gradients(grads, max_norm):
    """
    Clip gradients to prevent them from exceeding a maximum norm.

    Args:
        grads (jax.grad): The gradients to be clipped.
        max_norm (float): The maximum norm for clipping.
        
    Returns:
        jax.grad: The clipped gradients.
    """
    total_norm = jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda x: jnp.sum(x ** 2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jax.tree_map(lambda g: scale * g, grads)


def l2_regularization(model, weight_decay):
    """
    Computes the L2 regularization term for a pytree model.

    Args:
        model: A pytree model where tunable parameters are inexact arrays
        weight_decay (float): The weight decay coefficient (lambda) for L2 regularization.

    Returns:
        float: The L2 regularization term.
    """
    params = eqx.filter(model, eqx.is_inexact_array)
    sum_l2 = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), params, 0)
    return 0.5 * weight_decay * sum_l2


