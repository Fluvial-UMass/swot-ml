import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jutil
import numpy as np
from functools import partial
from tqdm.notebook import trange, tqdm
import logging
import pickle
from pathlib import Path
from datetime import datetime



def mse_loss(y, y_pred):
    mse = jnp.mean(jnp.square(y[:,-1] - y_pred[:,-1]))
    return mse

# Intermittent flow modified MSE
def if_mse_loss(y, y_pred, q):
    """
    Computes the intermittent flow modified mean squared error loss.

    Args:
        y (jax.Array): The true target values.
        y_pred (jax.Array): The predicted target values.
        q (jax.Array): The flow rate values.

    Returns:
        jax.Array: The intermittent flow modified mean squared error loss.
    """
    mse = mse_loss(y, y_pred)
    if_err = jnp.where(q==0,
                       y_pred-0,
                       jnp.nan)
    if_mse = jnp.nanmean(jnp.square(if_err))
    
    loss = jnp.nansum(jnp.array([mse, if_mse*0.1]))
    return loss

@eqx.filter_value_and_grad
def compute_loss(model, data, loss_name):
    """
    Computes the loss between the model predictions and the targets using the specified loss function.

    Args:
        model (LSTM): The LSTM model.
        data (dict): Dictionary containing all input data.
        loss_name (str): The name of the loss function to use.

    Returns:
        float: The computed loss.
    """
    y_pred = jax.vmap(model)(data)
    if loss_name == "mse":
        return mse_loss(data['y'], y_pred)
    elif loss_name == "if_mse":
        return if_mse_loss(data['y'], y_pred, data['x_dd'][:,-1])
    else:
        raise ValueError("Invalid loss function name.")
       
    
def clip_gradients(grads, max_norm):
    """
    Clip gradients to prevent them from exceeding a maximum norm.

    Args:
        grads (jax.grad): The gradients to be clipped.
        max_norm (float): The maximum norm for clipping.
        
    Returns:
        jax.grad: The clipped gradients.
    """
    total_norm = jutil.tree_reduce(lambda x, y: x + y, jutil.tree_map(lambda x: jnp.sum(x ** 2), grads))
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
    sum_l2 = jutil.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), params, 0)
    return 0.5 * weight_decay * sum_l2

@eqx.filter_jit    
def make_step(model, data, opt_state, optim, loss_name="mse", max_grad_norm=None, l2_weight=None):
    """
    Performs a single optimization step, updating the model parameters.

    Args:
        model (equinox.Module): Equinox model that takes in single dict of data arrays
        data (dict): Dictionary of batched data arrays for model input
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


def start_training(model, dataloader, lr_schedule, num_epochs, *,
                   start_epoch=1, 
                   opt_state=None, 
                   **kwargs):
    """
    Starts or continues the training of a given model.

    Args:
        model (eqx.Module): The model to be trained.
        dataloader (iterable): An iterable that provides batches of data.
        lr_schedule (function): A function that takes an epoch index and returns the learning rate.
        num_epochs (int): The number of epochs to train for.
        start_epoch (int, optional): The epoch index to start training from. Defaults to 0.
        opt_state (optax.OptState, optional): Optional initial state of the optimizer.
        **kwargs: Additional keyword arguments passed to the `make_step` function.

    Returns:
        eqx.Module: The trained model.
    """
    # Set up logging
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = Path(f"../logs/{current_date}") 
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "training.log"
    logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize optimizer 
    optim = optax.adam(lr_schedule(start_epoch))
    if opt_state is None:
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    error_count = 0
    loss_list = []
    dataloader.train = True
    for epoch in range(start_epoch, num_epochs):
        current_lr = lr_schedule(epoch)
        optim = optax.adam(current_lr)
        total_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch:{epoch} LR:{current_lr:0.4f}")
        for basins, dates, batch in pbar: 
            try:
                loss, model, opt_state = make_step(model, batch, opt_state, optim, **kwargs)
                total_loss += loss
                num_batches += 1
                pbar.set_postfix_str(f"Loss: {total_loss/num_batches:.4f}")
            except Exception as e:
                error_data = {"epoch": epoch,
                              "basins": basins,
                              "dates": dates,
                              "batch": batch,
                              "model_state": model,
                              "opt_state": opt_state}
                error_file = log_dir / f"error{error_count}_data.pkl"
                with open(error_file, "wb") as f:
                    pickle.dump(error_data, f)
                logging.error(f"Model step error: {str(e)}. Error data saved to {error_file}", exc_info=True)
                print(f"Model exception! Logged batch data and states to {error_file}")
                error_count += 1
                

        current_loss = total_loss / num_batches
        loss_list.append(current_loss)
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Time: {now}, Epoch: {epoch}, Loss: {current_loss:.4f}")
        
        # Save model state every 5 Epochs
        if (epoch % 5 == 0):
            model_state_file = log_dir / f"model_state_epoch{epoch}.pkl"
            with open(model_state_file, "wb") as f:
                pickle.dump(model, f)

    return model