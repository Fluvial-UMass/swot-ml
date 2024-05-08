import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from rich.progress import Progress
import logging
import pickle
import json
import os
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, *,
                 model_func,
                 model_args: dict, 
                 dataloader, 
                 lr_schedule, 
                 num_epochs: int, 
                 model: eqx.Module = None,
                 log_every=5, 
                 **step_kwargs):
        """        
        Initializes optimizers, logging, and sets up the training environment.
        
        Args:
            model: The model to be trained.
            dataloader: An iterable dataloader that provides batches of data.
            lr_schedule: A function that returns the learning rate given the epoch number.
            num_epochs: Total number of epochs to train the model.
            log_every: Interval of epochs after which to log and save the model state.
            **step_kwargs: Additional keyword arguments to pass to the training step function.
        """
        self.model_func = model_func
        self.model_args = model_args
        self.dataloader = dataloader
        self.lr_schedule = lr_schedule
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.step_kwargs = step_kwargs

        if model is None:
            self.model = model_func(**model_args)
        else:
            self.model = model

        self.loss_list = []
        self.epoch = 0
        self.epoch_str_fn = lambda l: f"[red]Epoch:{self.epoch} Loss:{l:.4f}"
        self.batch_str_fn = lambda b,l: f"[green]Batch:{b}/{len(dataloader)} Loss:{l:.4f}"
        
        self.optim = optax.adam(lr_schedule(self.epoch))
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        self.freeze_components(None, False)

        # Setup logging
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.log_dir = Path(f"../logs/{current_date}")
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / "training.log"
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    def start_training(self):
        """
        Starts or continues training the model.
    
        Manages the training loop, including updating progress bars, logging, handling keyboard 
        interruptions, and saving the model state at specified intervals. Catches a KeyboardInterrupt 
        to save the model state before exiting.
    
        Returns:
            The trained model after completing the training or upon an interruption.
        """ 
        try:
            with Progress() as progress:
                epoch_pbar = progress.add_task(self.epoch_str_fn(0), completed=self.epoch, total=self.num_epochs)
                batch_pbar = progress.add_task(self.batch_str_fn(0,0))
                
                while self.epoch <= self.num_epochs:
                    self.epoch += 1
                    loss, bad_grads = self._train_epoch(progress, batch_pbar)

                    logging.info(f"Time: {datetime.now()}, Epoch: {self.epoch}, Loss: {loss:.4f}")

                    # Log the counts of any bad gradients.
                    for type_key, tree_counts in bad_grads.items():
                        if tree_counts:
                            warning_str = f"{type_key} gradients detected:"
                            for tree_key, count in tree_counts.items():
                                warning_str += f"\n\t{tree_key}: {count}"
                            logging.warning(warning_str)

                    progress.update(epoch_pbar, advance=1, description=self.epoch_str_fn(loss))
                    if self.epoch % self.log_every == 0:
                        self.save_state()
                    self.loss_list.append(loss)
        except KeyboardInterrupt:
            pass

        # Cleanup and return 
        print("Training finished or interrupted. Model state saved.")
        if self.epoch % self.log_every != 0:
            self.save_state()
        plt.plot(self.loss_list)
    
    def _train_epoch(self, progress, batch_pbar):
        """
        Iterates over the dataloader batches, updates the model using the optimization step, and handles 
        any exceptions that occur during the training. Logs errors and saves error data if issues are encountered.
    
        Args:
            progress: The rich progress manager instance to update the progress bar.
            batch_pbar: The batch progress bar within the epoch.
    
        Returns:
            float: The average loss calculated over the epoch.
        """
        self.optim = optax.adam(self.lr_schedule(self.epoch))
        progress.reset(batch_pbar)
        progress.update(batch_pbar, completed=0, total=len(self.dataloader)-1, description=self.batch_str_fn(0,0))
        exception_count = 0
        consecutive_exceptions = 0
        total_loss = 0
        num_batches = 0
        bad_grads = {'vanishing': {}, 'exploding':{}}

        for basins, dates, batch in self.dataloader:
            try:
                batch = self.dataloader.shard_batch(batch)
                loss, grads, self.model, self.opt_state = make_step(self.model, batch, self.opt_state, 
                                                                    self.optim, self.filter_spec, **self.step_kwargs)

                if jnp.isnan(loss):
                    raise RuntimeError(f"NaN loss encountered")
                    
                total_loss += loss
                num_batches += 1
                consecutive_exceptions = 0
                
                # Monitor gradients
                grad_norms = jtu.tree_map(jnp.linalg.norm, grads)
                grad_norms = jtu.tree_leaves_with_path(grad_norms)
                # Check each gradient norm
                for keypath, norm in grad_norms:
                    tree_key = jtu.keystr(keypath)
                    type_key = 'vanishing' if norm < 1e-6 else 'exploding' if norm > 1e3 else None
                    if type_key is not None:
                        if tree_key not in bad_grads[type_key]:
                            bad_grads[type_key][tree_key] = 1
                        else:
                            bad_grads[type_key][tree_key] += 1

            except Exception as e:
                error_dir = self.log_dir / "exceptions" / f"epoch{self.epoch}_exception{exception_count}"
                self.save_state(error_dir)
                
                error_data = {"basins": basins,
                              "dates": dates,
                              "batch": batch}
                data_fp = error_dir / "data.pkl"
                with open(data_fp, "wb") as f:
                    pickle.dump(error_data, f)
                logging.error(f"Model step exception: {str(e)}. States and data saved to {data_fp}", exc_info=True)
                print(f"Model exception! Logged batch data and states to {data_fp}")
                exception_count += 1
                consecutive_exceptions += 1
            
            if consecutive_exceptions >= 5:
                raise RuntimeError(f"Too many consecutive exceptions ({consecutive_exceptions})")

            if num_batches > 0:
                average_loss = total_loss / num_batches
                progress.update(batch_pbar, advance=1, description=self.batch_str_fn(num_batches,average_loss))
    
        return average_loss, bad_grads

    def save_state(self, save_dir=None):
        if save_dir is None:
            save_dir = self.log_dir / f"epoch{self.epoch}"
        os.makedirs(save_dir, exist_ok=True)

        state = {'epoch': self.epoch,
                 'loss_list': self.loss_list,
                 'opt_state':self.opt_state}          
        with open(save_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)
            
        with open(save_dir / "model.eqx", "wb") as f:
            model_args_str = json.dumps(self.model_args)
            f.write((model_args_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

    def load_state(self, save_dir:str):
        with open(self.log_dir / save_dir / "state.pkl", "rb") as f:
            state = pickle.load(f)
            self.epoch = state['epoch']
            self.loss_list = state['loss_list']
            self.opt_state = state['opt_state']
             
        with open(self.log_dir / save_dir / "model.eqx", "rb") as f:
            model_args = json.loads(f.readline().decode())
            model = self.model_func(**model_args)
            self.model = eqx.tree_deserialise_leaves(f, model)

        data_path = self.log_dir / save_dir / "data.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            return data

    def load_last_state(self):
        epoch_regex = re.compile(r"epoch(\d+)")
        dirs = os.listdir(self.log_dir)
        matches = [epoch_regex.match(d) for d in dirs]
        epochs = [int(m.group(1)) for m in matches if isinstance(m, re.Match)]
        save_dir = f"epoch{max(epochs)}"
        self.load_state(save_dir)

    def freeze_components(self, component_names=None, freeze:bool=True):
        if isinstance(component_names, str):
                component_names = [component_names]
            
        # Returns True for any elements we want to be differentiable
        def diff_filter(keypath, _):
            keystr = jtu.keystr(keypath)
            # return not freeze for all components if None is passed
            if component_names is None:
                return not freeze
            # return not freeze for keystrs that exist in component_names
            elif any([component in keystr for component in component_names]):
                return not freeze
            # return True (differentiable) for any remaining components.
            else:
                return True
                
        self.filter_spec = jtu.tree_map_with_path(diff_filter, self.model)

def mse_loss(y, y_pred):
    mse = jnp.mean(jnp.square(y[...,-1] - y_pred[...,-1]))
    return mse

# Intermittent flow modified MSE
def mae_loss(y, y_pred, q):
    mse = jnp.mean(jnp.abs(y[...,-1] - y_pred[...,-1]))
    return mae

@eqx.filter_value_and_grad
def compute_loss(diff_model, static_model, data, loss_name):
    """
    Computes the loss between the model predictions and the targets using the specified loss function.

    Args:
        diff_model (equinox.Module): differentiable components of model.
        static_model (equinox.Module): static components of model.
        data (dict): Dictionary containing all input data.
        loss_name (str): The name of the loss function to use.

    Returns:
        float: The computed loss.
    """
    model = eqx.combine(diff_model, static_model)
    y_pred = jax.vmap(model)(data)

    def print_func(operand):
        data, y_pred = operand
        jax.debug.print("y: {a}, pred:{b}", a=data['y'][...,-1], b=y_pred[...,-1])
        return None

    is_nan = jnp.any(jnp.isnan(y_pred))
    jax.lax.cond(is_nan,
                 print_func,
                 lambda *args: None,
                 operand=(data, y_pred))
    
    if loss_name == "mse":
        return mse_loss(data['y'], y_pred)
    elif loss_name == "mae":
        return mae_loss(data['y'], y_pred)
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
    total_norm = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x ** 2), grads))
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
    sum_l2 = jtu.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), params, 0)
    return 0.5 * weight_decay * sum_l2

@eqx.filter_jit    
def make_step(model, data, opt_state, optim, filter_spec, loss_name="mse", max_grad_norm=None, l2_weight=None):
    """
    Performs a single optimization step, updating the model parameters.

    Args:
        model (equinox.Module): Equinox model that takes in single dict of data arrays
        data (dict): Dictionary of batched data arrays for model input
        opt_state: The state of the optimizer.
        optim: The optimizer.
        filter_spec (pytree): Pytree filter spec following structure of model. True elements will be updated.
        max_grad_norm (float, optional): The maximum norm for clipping gradients. Defaults to None.
        l2_reg (float, optional): The L2 regularization strength. Defaults to None.

    Returns:
        tuple: A tuple containing the loss, updated model, and updated optimizer state.
    """
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = compute_loss(diff_model, static_model, data, loss_name)
    
    if max_grad_norm is not None:
        grads = clip_gradients(grads, max_grad_norm)
    if l2_weight is not None:
        loss += l2_regularization(model, l2_weight)
        
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state




