import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import logging
import pickle
import json
import os
import re
import traceback
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import models

# While testing
from importlib import reload
reload(models)

class Trainer:
    """
    A class to handle training of equinox models.

    Attributes:
        cfg (dict): Configuration dictionary.
        dataloader: DataLoader object.
        log_dir (Path): Directory for logging.
        num_epochs (int): Number of epochs for training.
        log_interval (int): Interval for logging.
        lr_schedule: Learning rate scheduler.
        model: Model to be trained.
        loss_list (list): List to store loss values.
        epoch (int): Current epoch.
        optim: Optimizer.
        opt_state: Optimizer state.
        filter_spec: Specification for freezing components.
    """
    def __init__(self, cfg, dataloader=None, *, log_parent:Path=None, config_str=None, model=None):
        """        
        Initializes optimizers, logging, and sets up the training environment.

        Args:
            cfg (dict): Configuration dictionary.
            dataloader: DataLoader object.
            log_parent (Path, optional): Parent directory for logging.
            config_str (str, optional): Configuration string for logging.
            model (eqx.Module, optional): Pre-initialized model.
        """
        self.cfg = cfg
        self.dataloader = dataloader
        
        self.num_epochs = cfg['num_epochs']
        self.log_interval = cfg.get('log_interval', 5)
        
        self.lr_schedule = optax.exponential_decay(
            cfg['initial_lr'],
            cfg['num_epochs'],
            cfg['decay_rate'])

        self.loss_list = []
        self.epoch = 0
        self.train_key = jax.random.PRNGKey(cfg['model_args']['seed']+1)
         
        self.model = models.make(cfg) if model is None else model
        self.optim = optax.adam(self.lr_schedule(self.epoch))
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        #unfreeze everything
        self.freeze_components(None, False)

        # Setup logging
        if cfg['log']:
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            if log_parent is not None:
                if cfg.get('cfg_path') is not None:
                     self.log_dir = log_parent / f"{cfg.get('cfg_path').stem}_{current_date}"
                else:
                    self.log_dir = log_parent / current_date
            else:     
                self.log_dir = Path(f"../runs/notebook/{current_date}")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Logging at {self.log_dir}")
            
            cfg_file = self.log_dir / "config.pkl"
            with open(cfg_file, 'wb') as file:
                pickle.dump(self.cfg, file)

            log_file = self.log_dir / "training.log"
            logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', force=True)
            if config_str is not None:
                logging.info("Run Configuration\n"+config_str)

    def start_training(self):
        """
        Starts or continues training the model.

        Manages the training loop, including updating progress bars, logging, handling keyboard 
        interruptions, and saving the model state at specified intervals.

        Returns:
            The trained model after completing the training or upon an interruption.
        """
        if self.dataloader is None:
            raise RuntimeError(f"Trainer objects without a dataloader can only be used for loading model states.")
        
        try:
            while self.epoch < self.num_epochs:
                self.epoch += 1
                loss, bad_grads = self._train_epoch()

                info_str = f"Epoch: {self.epoch}, Loss: {loss:.4f}"
                if self.cfg['log']:
                    logging.info(info_str)
                if self.cfg['quiet']:
                    print(info_str)

                # Log the counts of any bad gradients.
                for type_key, tree_counts in bad_grads.items():
                    if tree_counts:
                        warning_str = f"{type_key} gradients detected:"
                        for tree_key, count in tree_counts.items():
                            warning_str += f"\n\t{tree_key}: {count}"

                        if self.cfg['log']:
                            logging.warning(warning_str)
                        else:
                            print(warning_str)

                if self.cfg['log'] & (self.epoch % self.log_interval == 0):
                    self.save_state()
                    
                self.loss_list.append(float(loss))
        except KeyboardInterrupt:
            pass

        # Cleanup and return 
        print("Training finished or interrupted. Model state saved.")
        if self.cfg['log']:
            if self.epoch % self.log_interval != 0:
                self.save_state()
            logging.info("~~~ training stopped ~~~")
            
        plt.plot(self.loss_list)
        plt.show()
    
    def _train_epoch(self):
        """
        Iterates over the dataloader batches, updates the model using the optimization step, and handles 
        any exceptions that occur during the training. Logs errors and saves error data if issues are encountered.

        Returns:
            float: The average loss calculated over the epoch.
        """
        lr = self.lr_schedule(self.epoch)
        self.optim = optax.adam(lr)
        consecutive_exceptions = 0
        batch_count = 0
        losses = []
        bad_grads = {'vanishing': {}, 'exploding':{}}

        pbar = tqdm(self.dataloader, disable=self.cfg['quiet'], desc=f"Epoch:{self.epoch:03.0f}")
        for data_tuple in pbar:
            basins, dates, batch = data_tuple
            batch = self.dataloader.shard_batch(batch)
            batch_count += 1

            # Split and update training key for dropout
            keys = jax.random.split(self.train_key, self.cfg['batch_size']+1)
            self.train_key = keys[0]
            batch_keys = keys[1:] 
            try: 
                loss, grads, self.model, self.opt_state = make_step(
                    self.model, 
                    batch,
                    batch_keys,
                    self.opt_state, 
                    self.optim, 
                    self.filter_spec, 
                    **self.cfg['step_kwargs']
                )

                if jnp.isnan(loss):
                    raise RuntimeError(f"NaN loss encountered")

                pbar.set_postfix_str(f"Loss:{loss:0.04f}")
                
                losses.append(loss)
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
                consecutive_exceptions += 1
                
                if self.cfg['log']:
                    error_dir = self.log_dir / "exceptions" / f"epoch{self.epoch}_batch{batch_count}"
                    self.save_state(error_dir)

                    with open(error_dir / "data.pkl", "wb") as f:
                        pickle.dump(data_tuple, f)
                    with open(error_dir / "exception.txt", "w") as f:
                        f.write(f"{str(e)}\n{traceback.format_exc()}")
                    error_str = f"{type(e).__name__} exception caught. See {error_dir} for data, model state, and trace."
                    logging.error(error_str)
                    
                else:
                    error_str = f"{str(e)}\n{traceback.format_exc()}"
                print(error_str)

            if consecutive_exceptions >= 3:
                raise RuntimeError(f"Too many consecutive exceptions ({consecutive_exceptions})")
            
            if len(losses) > 0:
                loss_arr = np.array(losses)
                mean_loss = loss_arr.mean()
                std_loss = loss_arr.std()
                z_score = ((loss-mean_loss)/std_loss)
                if (z_score > 10) and (len(losses) >= 10):
                    warning_str = f"Anomalous batch loss ({loss:0.4f}, z-score of {z_score:0.1f})."
                    if self.cfg['log']:
                        warning_dir = self.log_dir / "anomalies" / f"epoch{self.epoch}_batch{len(losses)}"
                        self.save_state(warning_dir)

                        with open(warning_dir / "data.pkl", "wb") as f:
                            pickle.dump(data_tuple, f)
                        warning_str += f"See {warning_dir.relative_to(self.log_dir)} for data and model state."
                        logging.warning(warning_str)
                    print(warning_str)
                    
            if (pbar.n+1) == pbar.total:
                pbar.set_postfix_str(f"Avg Loss:{mean_loss:0.04f}")
                pbar.refresh()
                
        return mean_loss, bad_grads
        
    
    def save_state(self, save_dir=None):
        if save_dir is None:
            save_dir = self.log_dir / f"epoch{self.epoch:03d}"
        os.makedirs(save_dir, exist_ok=True)
            
        with open(save_dir / "model.eqx", "wb") as f:
            model_args_str = json.dumps(self.cfg['model_args'])
            f.write((model_args_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

        with open(save_dir / "state.eqx", "wb") as f:
            state = {'epoch': self.epoch, 'loss_list': self.loss_list}
            state_str = json.dumps(state)
            f.write((state_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.opt_state)
   
    def load_state(self, save_dir:str):
        state_dir = self.log_dir / save_dir
        cfg, model, trainer_state, opt_state, data = load_state(state_dir, cfg)
        
        self.model = mode
        self.epoch = trainer_state['epoch']
        self.loss_list = trainer_state['loss_list']
        self.opt_state = opt_state
        return data
    
    def load_last_state(self):
        save_dir = _last_epoch_dir(self.log_dir)
        self.load_state(save_dir)

    def freeze_components(self, component_names=None, freeze:bool=True):
        # Updates the filterspec to set which parmaters can be updated.
        # Only accepts top-level element names in the pytree model. 
        
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
       
    
def load_state(state_dir, cfg=None):
    print(f"Loading model state from {state_dir}")
    if cfg is None:
        cfg_file = state_dir.parent / "config.pkl"
        with open(cfg_file, 'rb') as file:
            cfg = pickle.load(file)        
    lr_schedule = optax.exponential_decay(
        cfg['initial_lr'],
        cfg['num_epochs'],
        cfg['decay_rate'])
    
    with open(state_dir / "model.eqx", "rb") as f:
        model_args = json.loads(f.readline().decode())
        serialized_model = models.make(cfg)
        model = eqx.tree_deserialise_leaves(f, serialized_model)
    
    with open(state_dir / "state.eqx", "rb") as f:
        trainer_state = json.loads(f.readline().decode())
        optim = optax.adam(lr_schedule(trainer_state['epoch']))   
        serialized_opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        opt_state = eqx.tree_deserialise_leaves(f, serialized_opt_state)

    out = (cfg, model, trainer_state, opt_state)
    # Load any data that were stored alongside this record.
    data_path = state_dir / "data.pkl"
    if os.path.exists(data_path):
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
            out = (*out, data)
    return out


def _last_epoch_dir(log_dir):
    epoch_regex = re.compile(r"epoch(\d+)")
    dirs = os.listdir(log_dir)
    matches = [epoch_regex.match(d) for d in dirs]
    epoch_strs = [m.group(1) for m in matches if isinstance(m, re.Match)]
    last_epoch_idx = np.argmax([int(s) for s in epoch_strs])
    
    return log_dir / f"epoch{epoch_strs[last_epoch_idx]}"
    
def load_last_state(log_dir):
    save_dir = _last_epoch_dir(log_dir)
    return load_state(save_dir)

    

    
    
def mse_loss(y, y_pred, mask):
    mse = jnp.mean(jnp.square(y - y_pred), where=mask)
    return mse

def mae_loss(y, y_pred, mask):
    mae = jnp.mean(jnp.abs(y - y_pred), where=mask)
    return mae

def huber_loss(y, y_pred, mask, delta=1.0):
    residual = y - y_pred
    condition = jnp.abs(residual) <= delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * (jnp.abs(residual) - 0.5 * delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)

@eqx.filter_value_and_grad
def compute_loss(diff_model, static_model, data, keys, loss_name):
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
    y_pred = jax.vmap(model)(data, keys)

    y = data['y'][:,-1,:]
    valid_mask = ~jnp.isnan(data['y'][:,-1,:])
    masked_y = jnp.where(valid_mask, y, 0)
    masked_y_pred = jnp.where(valid_mask, y_pred, 0)
    if loss_name == "mse":
        return mse_loss(masked_y, masked_y_pred, valid_mask)
    elif loss_name == "mae":
        return mae_loss(masked_y, masked_y_pred, valid_mask)
    elif loss_name == "huber":
        return huber_loss(masked_y, masked_y_pred, valid_mask)
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
def make_step(model, data, keys, opt_state, optim, filter_spec, loss="mse", max_grad_norm=None, l2_weight=None):
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
    loss, grads = compute_loss(diff_model, static_model, data, keys, loss)
    
    if max_grad_norm is not None:
        grads = clip_gradients(grads, max_grad_norm)
    if l2_weight is not None:
        loss += l2_regularization(model, l2_weight)
        
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, grads, model, opt_state




