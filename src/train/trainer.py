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
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from importlib import reload
import models
import train.step
reload(models)
reload(train.step)

from .step import make_step


class Trainer:
    """
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
    def __init__(self, cfg, dataloader, *, log_parent=None, log_dir=None, continue_from=None):
        """        
        Initializes optimizers, logging, and sets up the training environment.

        Args:
            cfg (dict): Configuration dictionary.
            dataloader: DataLoader object.
            log_parent (Path, optional): Parent directory for logging.
            continue_from (Path, optional): Directory that contains epoch data
        """
        self.cfg = cfg
        self.dataloader = dataloader
        
        if cfg['log']: 
            self.setup_logging(log_parent, log_dir, continue_from)

        self.num_epochs = cfg['num_epochs']
        self.log_interval = cfg.get('log_interval', 5)
        
        self.lr_schedule = optax.exponential_decay(
            cfg['initial_lr'],
            cfg['num_epochs'],
            cfg['decay_rate'],
            cfg.get('transition_begin',0))

        self.train_key = jax.random.PRNGKey(cfg['model_args']['seed']+1)

        model_loaded = False
        if continue_from is not None:
            model_loaded, _ = self.load_last_state(continue_from)

        if not model_loaded:
            self.model = models.make(cfg)
            self.loss_list = []
            self.epoch = 0

        self.optim = optax.adam(self.lr_schedule(self.epoch))
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        # Set filterspec to train all components.
        self.freeze_components(None, False)
            
    def setup_logging(self, log_parent=None, log_dir=None, continue_from=None):
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_parent is not None:
            self.log_dir = log_parent / f"{self.cfg.get('cfg_path').stem}_{current_date}"
        elif log_dir is not None:
            self.log_dir = log_dir
        elif continue_from:
            self.log_dir = continue_from
        else:     
            self.log_dir = Path(f"../runs/notebook/{current_date}")
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging at {self.log_dir}")
        
        cfg_file = self.log_dir / "config.pkl"
        with open(cfg_file, 'ab') as file:
            pickle.dump(self.cfg, file)

        log_file = self.log_dir / "training.log"
        logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    def start_training(self, stop_at=np.inf):
        """
        Starts or continues training the model.

        Manages the training loop, including updating progress bars, logging, handling keyboard 
        interruptions, and saving the model state at specified intervals.

        Returns:
            The trained model after completing the training or upon an interruption.
        """
        while (self.epoch < self.num_epochs) and (self.epoch < stop_at):
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

        print("Training finished. Model state saved.")
        if self.cfg['log']:
            if self.epoch % self.log_interval != 0:
                self.save_state()
            logging.info("~~~ training stopped ~~~")
    
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
                    self.dataloader.dataset.denormalize_target,
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
                    
        pbar.set_postfix_str(f"Avg Loss:{np.mean(losses):0.04f}")
        pbar.refresh()
        
        return np.mean(losses), bad_grads
        
    
    def save_state(self, save_dir=None):
        if save_dir is None:
            save_dir = self.log_dir / f"epoch{self.epoch:03d}"
        os.makedirs(save_dir, exist_ok=True)
            
        with open(save_dir / "model.eqx", "wb") as f:
            model_args = self.cfg['model_args']
            if isinstance(model_args.get('graph_matrix'), np.ndarray):
                model_args['graph_matrix'] = model_args['graph_matrix'].tolist()

            model_args_str = json.dumps(model_args)
            f.write((model_args_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

        with open(save_dir / "state.eqx", "wb") as f:
            state = {'epoch': self.epoch, 'loss_list': self.loss_list}
            state_str = json.dumps(state)
            f.write((state_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.opt_state)
   
    def load_state(self, epoch_dir:Path):
        model_loaded, load_tuple = load_state(epoch_dir, self.cfg)
        if not model_loaded:
            return False, None
        cfg, model, trainer_state, opt_state, data = load_tuple
        self.model = model
        self.epoch = trainer_state['epoch']
        self.loss_list = trainer_state['loss_list']
        self.opt_state = opt_state
        return True, data
    
    def load_last_state(self, log_dir:Path):
        epoch_dir = _last_epoch_dir(log_dir)
        return self.load_state(epoch_dir)

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

def load_state(state_dir: Path, cfg=None):
    if not isinstance(state_dir, Path) or not state_dir.is_dir():
        print('Model state directory not found!')
        return False, None

    print(f"Loading model state from {state_dir}")
    if cfg is None:
        cfg_file = state_dir.parent / "config.pkl"
        with open(cfg_file, 'rb') as file:
            cfg = pickle.load(file)        
    lr_schedule = optax.exponential_decay(
        cfg['initial_lr'],
        cfg['num_epochs'],
        cfg['decay_rate'],
        cfg.get('transition_begin',0))
    
    with open(state_dir / "model.eqx", "rb") as f:
        model_args = json.loads(f.readline().decode())
        if 'graph_matrix' in model_args.keys():
            model_args['graph_matrix'] = np.array(model_args['graph_matrix'])
        cfg['model_args'] = model_args

        serialized_model = models.make(cfg)
        # Ensure all leaves are jnp float 32s.
        # Bandaid for some poorly specified graph adjacency matrices
        serialized_model = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            serialized_model
        )
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
    else:
        out = (*out, None)
    return True, out


def _last_epoch_dir(log_dir):
    epoch_regex = re.compile(r"epoch(\d+)")
    dirs = os.listdir(log_dir)
    matches = [epoch_regex.match(d) for d in dirs]
    epoch_strs = [m.group(1) for m in matches if isinstance(m, re.Match)]
    if len(epoch_strs)==0:
        return None
    else:
        last_epoch_idx = np.argmax([int(s) for s in epoch_strs])
        return log_dir / f"epoch{epoch_strs[last_epoch_idx]}"
    
def load_last_state(log_dir):
    save_dir = _last_epoch_dir(log_dir)
    return load_state(save_dir)




