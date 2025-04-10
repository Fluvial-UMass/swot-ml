import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

import numpy as np
import logging
import pickle
import json
import os
import sys
import re
import traceback
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime

import models
from .step import make_step, compute_loss_fn
from .early_stop import EarlyStopper


class Trainer:
    """Trainer class for training hydrological models.

    Attributes
    ----------
    cfg : dict
        Configuration dictionary.
    dataloader : data.HydroDataLoader
        DataLoader object.
    log_dir : Path
        Directory for logging.
    num_epochs : int
        Number of epochs for training.
    early_stopper: EarlyStopper
        Object for detecting early stopping conditions.
    lr_schedule : optax.Schedule
        Learning rate scheduler.
    model : eqx.Module
        Model to be trained.
    losses : list
        List to store loss values.
    epoch : int
        Current epoch.
    optim : optax.GradientTransformation
        Optimizer.
    opt_state : optax.OptState
        Optimizer state.
    filter_spec : PyTree
        Specification for freezing components.

    Methods
    -------
    __init__(cfg, dataloader, log_dir=None)
        Initializes the trainer.
    setup_logging(log_dir=None)
        Sets up logging for training.
    start_training(stop_at=np.inf)
        Starts or continues training the model.
    _train_epoch()
        Trains the model for one epoch.
    save_state(save_dir=None)
        Saves the model and trainer state.
    load_state(epoch_dir)
        Loads the model and trainer state.
    load_last_state(log_dir)
        Loads the last saved model and trainer state.
    freeze_components(component_names=None, freeze=True)
        Freezes or unfreezes specified components of the model.
    """
    cfg: dict
    logger: logging.Logger
    dataloader: "data.HydroDataLoader"
    log_dir: Path
    num_epochs: int
    lr_schedule: optax.Schedule
    model: eqx.Module
    losses: list
    epoch: int
    optim: optax.GradientTransformation  # Store the optimizer object
    opt_state: optax.OptState
    early_stopper: EarlyStopper | None  # Use the specific class
    filter_spec: PyTree | None  # Can be None if no filter needed
    train_key: jax.random.PRNGKey

    def __init__(self,
                 cfg: dict,
                 dataloader: "data.HydroDataLoader" = None,
                 *,
                 log_dir: Path | None = None,
                 checkpoint: dict | None = None):
        """Initializes the Trainer.

        Sets up logging, the learning rate schedule, the model, the optimizer, and the
        optimizer state.  Handles loading from a previous state if specified.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        dataloader : data.HydroDataLoader
            DataLoader object.
        log_dir : Path, optional
            Specific directory for logging.
        continue_from : Path, optional
            Directory containing a previous training state to load.
        static_leaves: list, optional
            List of top-level PyTree leaves that will be frozen during training.
            Defaults to none. 
        """
        self.cfg = cfg
        self.dataloader = dataloader

        self.log_dir = self._setup_logging(log_dir)

        self.num_epochs = cfg['num_epochs']
        self.log_interval = cfg.get('log_interval', 5)
        self.validate_interval = cfg.get('validate_interval', 5)
        self.lr_schedule = _create_lr_schedule(cfg)

        seed = cfg['model_args']['seed'] + 1
        self.train_key = jax.random.PRNGKey(seed)

        if checkpoint:
            self.epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.model = checkpoint['model']
            self.optim = checkpoint['optim']
            self.opt_state = checkpoint['opt_state']
            self.early_stopper = checkpoint['early_stopper']
        else:
            self.epoch = 0
            self.losses = []
            self.model = models.make(cfg)
            self.optim = optax.adam(self.lr_schedule(self.epoch))
            self.opt_state = self.optim.init(
                eqx.filter(self.model, eqx.is_inexact_array))

            if cfg.get('early_stopping'):
                self.early_stopper = EarlyStopper(**cfg['early_stopping'])
            else:
                None

        # Initialize the filterspec. Defaults to training all components.
        self.freeze_components([])

    def _setup_logging(self, log_dir=None):
        """Sets up logging for training.

        Creates the log directory and configures logging to a file.

        Parameters
        ----------
        log_dir : Path, optional
            Specific directory for logging.

        Returns
        -------
        log_dir : Path
            The logging directory.
        """

        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)  # Output to standard out
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if self.cfg['log']:
            if log_dir is None:
                cfg_path = self.cfg.get('cfg_path')
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = cfg_path.parent / f"{cfg_path.stem}_{current_date}"

            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Logging at {log_dir}")

            cfg_file = log_dir / "config.pkl"
            with open(cfg_file, 'ab') as file:
                pickle.dump(self.cfg, file)

            log_file = log_dir / "training.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        return log_dir

    def _cleanup_logger(self):
        # Check if the logger attribute exists and is actually a logger instance
        if hasattr(self, 'logger') and isinstance(self.logger, logging.Logger):
            # Iterate over a *copy* of the handlers list ([ : ])
            # because we are modifying the list during iteration.
            for handler in self.logger.handlers[:]:
                try:
                    # Flush and close the handler to release resources (e.g., file handles)
                    handler.flush()
                    # Check if handler has close method before calling
                    if hasattr(handler, 'close'):
                        handler.close()
                except Exception as e:
                    # Log error to stderr, as the logger itself might be problematic
                    print(f"Warning: Error closing handler {handler}: {e}",
                          file=sys.stderr)
                finally:
                    # Ensure the handler is removed even if closing failed
                    self.logger.removeHandler(handler)
        self.logger = None

    def start_training(self, stop_at=np.inf):
        """Starts or continues training the model.

        Manages the training loop, including updating progress bars, logging, handling
        keyboard interruptions, and saving the model state at specified intervals.

        Parameters
        ----------
        stop_at : float, optional
            Epoch to stop training.

        Returns
        -------
        model : eqx.Module
            The trained model.
        """
        while (self.epoch < self.num_epochs) and (self.epoch < stop_at):
            self.epoch += 1
            loss, bad_grads = self._train_epoch()
            self.losses.append(float(loss))
            self.logger.info(f"Epoch: {self.epoch}, Loss: {loss:.4f}")

            # Log the counts of any bad gradients.
            for type_key, tree_counts in bad_grads.items():
                if tree_counts:
                    warning_str = f"{type_key} gradients detected:"
                    for tree_key, count in tree_counts.items():
                        warning_str += f"\n\t{tree_key}: {count}"
                    self.logger.info(warning_str)

            if (self.epoch == 0) or (self.epoch % self.validate_interval == 0):
                v_loss = self.get_validation_loss()
                self.logger.info(f"Epoch: {self.epoch}, Validation Loss: {v_loss:.4f}")
            else:
                v_loss = None

            if v_loss and self.early_stopper:
                if self.early_stopper(v_loss):
                    self.logger.info("Training stopped by EarlyStopper.")
                    self.cfg['num_epochs'] = self.epoch
                    self.save_state()
                    break  # exit training loop

            if (self.epoch % self.log_interval == 0):
                self.save_state()

        if (self.epoch % self.log_interval != 0):
            self.save_state()
        self.logger.info("~~~ training done ~~~")
        self._cleanup_logger()

    def _train_epoch(self) -> tuple[float, dict[str, dict]]:
        """Trains the model for one epoch.

        Iterates over the dataloader batches, updates the model using the optimization
        step, and handles any exceptions that occur during the training. Logs errors and
        saves error data if issues are encountered.

        Returns
        -------
        loss : float
            The average loss for the epoch.
        bad_grads : dict[str, dict[str:int]]
            A dictionary of vanishing and exploding gradients, organized by model layer.
        """
        lr = self.lr_schedule(self.epoch)
        self.optim = optax.adam(lr)
        consecutive_exceptions = 0
        batch_count = 0
        losses = []
        bad_grads = {'vanishing': {}, 'exploding': {}}

        pbar = tqdm(self.dataloader,
                    disable=self.cfg['quiet'],
                    desc=f"Epoch:{self.epoch:03.0f}")
        for data_tuple in pbar:
            basins, dates, batch = data_tuple
            # batch = self.dataloader.shard_batch(batch)
            batch_count += 1

            # Split and update training key for dropout
            keys = jax.random.split(self.train_key, self.cfg['batch_size'] + 1)
            self.train_key = keys[0]
            batch_keys = keys[1:]
            try:
                loss, grads, self.model, self.opt_state = make_step(
                    self.model, batch, batch_keys, self.opt_state, self.optim,
                    self.filter_spec, self.dataloader.dataset.denormalize_target,
                    **self.cfg['step_kwargs'])

                if jnp.isnan(loss):
                    raise RuntimeError(f"NaN loss encountered")

                pbar.set_postfix_str(f"Loss:{loss:0.04f}")
                losses.append(loss)
                exceptions = 0

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
                exceptions += 1

                if self.cfg['log']:
                    error_dir = self.log_dir / "exceptions" / f"epoch{self.epoch}_batch{batch_count}"
                    self.save_state(error_dir)

                    with open(error_dir / "data.pkl", "wb") as f:
                        pickle.dump(data_tuple, f)
                    with open(error_dir / "exception.txt", "w") as f:
                        f.write(f"{str(e)}\n{traceback.format_exc()}")
                    error_str = f"{type(e).__name__} exception caught. See {error_dir} for data, model state, and trace."
                else:
                    error_str = f"{str(e)}\n{traceback.format_exc()}"

                self.logger.error(error_str)

            if exceptions >= 3:
                raise RuntimeError(f"Too many consecutive exceptions ({exceptions})")

        pbar.set_postfix_str(f"Avg Loss:{np.mean(losses):0.04f}")
        pbar.refresh()

        return np.mean(losses), bad_grads

    def get_validation_loss(self) -> float:
        # Set model and dataloader for inference
        self.model = eqx.nn.inference_mode(self.model, True)
        self.dataloader.dataset.update_indices('test')

        batch_keys = jax.random.split(self.train_key, self.cfg['batch_size'])
        losses = []
        pbar = tqdm(self.dataloader,
                    disable=self.cfg['quiet'],
                    desc=f"Validating Epoch:{self.epoch:03.0f}")

        for _, _, batch in pbar:
            diff_model, static_model = eqx.partition(self.model, self.filter_spec)
            loss = compute_loss_fn(diff_model, static_model, batch, batch_keys,
                                   self.dataloader.dataset.denormalize_target,
                                   **self.cfg['step_kwargs'])
            losses.append(loss)

        # Reset model and dataloader for training
        self.model = eqx.nn.inference_mode(self.model, False)
        self.dataloader.dataset.update_indices('train')

        return np.mean(losses)

    def freeze_components(self, component_names: list[str] | str = []):
        """Freezes or unfreezes specified components of the model.

        Updates the filter specification to control which parameters are updated
        during training. Only accepts top-level element names in the pytree model.

        Parameters
        ----------
        component_names : list[str] | str, optional
            List of component names to freeze. If not passed, all components are unfrozen.
        """
        if isinstance(component_names, str):
            component_names = [component_names]

        # Returns True for any elements we want to be differentiable
        def diff_filter(keypath, _):
            keystr = jtu.keystr(keypath)
            # return not freeze for all components if None is passed
            if component_names is None:
                return True
            # return not freeze for keystrs that exist in component_names
            elif any([component in keystr for component in component_names]):
                return False
            # return True (differentiable) for any remaining components.
            else:
                return True

        self.filter_spec = jtu.tree_map_with_path(diff_filter, self.model)

    # --- Methods for Saving/Loading State ---

    def save_state(self, save_dir: Path | None = None) -> None:
        """Saves the model and trainer state.

        Saves the model, optimizer state, epoch number, and loss list to the specified
        directory.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to save the state. If None, saves to a directory within the log
            directory named for the current epoch.
        """
        if not self.cfg['log']:
            return

        if save_dir is None:
            save_dir = self.log_dir / f"epoch{self.epoch:03d}"
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir / "model_and_opt.eqx", "wb") as f:
            model_args = self.cfg['model_args']
            if isinstance(model_args.get('graph_matrix'), np.ndarray):
                model_args['graph_matrix'] = model_args['graph_matrix'].tolist()

            model_args_str = json.dumps(model_args)
            f.write((model_args_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)
            eqx.tree_serialise_leaves(f, self.opt_state)

        with open(save_dir / "trainer_state.json", 'w') as f:
            state = {
                'epoch': self.epoch,
                'losses': self.losses,
            }
            if self.early_stopper:
                state['early_stopper'] = self.early_stopper.get_state()

            json.dump(state, f, default=float)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: Path):
        """Loads the trainer state from a checkpoint directory and returns a new Trainer instance."""

        # --- Load Config ---
        with open(checkpoint_dir.parent / "config.pkl", 'rb') as f:
            cfg = pickle.load(f)
        lr_schedule = _create_lr_schedule(cfg)

        # --- Load Trainer State (JSON) ---
        with open(checkpoint_dir / "trainer_state.json", 'r') as f:
            trainer_state_data = json.load(f)

        epoch = trainer_state_data['epoch']
        losses = trainer_state_data['losses']

        stopper_state = trainer_state_data.get('early_stopper', None)
        if stopper_state:
            early_stopper = EarlyStopper.from_state(stopper_state)
        else:
            early_stopper = None

        # --- Load Model and Optimizer State ---
        with open(checkpoint_dir / "model_and_opt.eqx", "rb") as f:
            model_args = json.loads(f.readline().decode())
            if 'graph_matrix' in model_args:
                model_args['graph_matrix'] = np.array(model_args['graph_matrix'])

            cfg['model_args'] = model_args
            serialized_model = models.make(cfg)

            # Ensure all leaves are jnp float 32s.
            # Bandaid for some poorly specified graph adjacency matrices
            serialized_model = jax.tree_util.tree_map(
                lambda x: jnp.array(x)
                if isinstance(x, np.ndarray) else x, serialized_model)
            model = eqx.tree_deserialise_leaves(f, serialized_model)

            optim = optax.adam(lr_schedule(epoch))
            serialized_opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            opt_state = eqx.tree_deserialise_leaves(f, serialized_opt_state)

        # --- Create and Populate New Trainer Instance ---
        print("Creating new Trainer instance...")
        # Call cls (Trainer) constructor with loaded/recreated components
        trainer = cls(cfg=cfg,
                      log_dir=checkpoint_dir.parent,
                      checkpoint={
                          'epoch': epoch,
                          'losses': losses,
                          'model': model,
                          'optim': optim,
                          'opt_state': opt_state,
                          'early_stopper': early_stopper
                      })

        return trainer

    @classmethod
    def load_last_checkpoint(cls, log_dir: Path):
        """Finds the directory of the last saved epoch or loads a fresh Trainer from config if no checkpoints exist.


        Parameters
        ----------
        log_dir : Path
            Directory containing the saved epoch directories.

        Returns
        -------
        Trainer | None
            The path to the last epoch directory, or None if no epoch directories are found.
        """
        epoch_regex = re.compile(r"epoch(\d+)")
        dirs = os.listdir(log_dir)
        matches = [epoch_regex.match(d) for d in dirs]
        epoch_strs = [m.group(1) for m in matches if isinstance(m, re.Match)]

        if epoch_strs:
            last_epoch_idx = np.argmax([int(s) for s in epoch_strs])
            checkpoint_dir = log_dir / f"epoch{epoch_strs[last_epoch_idx]}"
            return cls.load_checkpoint(checkpoint_dir)
        else:
            # --- Load Config and create fresh Trainer instance ---
            config_path = log_dir / "config.pkl"
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    cfg = pickle.load(f)
                print("No checkpoints found. Creating Trainer from config...")
                return cls(cfg=cfg, log_dir=log_dir)
            else:
                raise FileNotFoundError(
                    f"No checkpoints or config.pkl found in {log_dir}")


def _create_lr_schedule(cfg):
    """Helper to create LR schedule from config."""
    try:
        return optax.exponential_decay(cfg['initial_lr'],
                                       cfg['num_epochs'], cfg['decay_rate'],
                                       cfg.get('transition_begin', 0))
    except KeyError as e:
        raise ValueError(f"Missing required LR schedule config key: {e}")
