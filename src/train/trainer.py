import logging
import pickle
import json
import os
import sys
import re
import traceback
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree
from tqdm import tqdm

import models
from config import Config
from data import CachedBasinGraphDataLoader
from .step import make_step, compute_loss
from .early_stop import EarlyStopper


class Trainer:
    """Trainer class for training hydrological models.

    Methods
    -------
    __init__(cfg, dataloader, log_dir=None)
        Initializes the trainer.
    setup_logging(log_dir=None)
        Sets up logging for training.
    start_training(stop_at=np.inf)
        Starts or continues training the model.
    _train_step()
        Trains the model for one step.
    save_state(save_dir=None)
        Saves the model and trainer state.
    freeze_components(component_names=None, freeze=True)
        Freezes or unfreezes specified components of the model.
    load_checkpoint(step_dir)
        Loads the model and trainer state.
    load_last_checkpoint(log_dir)
        Loads the last saved model and trainer state.
    """

    cfg: Config
    logger: logging.Logger
    training_dl: CachedBasinGraphDataLoader
    validation_dl: CachedBasinGraphDataLoader
    log_dir: Path
    lr_schedule: optax.Schedule
    model: eqx.Module
    losses: list
    step: int
    optim: optax.GradientTransformation
    opt_state: optax.OptState
    early_stopper: EarlyStopper | None
    filter_spec: PyTree | None
    train_key: jax.random.PRNGKey
    last_long_time: float

    def __init__(
        self,
        cfg: Config,
        training_dl: CachedBasinGraphDataLoader = None,
        validation_dl: CachedBasinGraphDataLoader = None,
        *,
        log_dir: Path | None = None,
        checkpoint: dict | None = None,
    ):
        """Initializes the Trainer.

        Sets up logging, the learning rate schedule, the model, the optimizer, and the optimizer state.  Handles loading from a previous state if specified.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        training_dl : data.HydroDataLoader
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
        self.training_dl = training_dl
        self.validation_dl = validation_dl

        self.lr_schedule = _create_lr_schedule(cfg)
        self.log_dir = self._setup_logging(log_dir)
        self.train_key = jax.random.PRNGKey(cfg.model_args.seed)

        if checkpoint:
            # This is only really used from the class method load_checkpoint()
            self.step = checkpoint["step"]
            self.losses = checkpoint["losses"]
            self.model = checkpoint["model"]
            self.optim = checkpoint["optim"]
            self.opt_state = checkpoint["opt_state"]
            self.early_stopper = checkpoint["early_stopper"]
        else:
            self.step = 0
            self.losses = []
            self.cfg, self.model = models.make(cfg, training_dl)
            self.optim = optax.adam(self.lr_schedule)
            self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))
            if cfg.early_stop_kwargs is not None:
                self.early_stopper = EarlyStopper(**cfg.early_stop_kwargs.model_dump())
            else:
                self.early_stopper = None

        # Initialize the filterspec. Defaults to training all components of the model.
        self.freeze_components([])
        self.checkpoint_losses = []

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
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)  # Output to standard out
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if self.cfg.log:
            if log_dir is None:
                cfg_path = self.cfg.cfg_path
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = cfg_path.parent / f"{cfg_path.stem}_{current_date}"
            log_dir.mkdir(parents=True, exist_ok=True)

            print(f"Logging at {log_dir}")
            log_file = log_dir / "training.log"
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.cfg.to_json(log_dir / "config.json")  # dump config to log dir
        return log_dir

    def _cleanup_logger(self):
        # Check if the logger attribute exists and is actually a logger instance
        if hasattr(self, "logger") and isinstance(self.logger, logging.Logger):
            # Iterate over a *copy* of the handlers list ([ : ]) because we are modifying the list during iteration.
            for handler in self.logger.handlers[:]:
                try:
                    # Flush and close the handler to release resources (e.g., file handles)
                    handler.flush()
                    # Check if handler has close method before calling
                    if hasattr(handler, "close"):
                        handler.close()
                except Exception as e:
                    # Log error to stderr, as the logger itself might be problematic
                    print(f"Warning: Error closing handler {handler}: {e}", file=sys.stderr)
                finally:
                    # Ensure the handler is removed even if closing failed
                    self.logger.removeHandler(handler)
        self.logger = None

    def start_training(self, stop_at=np.inf):
        """Starts or continues training the model.

        Manages the training loop, including updating progress bars, logging, handling keyboard interruptions, and
        saving the model state at specified intervals.

        Parameters
        ----------
        stop_at : float, optional
            Step count at which to stop training.

        Returns
        -------
        model : eqx.Module
            The trained model.
        """
        if self.training_dl is None:
            raise ValueError("Trainer was created without training dataloader.")

        max_steps = min(self.cfg.max_training_steps, stop_at)
        max_hours = self.cfg.max_training_hours

        self.logger.info(f"~~~ Starting training ({max_steps=}, {max_hours=}) ~~~")
        start_time = time.time()
        pbar = tqdm(
            self.training_dl,
            total=max_steps if np.isfinite(max_steps) else None,
            initial=self.step,
            disable=self.cfg.quiet,
            desc="Training Steps",
        )

        # Logging intervals and states
        log_steps = self.cfg.log_every_n_steps
        log_minutes = self.cfg.log_every_n_minutes
        last_log_step = self.step
        last_log_time = time.time()

        for _, _, batch in pbar:
            batch = batch.to_jax()
            # Perform one training step
            loss, grads = self._train_step(batch)
            self.checkpoint_losses.append(float(loss))
            self.step += 1
            pbar.update(1)

            # Validation
            if self.step % self.cfg.validate_every_n_steps == 0:
                v_loss = self.get_validation_loss()
                if v_loss and self.early_stopper and self.early_stopper(v_loss):
                    self.logger.info("Training stopped by EarlyStopper.")
                    break

            # Check stopping conditions
            if self.step >= self.cfg.max_training_steps:
                self.logger.info("Max training steps reached.")
                break
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= max_hours:
                self.logger.info("Max training hours reached.")
                break

            # log if EITHER step or time condition is met, then reset both trackers=
            time_since_log_min = (time.time() - last_log_time) / 60.0
            log_by_time = time_since_log_min >= log_minutes
            log_by_step = (self.step - last_log_step) >= log_steps

            if log_by_time or log_by_step:
                self.check_grads(grads)
                self.save_state()
                last_log_step = self.step
                last_log_time = time.time()

        self.save_state()  # Final save
        self.logger.info("~~~ Training done ~~~")
        self._cleanup_logger()

    def _train_step(self, batch) -> tuple[float | None, dict]:
        """Performs a single training step."""
        keys = jax.random.split(self.train_key)
        self.train_key = keys[0]
        step_key = keys[1]

        try:
            loss, grads, self.model, self.opt_state = make_step(
                self.model,
                batch,
                step_key,
                self.opt_state,
                self.optim,
                self.filter_spec,
                **self.cfg.step_kwargs.model_dump(),
            )
            if jnp.isnan(loss):
                raise RuntimeError("NaN loss encountered")

            return loss, grads

        except Exception as e:
            if self.cfg.log:
                error_dir = self.log_dir / "exceptions" / f"step_{self.step}"
                self.save_state(error_dir)
                with open(error_dir / "data.pkl", "wb") as f:
                    pickle.dump(batch, f)
                with open(error_dir / "exception.txt", "w") as f:
                    f.write(f"{str(e)}\n{traceback.format_exc()}")
                error_str = f"{type(e).__name__} at step {self.step}. See {error_dir} for details."
            else:
                error_str = f"{str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_str)
            raise e

    # Monitor gradients
    def check_grads(self, grads):
        grad_norms = jtu.tree_map(jnp.linalg.norm, grads)
        grad_norms = jtu.tree_leaves_with_path(grad_norms)

        bad_grads = {"vanishing": {}, "exploding": {}}
        for keypath, norm in grad_norms:
            tree_key = jtu.keystr(keypath)
            type_key = "vanishing" if norm < 1e-6 else "exploding" if norm > 1e3 else None
            if type_key is not None:
                if tree_key not in bad_grads[type_key]:
                    bad_grads[type_key][tree_key] = 1
                else:
                    bad_grads[type_key][tree_key] += 1

        # Log the counts of any bad gradients.
        for type_key, tree_counts in bad_grads.items():
            if tree_counts:
                warning_str = f"{type_key} gradients detected:"
                for tree_key, count in tree_counts.items():
                    warning_str += f"\n\t{tree_key}: {count}"
                self.logger.info(warning_str)

    def get_validation_loss(self) -> float:
        # Set model and dataloader for inference
        self.model = eqx.nn.inference_mode(self.model, True)
        batch_keys = jax.random.split(self.train_key, self.cfg.batch_size)
        losses = []
        pbar = tqdm(
            self.validation_dl, disable=self.cfg.quiet, desc=f"Validating Step:{self.step:06d}"
        )
        for _, _, batch in pbar:
            loss = compute_loss(
                self.model,
                batch,
                batch_keys,
                **self.cfg.step_kwargs.model_dump(),
            )
            losses.append(loss)

        v_loss = np.mean(losses)
        self.logger.info(f"Step: {self.step:06d}, Validation Loss: {v_loss:.4f}")

        # Reset model inference mode (dropout) after validation
        self.model = eqx.nn.inference_mode(self.model, False)

        return v_loss

    def freeze_components(self, component_names: list[str] | str = []):
        """Freezes or unfreezes specified components of the model.

        Updates the filter specification to control which parameters are updated during training. Only accepts top-level element names in the pytree model.

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

        Saves the model, optimizer state, step number, and loss list to the specified directory.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to save the state. If None, saves to a directory within the log
            directory named for the current step.
        """
        if not self.cfg.log:
            return
        if save_dir is None:
            save_dir = self.log_dir / f"step_{self.step:06d}"
        os.makedirs(save_dir, exist_ok=True)

        self.logger.info(
            f"Saving model checkpoint at step {self.step:06d}. "
            f"Average loss since last checkpoint: {np.mean(self.checkpoint_losses):0.3f}"
        )
        # Shuffle checkpoint losses over to main list and then reset.
        self.losses.extend(self.checkpoint_losses)
        self.checkpoint_losses = []

        with open(save_dir / "model_and_opt.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, self.model)
            eqx.tree_serialise_leaves(f, self.opt_state)

        with open(save_dir / "trainer_state.json", "w") as f:
            state = {"step": self.step, "losses": self.losses}
            if self.early_stopper:
                state["early_stopper"] = self.early_stopper.get_state()
            json.dump(state, f, default=float)

        self.cfg.to_json(save_dir / "config.json")

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: Path):
        """Loads the trainer state from a checkpoint directory and returns a new Trainer instance."""

        # --- Load Config ---
        cfg = Config.from_file(checkpoint_dir / "config.json")
        lr_schedule = _create_lr_schedule(cfg)

        # --- Load Trainer State (JSON) ---
        with open(checkpoint_dir / "trainer_state.json", "r") as f:
            trainer_state_data = json.load(f)

        step = trainer_state_data["step"]
        losses = trainer_state_data["losses"]

        stopper_state = trainer_state_data.get("early_stopper", None)
        if stopper_state:
            early_stopper = EarlyStopper.from_state(stopper_state)
        else:
            early_stopper = None

        # --- Load Model and Optimizer State ---
        with open(checkpoint_dir / "model_and_opt.eqx", "rb") as f:
            _, serialized_model = models.make(cfg)

            # Ensure all leaves are jnp float 32s.
            # Bandaid for some poorly specified graph adjacency matrices
            serialized_model = jax.tree_util.tree_map(
                lambda x: (jnp.array(x) if isinstance(x, np.ndarray) else x),
                serialized_model,
            )
            model = eqx.tree_deserialise_leaves(f, serialized_model)

            optim = optax.adam(lr_schedule(step))
            serialized_opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            opt_state = eqx.tree_deserialise_leaves(f, serialized_opt_state)

        # --- Create and Populate New Trainer Instance ---
        print(f"Loading trainer from checkpoint {checkpoint_dir.stem}")
        # Call class constructor with loaded/recreated components
        trainer = cls(
            cfg=cfg,
            log_dir=checkpoint_dir.parent,
            checkpoint={
                "step": step,
                "losses": losses,
                "model": model,
                "optim": optim,
                "opt_state": opt_state,
                "early_stopper": early_stopper,
            },
        )

        return trainer

    @classmethod
    def load_last_checkpoint(cls, log_dir: Path):
        # --- CHANGE: Regex now looks for 'step_...' directories ---
        step_regex = re.compile(r"step_(\d+)")
        dirs = [d for d in os.listdir(log_dir) if os.path.isdir(log_dir / d)]
        matches = [step_regex.match(d) for d in dirs]
        step_strs = [m.group(1) for m in matches if isinstance(m, re.Match)]
        if step_strs:
            last_step_idx = np.argmax([int(s) for s in step_strs])
            checkpoint_dir = log_dir / f"step_{step_strs[last_step_idx]}"
            return cls.load_checkpoint(checkpoint_dir)
        else:
            config_path = log_dir / "config.json"
            if config_path.exists():
                print("No checkpoints found. Creating fresh Trainer from config...")
                cfg = Config.from_file(config_path)
                return cls(cfg=cfg, log_dir=log_dir)
            else:
                raise FileNotFoundError(f"No checkpoints or config.json found in {log_dir}")


def _create_lr_schedule(cfg: Config):
    """Helper to create LR schedule from config based on total training steps."""
    return optax.exponential_decay(
        cfg.initial_lr,
        cfg.max_training_steps,  # Total steps for the decay
        cfg.decay_rate,
        cfg.transition_begin,
    )
