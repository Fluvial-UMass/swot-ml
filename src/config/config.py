from pathlib import Path
from typing import Literal, Any
from enum import Enum
import yaml
import functools
import warnings

import numpy as np
import json
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
)
from pydantic.types import DirectoryPath, FilePath
from datetime import datetime

# Import model argument classes from model_args.py
from .model_args import ModelArgs


class Features(BaseModel):
    dynamic: dict[str, list[str]]
    static: list[str] | None = None
    target: dict[str, list[str]]


class StepKwargs(BaseModel):
    loss_name: Literal["mse", "mae", "huber", "nse", "spin_up_nse"] = "mse"
    target_weights: list[float] | None = None
    max_grad_norm: float | None = None
    agreement_weight: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def check_deprecated_loss(cls, values):
        if isinstance(values, dict) and "loss" in values:
            warnings.warn("'loss' is deprecated. Use 'loss_name' instead.", DeprecationWarning)
            # Only override loss_name if it wasn't explicitly passed
            values.setdefault("loss_name", values["loss"])
        return values


class EarlyStopKwargs(BaseModel):
    patience: int = Field(0, gt=0)
    threshold: float = Field(..., gt=0, lt=1.0)


class ValueFilter(BaseModel):
    column: str
    operation: str
    value: Any
    feature_list: list[str] | None = None
    feature_group: str | None = None

    @model_validator(mode="before")
    @classmethod
    def feature_type(cls, values: dict):
        f_list = values.get("feature_list")
        f_group = values.get("feature_group")
        if f_list and f_group:
            raise ValueError("Specify only one of 'feature_list' or 'feature_group', not both.")
        if not f_list and not f_group:
            raise ValueError("Specify exactly one of 'feature_list' or 'feature_group'.")
        return values


class DataSubset(str, Enum):
    train = "train"  # Loss, gradients, and back propagation
    validate = "validate"  # Loss only
    test = "test"  # Error statistics only
    predict = "predict"  # Same as test but for sequences without test data


class Config(BaseModel):
    def to_json(self, path: str | Path, exclude_none: bool = True, exclude_unset: bool = True):
        """Dump the config to a JSON file."""
        cfg_out = self.model_copy(deep=True)
        json_str = cfg_out.model_dump_json(
            indent=4, exclude_none=exclude_none, exclude_unset=exclude_unset
        )
        with open(path, "w") as f:
            f.write(json_str)

    @classmethod
    def from_file(cls, file_path: Path | str) -> "Config":
        """
        Load the config from a YAML or JSON file. Automatically detects file type by extension.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        ext = file_path.suffix.lower()
        if ext in [".yml", ".yaml"]:
            with open(file_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        elif ext == ".json":
            with open(file_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

        cfg = cls(**cfg_dict)
        cfg.cfg_path = file_path
        return cfg

    def __str__(self) -> str:
        """
        Provides a nicely formatted JSON string of all model attributes,
        """
        return self.model_dump_json(indent=4)

    # Data paths
    data_root: DirectoryPath
    zarr_dir: DirectoryPath
    attributes_file: FilePath
    train_basin_file: FilePath
    test_basin_file: FilePath
    graph_network_file: FilePath

    # Data processing
    features: Features
    in_memory: bool = False
    train_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    validate_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    test_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    predict_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    add_rolling_means: list[int] | None = None
    clip_feature_range: dict[str, list[float | None]] = Field(default_factory=dict)
    value_filters: list[ValueFilter] = Field(default_factory=list)
    exclude_target_from_index: list[str] = Field(default_factory=list)

    # Feature scaling / encoding
    log_norm_cols: list[str] = Field(default_factory=list)
    range_norm_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    bitmask_cols: list[str] = Field(default_factory=list)

    # DataLoader
    data_subset: DataSubset = DataSubset.train
    shuffle: bool = True
    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)
    timeout: int = Field(..., ge=0)
    persistent_workers: bool
    pin_memory: bool = False
    drop_last: bool = True
    backend: Literal["cpu", "gpu", "tpu"] | None = None
    num_devices: int | None = Field(None, gt=0)

    # Model
    sequence_length: int
    model_args: ModelArgs = Field(discriminator="name")

    # Trainer
    num_epochs: int = Field(..., gt=0)
    validate_interval: int = Field(0, gt=0)  # 0 disables by falsiness
    initial_lr: float = Field(..., gt=0, lt=1.0)
    decay_rate: float = Field(..., gt=0, lt=1.0)
    transition_begin: int = Field(0, ge=0)
    step_kwargs: StepKwargs
    early_stop_kwargs: EarlyStopKwargs = None

    # Meta
    parameter_search_grid: dict = Field(default_factory=dict)

    # Outputs
    quiet: bool = True
    log: bool = True
    log_interval: int = Field(5, gt=0)

    # Internal (not from YML)
    cfg_path: Path = Path()

    # MODEL / BEFORE
    @model_validator(mode="before")
    @classmethod
    def validate_paths(cls, values: dict):
        def resolve_path(attr_str):
            str_path = values.get(attr_str)
            if str_path:
                return Path(str_path).resolve()
            else:
                return None

        abs_paths = [
            "data_root",
            "zarr_dir",
        ]
        for attr in abs_paths:
            values[attr] = resolve_path(attr)

        rel_paths = [
            "attributes_file",
            "train_basin_file",
            "test_basin_file",
            "graph_network_file",
        ]
        for attr in rel_paths:
            values[attr] = values["data_root"] / values[attr]

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_workers(cls, values: dict):
        num_workers = values.get("num_workers", 1)
        if num_workers == 0:
            values["timeout"] = 0
            values["persistent_workers"] = False
        else:
            values["timeout"] = values.get("timeout", 900)
            values["persistent_workers"] = values.get("persistent_workers", True)
        return values

    # MODEL / AFTER
    @model_validator(mode="after")
    def validate_nse_requires_seq2seq(self):
        """
        Ensure that if loss_name == 'nse', then model_args.seq2seq == True.
        Only checks models that actually have the seq2seq attribute.
        """
        if self.step_kwargs.loss_name in ["nse", "spin_up_nse"]:
            if hasattr(self.model_args, "seq2seq"):
                if not self.model_args.seq2seq:
                    raise ValueError(
                        "When using loss_name='nse', model_args.seq2seq must be set to True."
                    )
            else:
                raise ValueError(
                    "loss_name='nse' requires a model type with a 'seq2seq' attribute AND seq2seq=True."
                )
        return self

    # FIELD / BEFORE
    @field_validator(
        "train_date_range",
        "validate_date_range",
        "test_date_range",
        "predict_date_range",
        mode="before",
    )
    def parse_time_slice(cls, v):
        if (not isinstance(v, list)) or (len(v) != 2):
            raise ValueError("date ranges must be a list of two elements (start, end).")
        # Convert to datetime objects
        t_start = datetime.fromisoformat(v[0])
        t_end = datetime.fromisoformat(v[1])
        return [t_start, t_end]

    @field_validator("clip_feature_range", mode="before")
    def process_clip_feature_range(cls, v):
        if not v:
            return {}
        processed_ranges = {}
        for key, range_list in v.items():
            if (not isinstance(range_list, list)) or (len(range_list) != 2):
                raise ValueError(f"Each clip range for '{key}' must have exactly 2 elements.")
            lower = -np.inf if range_list[0] is None else range_list[0]
            upper = np.inf if range_list[1] is None else range_list[1]
            processed_ranges[key] = [lower, upper]
        return processed_ranges

    # FIELD / AFTER
    @field_validator("step_kwargs", mode="after")
    def validate_step_kwargs(cls, v: StepKwargs, info):
        cfg = info.data  # Access the entire config data
        if v.agreement_weight != 0:
            required_targets = {"ssc", "flux", "usgs_q"}
            if not required_targets.issubset(set(cfg["features"].target)):
                raise ValueError(
                    "Must predict at least ssc, flux, and usgs_q when using flux agreement regularization."
                )
        # Convert target_weights dict to list based on features.target order
        if v.target_weights is not None:
            if isinstance(v.target_weights, dict):
                v.target_weights = [
                    v.target_weights.get(target, 1) for target in cfg["features"].target
                ]
            elif isinstance(v.target_weights, list):
                # Just need to check that it is the right length.
                if len(v.target_weights) != len(cfg["features"].target):
                    raise ValueError(
                        f"Length of target_weights list ({len(v.target_weights)}) does not match number of targets ({len(cfg['features'].target)})."
                    )
            else:
                raise TypeError(
                    f"target_weights must be a dict or a list, got {type(v.target_weights)}."
                )
        else:
            # Default all to 1 if not specified
            v.target_weights = [1] * len(cfg["features"].target)
        return v

    # CONFIG MANIPULATION
    def rgetattr(self, attr: str, *args):
        """
        Recursively get attribute from the instance using dot notation.
        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [self] + attr.split("."))

    def rsetattr(self, attr: str, value: Any):
        """
        Recursively set attribute on the instance using dot notation.
        """
        pre, _, post = attr.rpartition(".")
        return setattr(self.rgetattr(pre) if pre else self, post, value)

    def update_from_grid(self, idx: int) -> "Config":
        param_dict = self.parameter_search_grid
        if param_dict == {}:
            raise AttributeError("param_search_dict is empty. Cannot update from grid.")

        def flatten_param_dict(d: dict, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from flatten_param_dict(v, full_key)
                else:
                    yield full_key, v

        dotted_keys, value_set = zip(*flatten_param_dict(param_dict))

        # Compute the total number of combinations
        sizes = [len(v) for v in value_set]
        n_total = np.prod(sizes)
        if idx < 0 or idx >= n_total:
            raise IndexError(f"Index {idx} is out of range for grid of size {n_total}")

        # Do not change the seed! This ensures deterministic grid shuffling between runs.
        rng = np.random.default_rng(42)
        # Shuffle the grid indices (permutation of size n_total) and select our index
        grid_idx = rng.permutation(n_total)[idx]

        # Convert grid_idx to N-dimensional indices for each parameter
        unravel = np.unravel_index(grid_idx, sizes)
        new_values = [v[i] for v, i in zip(value_set, unravel)]

        for attr, value in zip(dotted_keys, new_values):
            self.rsetattr(attr, value)
