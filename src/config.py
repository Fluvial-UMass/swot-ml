from pathlib import Path
from typing import Literal, Any
from enum import Enum
import yaml
import numpy as np
import itertools
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    field_validator,
)
from pydantic.types import DirectoryPath, FilePath
from datetime import datetime


class Features(BaseModel):
    dynamic: dict[str, list[str]]
    static: list[str] | None = None
    target: list[str]


class StepKwargs(BaseModel):
    loss: Literal["mse", "mae", "huber"] = "mse"
    target_weights: list[float] | None = None
    max_grad_norm: float | None = None
    agreement_weight: float = 0.0

    
class EarlyStopKwargs(BaseModel):
    patience: int = Field(0, gt=0)
    threshold: float = Field(..., gt=0, lt=1.0)


class SeqAttnArgs(BaseModel):
    name: Literal["lstm_mlp_attn", "flexible_hybrid"]
    hidden_size: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, lt=1.0)
    seed: int = 0
    target: list[str] = None
    seq_length: int = None
    dynamic_sizes: dict[str, int] = None
    static_size: int = None
    time_aware: dict[str, bool] = None

    def as_kwargs(self) -> dict:
        return self.model_dump(exclude={"name"})


class GraphLSTMArgs(BaseModel):
    name: Literal["graph_lstm"]
    hidden_size: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, lt=1.0)
    seed: int = 0
    target: list[str] = None
    dynamic_size: int = None
    static_size: int = None
    graph_matrix: Any = None

    def as_kwargs(self) -> dict:
        return self.model_dump(exclude={"name"})


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
    pre_train = "pre_train"
    train = "train"
    test = "test"
    predict = "predict"
    predict_all = "predict_all"


class Config(BaseModel):
    # Data paths
    data_dir: DirectoryPath
    time_series_dir: DirectoryPath
    attributes_file: FilePath
    train_basin_file: FilePath
    test_basin_file: FilePath
    graph_network_file: FilePath | None

    # Data processing
    features: Features
    time_slice: list[datetime] = Field(..., min_length=2, max_length=2)
    split_time: datetime | None = None
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
    drop_last: bool = False
    backend: Literal["cpu", "gpu", "tpu"] | None = None
    num_devices: int | None = Field(None, gt=0)

    # Model
    sequence_length: int
    model_args: SeqAttnArgs | GraphLSTMArgs = Field(discriminator="name")

    # Trainer
    num_epochs: int = Field(..., gt=0)
    validate_interval: int = Field(0, gt=0)  # 0 disables by falsiness
    initial_lr: float = Field(..., gt=0, lt=1.0)
    decay_rate: float = Field(..., gt=0, lt=1.0)
    transition_begin: int = Field(0, ge=0)
    step_kwargs: StepKwargs
    early_stopping: EarlyStopKwargs

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
        # Get the base data directory
        base_dir = Path(values.get("data_dir")).resolve()
        values["data_dir"] = base_dir

        values["time_series_dir"] = base_dir / values.get("time_series_dir", "time_series")
        values["attributes_file"] = (
            base_dir / "attributes" / values.get("attributes_file", "attributes.csv")
        )
        values["train_basin_file"] = base_dir / values.get("train_basin_file")
        values["test_basin_file"] = base_dir / values.get("test_basin_file")

        graph_file = values.get("graph_network_file")
        values["graph_network_file"] = base_dir / graph_file if graph_file else None

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
            values["persistent_workers"] = values.get("persistent_workers", False)
        return values

    # FIELD / BEFORE
    @field_validator("time_slice", mode="before")
    def parse_time_slice(cls, v):
        if (not isinstance(v, list)) or (len(v) != 2):
            raise ValueError("time_slice must be a list of two elements (start, end).")
        # Convert to datetime objects
        t_start = datetime.fromisoformat(v[0])
        t_end = datetime.fromisoformat(v[1])
        return [t_start, t_end]

    @field_validator("split_time", mode="before")
    def parse_split_time(cls, v):
        if v is not None:
            # Convert to datetime object
            return datetime.fromisoformat(v)
        return v

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


def read_yml(yml_path: str | Path) -> dict[str, Any]:
    if not isinstance(yml_path, Path):
        yml_path = Path(yml_path)
    with open(yml_path, "r") as f:
        yml = yaml.safe_load(f)
    return yml


def read_config(yml_path: str | Path) -> Config:
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    raw_cfg = read_yml(yml_path)
    try:
        cfg = Config(**raw_cfg)
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        raise
    cfg.cfg_path = yml_path
    return cfg


def get_grid_update_tuples(
    param_dict: dict[str : list | dict[str:list]],
) -> tuple[list[str | tuple, list[tuple]]]:
    """Generates shuffled lists of keys and hyperparameter combinations for grid search.

    Parameters
    ----------
    param_dict : dict[str:list | dict[str:list]]
        A dictionary containing keys and values that outline the entire hyperparameter grid space.

    Returns
    -------
    list[str | tuple[str, str]]
        The order of hyperparameters in the update tuples
    list[tuple]
        A shuffled list of all possible hyperparameter combinations.

    Raises
    ------
    ValueError
        If the 'param_search_dict' contains anything other than lists and dictionaries of lists.
    RuntimeError
        If tuple keys have length other than 2.
    """
    key_list = []
    value_list = []
    for k1, v1 in param_dict.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                key_list.append((k1, k2))
                value_list.append(v2)
        elif isinstance(v1, list):
            key_list.append(k1)
            value_list.append(v1)
        else:
            raise ValueError("param_dict must contain only lists and dicts of lists")
    rng = np.random.default_rng(42)  # Do not change!
    param_grid_list = list(itertools.product(*value_list))
    rng.shuffle(param_grid_list)
    return key_list, param_grid_list


def update_cfg_from_grid(cfg: Config, idx: int) -> Config:
    """Updates the configuration dictionary with a hyperparameter combination from the grid.

    Parameters
    ----------
    cfg : Config
        The configuration object.
    idx : int
        The index of the hyperparameter combination to use.

    Returns
    -------
    Config
        The updated configuration object.

    Raises
    ------
    IndexError
        If `idx` is out of range for the `param_grid_list`.
    RuntimeError
        If tuple keys have length other than 2.
    """
    # Check if 'param_search_dict' exists in the config before calling get_grid_update_tuples
    if not hasattr(cfg, "param_search_dict") or (not cfg.param_search_dict):
        raise AttributeError(
            "param_search_dict not found in the configuration. Cannot update from grid."
        )
    key_list, param_grid_list = get_grid_update_tuples(cfg.param_search_dict)
    updates = param_grid_list[idx]
    # Create a mutable dictionary from the config object
    cfg_dict = cfg.model_dump()
    # Insert the updates into the config dictionary
    for k, v in zip(key_list, updates):
        if isinstance(k, tuple):
            if len(k) != 2:
                raise RuntimeError("tuple keys in 'param_search_dict' must have length 2")
            cfg_dict[k[0]][k[1]] = v
        else:
            cfg_dict[k] = v
    # Re-validate and create a new Config object
    return Config(**cfg_dict)
