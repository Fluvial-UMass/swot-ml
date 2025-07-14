import jax
from jaxtyping import PyTree
import equinox as eqx

from config import Config
from config.model_args import (
    SeqAttnArgs,
    StackArgs,
    GraphLSTMArgs,
    MSAttnArgs,
    RaindropArgs,
    STGTArgs,
    STGATArgs,
)
from data import HydroDataLoader, HydroDataset

from models.lstm_mlp_attn import LSTM_MLP_ATTN
from models.stacked_lstm import STACKED_LSTM
from models.ms_attention import MS_ATTN
from models.raindrop import RAINDROP
from models.st_graph_transformer import ST_Graph_Transformer
from models.graph_transformer import ST_GATransformer
# from models.rg_lstm import Graph_LSTM


# Dictionary of valid model names and their constructors
MODEL_MAP = {
    "lstm_mlp_attn": LSTM_MLP_ATTN,
    "stacked_lstm": STACKED_LSTM,
    "ms_attn": MS_ATTN,
    "raindrop": RAINDROP,
    "st_graph_transformer": ST_Graph_Transformer,
    "st_gat": ST_GATransformer,
    # "graph_lstm": Graph_LSTM,
}


def make(cfg: Config, dl: HydroDataLoader = None):
    """Creates a model based on the provided configuration.

    Parameters
    ----------
    cfg: dict
        A dictionary containing the configuration for the model.

    Returns
    -------
    Config
        The config object used to make the model.
    eqx.Module
        The created model.
    """
    if dl is not None:
        cfg = set_model_data_args(cfg, dl.dataset)

    model_fn = MODEL_MAP[cfg.model_args.name]
    model = model_fn(**cfg.model_args.as_kwargs())
    num_params, memory_bytes = count_parameters(model)
    size, unit = human_readable_size(memory_bytes)
    print(f"Model contains {num_params:,} parameters, using {size:.2f}{unit} memory.")
    return cfg, model


def set_model_data_args(cfg: Config, dataset: HydroDataset):
    """Set model arguments based on configuration and dataset."""
    dyn_feat = dict(dataset.features["dynamic"])

    if isinstance(cfg.model_args, SeqAttnArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features["static"])
        cfg.model_args.time_aware = dataset.time_gaps

    elif isinstance(cfg.model_args, STGATArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features["static"])

    elif isinstance(cfg.model_args, STGTArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features["static"])

    elif isinstance(cfg.model_args, RaindropArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.num_locations = dataset.graph_matrix.shape[0]
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features["static"])

    elif isinstance(cfg.model_args, StackArgs):
        cfg.model_args.dynamic_size = len(list(dyn_feat.values())[0])
        cfg.model_args.static_size = len(dataset.features["static"])

    elif isinstance(cfg.model_args, MSAttnArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features["static"])

    elif isinstance(cfg.model_args, GraphLSTMArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.dynamic_size = len(list(dyn_feat.values())[0])
        cfg.model_args.static_size = len(dataset.features["static"])
        cfg.model_args.graph_matrix = dataset.graph_matrix
    else:
        raise ValueError(f"Unknown model_args type: {type(cfg.model_args)}")
    return cfg


def count_parameters(model: PyTree):
    """Counts the trainable parameters in a model and estimates its memory usage.

    Parameters
    ----------
    model: PyTree
        Model to count parameters.

    Returns
    -------
    num_params: int
        Total number of trainable parameters in the model.
    memory_bytes: int
        Estimated memory usage of the model in bytes, assuming each parameter is a
        32-bit float.
    """
    # Use tree_flatten to get a list of arrays and ensure is_leaf treats arrays as leaves
    params, _ = jax.tree_util.tree_flatten(model)
    # Count the total number of parameters
    num_params = sum(param.size for param in params if eqx.is_inexact_array(param))
    # Calculate memory usage assuming 4 bytes per parameter (32-bit float)
    memory_bytes = num_params * 4
    return num_params, memory_bytes


# Convert bytes to a human-readable format
def human_readable_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            break
        size /= 1024.0
    return size, unit
