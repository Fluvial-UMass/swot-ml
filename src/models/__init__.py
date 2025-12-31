import equinox as eqx
import jax

from config import Config
from config.model_args import (
    MSAttnArgs,
    SeqAttnArgs,
    StackArgs,
    STGATArgs,
    SFGRNNArgs,
    MCDALSTMArgs,
)
from data import CachedBasinGraphDataLoader, CachedBasinGraphDataset
from models.base_model import BaseModel
from models.graph_transformer import ST_GATransformer
from models.lstm_mlp_attn import LSTM_MLP_ATTN
from models.ms_attention import MS_ATTN
from models.stacked_lstm import STACKED_LSTM
from models.sparse_fusion_gann import SparseFusionGRNN
from models.mc_da_lstm import MCDALSTM

# Dictionary of valid model names and their constructors
MODEL_MAP = {
    "lstm_mlp_attn": LSTM_MLP_ATTN,
    "stacked_lstm": STACKED_LSTM,
    "ms_attn": MS_ATTN,
    "st_gat": ST_GATransformer,
    "sf_grnn": SparseFusionGRNN,
    "mc_da_lstm": MCDALSTM,
}


def make(cfg: Config, dl: CachedBasinGraphDataLoader = None):
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
    num_params = count_parameters(model)
    print(f"Model contains {num_params:,} parameters")
    return cfg, model


def set_model_data_args(cfg: Config, dataset: CachedBasinGraphDataset):
    """Set model arguments based on configuration and dataset."""
    dyn_feat = dict(dataset.features.dynamic)

    if isinstance(cfg.model_args, SeqAttnArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features.static)
        cfg.model_args.time_aware = cfg.time_gaps

    elif isinstance(cfg.model_args, (STGATArgs, SFGRNNArgs, MCDALSTMArgs)):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dense_sizes = {
            k: len(dataset.features.dynamic[k])
            for k, has_gaps in cfg.time_gaps.items()
            if not has_gaps
        }
        cfg.model_args.sparse_sizes = {
            k: len(dataset.features.dynamic[k]) for k, has_gaps in cfg.time_gaps.items() if has_gaps
        }
        if dataset.features.static is not None:
            cfg.model_args.static_size = len(dataset.features.static)
        else:
            cfg.model_args.static_size = 0

    elif isinstance(cfg.model_args, StackArgs):
        cfg.model_args.dynamic_size = len(list(dyn_feat.values())[0])
        cfg.model_args.static_size = len(dataset.features.static)

    elif isinstance(cfg.model_args, MSAttnArgs):
        cfg.model_args.target = dataset.target
        cfg.model_args.seq_length = cfg.sequence_length
        cfg.model_args.dynamic_sizes = {k: len(v) for k, v in dyn_feat.items()}
        cfg.model_args.static_size = len(dataset.features.static)

    else:
        raise ValueError(f"Unknown model_args type: {type(cfg.model_args)}")

    return cfg


def count_parameters(model: BaseModel):
    """Counts the trainable parameters in a model and estimates its memory usage.

    Parameters
    ----------
    model: PyTree
        Model to count parameters.

    Returns
    -------
    num_params: int
        Total number of trainable parameters in the model.
    """
    # Use tree_flatten to get a list of arrays and ensure is_leaf treats arrays as leaves
    params, _ = jax.tree_util.tree_flatten(model)
    # Count the total number of parameters
    num_params = sum(param.size for param in params if eqx.is_inexact_array(param))
    # Calculate memory usage assuming 4 bytes per parameter (32-bit float)
    return num_params
