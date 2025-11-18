from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import equinox as eqx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jax.random import PRNGKey
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from data import CachedBasinGraphDataLoader, GraphBatch
from models import BaseModel
from models.layers.heads import GMM


@eqx.filter_jit
def _model_map(model, batch: GraphBatch, key: PRNGKeyArray) -> dict[str, Array | dict[str, Array]]:
    """Applies the model to a batch of data using jax.vmap."""
    y_pred = model(batch, key)
    return y_pred


def _direct_values(
    pred: Array, node_mask: Array, denorm_fn: Callable | None
) -> tuple[Array, Array]:
    # TODO Do we want to save anything more than the final prediction?
    if len(pred.shape) == 3:
        pred = pred[-1, ...]  # Grab the final prediction for now.
    pred = pred[..., 0][node_mask]  # Drop padded nodes
    pred = denorm_fn(pred) if denorm_fn else pred

    return np.asarray(pred)  # np to detach from jax


def _gmm_expected_values(
    dist_pred: dict[str, Array],
    node_mask: Array,
    denorm_fn: Callable | None,
    denorm_std_fn: Callable | None,
) -> tuple[Array, Array]:
    mu = np.asarray(dist_pred["mu"][node_mask])
    sigma = np.asarray(dist_pred["sigma"][node_mask])
    pi = np.asarray(dist_pred["pi"][node_mask])

    mu_exp = np.sum(pi * mu, axis=-1)

    # Expected variance
    var_exp = np.sum(pi * (sigma**2 + (mu - mu_exp[..., None]) ** 2), axis=-1)
    std_exp = np.sqrt(var_exp)

    mu_exp_denorm = denorm_fn(mu_exp) if denorm_fn else mu_exp
    std_exp_denorm = denorm_std_fn(mu_exp, std_exp) if denorm_fn else std_exp

    return mu_exp, std_exp


def model_iterate(
    model: BaseModel,
    dataloader: CachedBasinGraphDataLoader,
    quiet: bool = False,
    denormalize: bool = True,
) -> Iterator[dict]:
    """Iterates through a dataloader and yields predictions from the model.

    Parameters
    ----------
    model: eqx.Module
        The model to be applied to the data.
    dataloader: Dataloader
        The dataloader providing batches of data.
    quiet: bool, optional
        If True, disables the progress bar. Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.

    Yields
    -------
    Iterator[dict]
        An iterator yielding dictionaries. Each dictionary contains the following keys:
            - 'basin': Basin IDs.
            - 'date': Dates.
            - 'y_pred': Model predictions.
            - 'y': Target values (if available).
            - 'dt': Dynamic time features (if available).
    """
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    # Dummy key (only used for dropout, which we just turned off).
    key = PRNGKey(0)

    denorm_fns = {
        t: partial(dataloader.denormalize, name=t) if denormalize else None for t in model.target
    }
    denorm_std_fns = {
        t: partial(dataloader.denormalize_std, name=t) if denormalize else None
        for t in model.target
    }

    for basin, date, batch in tqdm(dataloader, disable=quiet):
        batch = batch.to_jax()
        y_pred = _model_map(model, batch, key)

        out_dict = {"basin": basin, "date": date, "y_pred": {}}
        for target_name, target_pred in y_pred.items():
            denorm_fn = denorm_fns[target_name]
            denorm_std_fn = denorm_std_fns[target_name]
            # Check the type of this target's head. Changes how we create predictions.
            if isinstance(model.head[target_name], GMM):
                t_exp, t_std = _gmm_expected_values(
                    target_pred, batch.node_mask, denorm_fn, denorm_std_fn
                )
                out_dict["y_pred"][target_name] = t_exp
                out_dict["y_pred"][target_name + "_std"] = t_std
            else:
                t_exp = _direct_values(target_pred, batch.node_mask, denorm_fn)
                out_dict["y_pred"][target_name] = t_exp

            if batch.y is not None:
                if "y" not in out_dict.keys():
                    out_dict["y"] = {}

                # TODO Do we want to save anything more than the final prediction?
                y = batch.y[target_name][-1, ..., 0][batch.node_mask]
                y = denorm_fn(y) if denorm_fn else y
                out_dict["y"][target_name] = np.asarray(y)

        yield out_dict


def model_df_iterate(
    model: eqx.Module,
    dataloader: CachedBasinGraphDataLoader,
    *,
    quiet: bool = False,
    denormalize: bool = True,
):
    subbasin_map = dataloader.dataset.basin_subbasin_map

    # Iterate through batches from the dataloader
    for batch_results in model_iterate(model, dataloader, quiet=quiet, denormalize=denormalize):
        y_pred_dict = batch_results["y_pred"]
        y_obs_dict = batch_results.get("y", {})

        df_parts = {}
        df_parts["pred"] = pd.DataFrame({k: v for k, v in y_pred_dict.items()})
        if y_obs_dict:
            df_parts["obs"] = pd.DataFrame({k: v for k, v in y_obs_dict.items()})

        df = pd.concat(df_parts, axis=1)

        rows = [
            (basin, subbasin, date)
            for basin, date in zip(batch_results["basin"], batch_results["date"])
            for subbasin in subbasin_map[basin]
        ]
        row_index = pd.MultiIndex.from_tuples(rows, names=["basin", "subbasin", "date"])
        df.index = row_index

        yield df


def predict(model: eqx.Module, dataloader, *, quiet: bool = False, denormalize: bool = True):
    """Generates predictions from a model and dataloader.

    The function iterates through the dataloader, applies the model to each batch, and
    collects the predictions, target values, and dynamic time features (if available)
    into lists. These lists are then concatenated into arrays and organized into a
    pandas DataFrame with a MultiIndex.

    Parameters
    ----------
    model: eqx.Module
        The model to use for generating predictions.
    dataloader: Dataloader
        The dataloader providing data for the model.
    quiet: bool, optional
        If True, suppresses the progress bar. Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the predictions, observations (if available), and dynamic
        time features (if available). The DataFrame has a MultiIndex with levels 'basin'
        and 'date'. The columns are also MultiIndexed with levels 'Type' and 'Feature'.
    """

    batch_dfs = []
    # Iterate through the dataset, make predictions and collect data in lists.
    for batch_df in model_df_iterate(model, dataloader, quiet, denormalize):
        batch_dfs.append(batch_df)

    df = pd.concat(batch_dfs)

    return df


def predict_to_parquet(
    model: eqx.Module,
    dataloader: CachedBasinGraphDataLoader,
    output_path: Path | str,
    *,
    quiet: bool = False,
    denormalize: bool = True,
):
    """
    Generates predictions and streams them directly to a Parquet file.

    This function processes the dataset in batches, creates a pandas DataFrame for
    each batch, and appends it to a Parquet file on disk. This approach is
    memory-efficient as it avoids loading all predictions into memory at once.

    Parameters
    ----------
    model: eqx.Module
        The model to use for generating predictions.
    dataloader: DataLoader
        The dataloader providing data for the model.
    output_path: Union[str, Path]
        The path to the output Parquet file.
    quiet: bool, optional
        If True, suppresses the progress bar from the underlying iterator.
        Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.
    """
    writer = None  # Initialize the Parquet writer

    try:
        # Iterate through batches from the dataloader
        for batch_df in model_df_iterate(model, dataloader, quiet=quiet, denormalize=denormalize):
            #  Clean NaNs (unobserved subbasins)
            df = batch_df.dropna(how="any")

            # Write the batch DataFrame to the Parquet file
            table = pa.Table.from_pandas(df)

            if writer is None:
                # For the first batch, create the writer with the table's schema
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

    finally:
        # Ensure the writer is closed to finalize the file
        if writer:
            writer.close()
            if not quiet:
                print(f"Results successfully written to {output_path}")
