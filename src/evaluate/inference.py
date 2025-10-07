from typing import Iterator
from pathlib import Path

import equinox as eqx
import jax
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from jaxtyping import PRNGKeyArray

from data import BasinGraphDataLoader, GraphBatch


@eqx.filter_jit
def _model_map(model, batch: GraphBatch, key: PRNGKeyArray):
    """Applies the model to a batch of data using jax.vmap."""
    y_pred = model(batch, key)
    return y_pred


def model_iterate(
    model: eqx.Module,
    dataloader: BasinGraphDataLoader,
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
    # Dummy key (only used for dropout, which is off).
    key = jax.random.PRNGKey(0)

    for basin, date, batch in tqdm(dataloader, disable=quiet):
        y_pred = _model_map(model, batch, key)[batch.node_mask]

        # TODO bandaid for seq2seq models with 4 dimensions.
        if len(y_pred.shape) == 4:
            # Grab the final prediction for now.
            y_pred = y_pred[-1, ...]

        if denormalize:
            y_pred = dataloader.denormalize_target(y_pred)

        out_dict = {"basin": basin, "date": date, "y_pred": y_pred}

        if batch.y is not None:
            y = batch.y[-1, ...][batch.node_mask]
            if denormalize:
                y = dataloader.denormalize_target(y)
            out_dict["y"] = y

        yield out_dict


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
    # inference_mode = dataloader.dataset.inference_mode
    basins = []
    dates = []
    y_hat_list = []
    y_list = []

    # Iterate through the dataset, make predictions and collect data in lists.
    for result_dict in model_iterate(model, dataloader, quiet, denormalize):
        basins.extend(result_dict["basin"])
        dates.extend(result_dict["date"])
        y_hat_list.append(result_dict["y_pred"])

        if "y" in result_dict.keys():
            y_list.append(result_dict.get("y"))

    # Concatenate all the data lists into arrays.
    y_hat_arr = np.concatenate(y_hat_list)

    if len(y_list) > 0:
        y_arr = np.concatenate(y_list)
        data = np.concatenate((y_arr, y_hat_arr), axis=-1)
        cols = ["obs", "pred"]
    else:
        data = y_hat_arr
        cols = ["pred"]

    subbasin_map = dataloader.dataset.basin_subbasin_map  # dict[str: list[str]]
    rows = [
        (basin, subbasin, date)
        for basin, date in zip(basins, dates)
        for subbasin in subbasin_map[basin]
    ]

    # Place the data arrays into a dataframe with multilevel indices.
    row_index = pd.MultiIndex.from_tuples(rows, names=["basin", "subbasin", "date"])
    column_index = pd.MultiIndex.from_product(
        [cols, dataloader.dataset.target],
        names=["Type", "Feature"],
    )
    results = pd.DataFrame(
        data,
        index=row_index,
        columns=column_index,
    )

    return results


def predict_to_parquet(
    model: eqx.Module,
    dataloader: BasinGraphDataLoader,
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
        for batch_results in model_iterate(model, dataloader, quiet=quiet, denormalize=denormalize):
            # --- 1. Combine prediction and observation data for the batch ---
            y_pred_batch = batch_results["y_pred"]

            if "y" in batch_results:
                y_obs_batch = batch_results["y"]
                # Combine observations and predictions side-by-side
                data_for_df = np.concatenate((y_obs_batch, y_pred_batch), axis=-1)
                col_types = ["obs", "pred"]
            else:
                data_for_df = y_pred_batch
                col_types = ["pred"]

            # --- 2. Construct the row index for the batch ---
            basins_in_batch = batch_results["basin"]
            dates_in_batch = batch_results["date"]
            subbasin_map = dataloader.dataset.basin_subbasin_map
            rows = [
                (basin, subbasin, date)
                for basin, date in zip(basins_in_batch, dates_in_batch)
                for subbasin in subbasin_map[basin]
            ]
            row_index = pd.MultiIndex.from_tuples(rows, names=["basin", "subbasin", "date"])

            # --- 3. Construct the column index ---
            column_index = pd.MultiIndex.from_product(
                [col_types, dataloader.dataset.target],
                names=["Type", "Feature"],
            )

            # --- 4. Create a DataFrame for the current batch ---
            batch_df = pd.DataFrame(
                data_for_df,
                index=row_index,
                columns=column_index,
            )

            # --- 5. Write the batch DataFrame to the Parquet file ---
            table = pa.Table.from_pandas(batch_df)

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
