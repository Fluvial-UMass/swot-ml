import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Iterator
from jaxtyping import Array, PRNGKeyArray


@eqx.filter_jit
def _model_map(model, batch: dict[str:Array], keys: list[PRNGKeyArray]):
    """Applies the model to a batch of data using jax.vmap."""
    # TODO: If we start training with mixes of different basins we will need to fix this.
    static_keys = ["graph"]
    in_axes_data = {k: (None if k in static_keys and k in batch else 0) for k in batch}
    in_axes_keys = 0
    y_pred = jax.vmap(model, in_axes=(in_axes_data, in_axes_keys))(batch, keys)
    return y_pred

    # return jax.vmap(model)(batch, keys)


def model_iterate(
    model: eqx.Module,
    dataloader,
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
    # Dummy batch keys (only used for dropout, which is off).
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, dataloader.batch_size)

    inference_mode = dataloader.dataset.inference_mode
    for basin, date, batch in tqdm(dataloader, disable=quiet):
        # batch = dataloader.shard_batch(batch)
        y_pred = _model_map(model, batch, keys)

        if denormalize:
            y_pred = dataloader.dataset.denormalize_target(y_pred)

        out_dict = {"basin": basin, "date": date, "y_pred": y_pred}

        if not inference_mode:
            y = batch["y"][:, -1, :]
            if denormalize:
                y = dataloader.dataset.denormalize_target(y)
            out_dict["y"] = y

        if "dynamic_dt" in batch.keys():
            dt_arr = np.stack(
                [v[:, -1] for v in batch["dynamic_dt"].values()],
                axis=1,
            )
            out_dict["dt"] = dt_arr

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
    dt_list = []

    # Iterate through the dataset, make predictions and collect data in lists.
    for result_dict in model_iterate(model, dataloader, quiet, denormalize):
        basins.extend(result_dict["basin"])
        dates.extend(result_dict["date"])
        y_hat_list.append(result_dict["y_pred"])
        if "y" in result_dict.keys():
            y_list.append(result_dict.get("y"))
        if "dt" in result_dict.keys():
            dt_list.append(result_dict.get("dt"))

    # Concatenate all the data lists into arrays.
    y_hat_arr = np.concatenate(y_hat_list)

    if len(y_list) > 0:
        y_arr = np.concatenate(y_list)
        data = np.concatenate((y_arr, y_hat_arr), axis=-1)
        cols = ["obs", "pred"]
    else:
        data = y_hat_arr
        cols = ["pred"]

    if dataloader.dataset.graph_mode:
        n_samples = len(basins)
        graph_nodes = y_arr.shape[-2]
        # Expand, reshape etc to get matching dimensions
        dates = np.repeat(dates, graph_nodes)  # Shape: (n_samples * graph_nodes,)
        basins = np.concatenate(basins)  # Shape: (n_samples * graph_nodes,)
        data = data.reshape(n_samples * graph_nodes, data.shape[-1])

    # Place the data arrays into a dataframe with multilevel indices.
    datetime_index = pd.MultiIndex.from_arrays(
        [basins, dates],
        names=["basin", "date"],
    )
    column_index = pd.MultiIndex.from_product(
        [cols, dataloader.dataset.target],
        names=["Type", "Feature"],
    )
    results = pd.DataFrame(
        data,
        index=datetime_index,
        columns=column_index,
    )

    # Might break if dt is implemented for graph mode. Not planned.
    if len(dt_list) > 0:
        dt_index = pd.MultiIndex.from_product(
            [["dt"], dataloader.dataset.features["dynamic"].keys()],
            names=["Type", "Feature"],
        )
        results[dt_index] = np.concatenate(dt_list)

    return results
