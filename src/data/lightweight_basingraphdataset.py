import warnings
import json
from typing import NamedTuple
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import xarray as xr
import dask.dataframe as dd
import dask.array as da
import zarr
import networkx as nx
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from jaxtyping import Array

from config import Config, DataSubset, Features


class GraphBatch(NamedTuple):
    dynamic: dict[str, Array]
    graph_edges: Array
    graph_idx: Array = None
    node_mask: Array = None
    edge_mask: Array = None
    static: Array = None
    y: Array = None

    def __getitem__(self, key):
        warnings.warn(
            f"Batch: dict-style access ('batch[\"{key}\"]') is deprecated. Use attribute access ('batch.{key}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)


# --- STEP 1: Define global variables to hold shared data ---
# They are initialized to None and will be populated by a setup function.
_LOG_PAD = 0.001
_SUBSET = None
_BASINS = None
_FEATURES = None
_TRAINING_STATS = None
_STATIC = None
_ZARR_STORES = None
_BASIN_GRAPHS = None
_BASIN_SUBBASIN_MAP = None
_SAMPLE_LIST = None
_BASIN_INDEX_MAP = None


def get_basin_index_map():
    return _BASIN_INDEX_MAP


def get_basin_subbasin_counts():
    return {basin: len(sub) for basin, sub in _BASIN_SUBBASIN_MAP.items()}


def return_globals():
    g = {
        "subset": _SUBSET,
        "basins": _BASINS,
        "features": _FEATURES,
        "training_stats": _TRAINING_STATS,
        "static_attr": _STATIC,
        "dynamic_data": _ZARR_STORES,
        "graphs": _BASIN_GRAPHS,
        "basin_subbasin_map": _BASIN_SUBBASIN_MAP,
        "sample_list": _SAMPLE_LIST,
        "basin_index_map": _BASIN_INDEX_MAP,
    }
    return g


# --- STEP 2: Create a function to initialize the globals ---
# This function does the heavy lifting ONCE in the main process.
def initialize_dataset_globals(cfg: Config, data_subset: DataSubset):
    """
    Loads all heavy data required by the dataset into global variables.
    This must be called once from the main process before creating a DataLoader.
    """
    global \
        _SUBSET, \
        _BASINS, \
        _FEATURES, \
        _TRAINING_STATS, \
        _STATIC, \
        _ZARR_STORES, \
        _BASIN_GRAPHS, \
        _BASIN_SUBBASIN_MAP, \
        _SAMPLE_LIST, \
        _BASIN_INDEX_MAP

    if _SUBSET == data_subset:
        print(f"Global data already initialized for {data_subset=}.")
        return
    _SUBSET = data_subset

    _BASINS = _get_basin_list(cfg, data_subset)
    _FEATURES = cfg.features

    # First get the encoding and normalization data based on training subset if needed
    if _TRAINING_STATS is None:
        _TRAINING_STATS = _get_training_stats(cfg)

    # Then load, encode, and scale the data for this subset
    _STATIC, _FEATURES, _BASIN_SUBBASIN_MAP = _load_attributes(
        cfg, _BASINS, _FEATURES, _TRAINING_STATS
    )
    _ZARR_STORES, _FEATURES = _open_zarr_store(cfg, _BASINS, _SUBSET, _FEATURES, _TRAINING_STATS)
    _BASIN_GRAPHS = _load_basin_graphs(cfg, _BASINS)
    _SAMPLE_LIST, _BASIN_INDEX_MAP = _create_indices(cfg, _BASINS, _SUBSET)

    print("✅ Global resources are loaded and ready.")


def _create_indices(cfg: Config, basins, subset: DataSubset):
    # First get the dates that can build a complete sequence
    time_slice = _get_time_slice(cfg, subset)
    seq_len = np.timedelta64(cfg.sequence_length, "D")
    min_train_date = np.datetime64(time_slice.start) + seq_len

    # For each basin, we will need to identify valid dates based on valid observations.
    def valid_target(ds):
        not_nan_arr = ~np.isnan(ds[cfg.features.target]).to_array()
        return not_nan_arr.any(dim=["subbasin", "variable"]).values

    # Loop through the basins and get a list of dates
    basin_date_map = {}
    for basin in tqdm(basins, disable=cfg.quiet, desc="Updating Indices"):
        basin_ds = _ZARR_STORES[basin]

        valid_seq_mask = basin_ds["date"] >= min_train_date
        basin_ds = basin_ds.sel(date=valid_seq_mask)

        # Mask out dates without valid data if we need it.
        if subset in ["train", "test"]:
            target_mask = valid_target(basin_ds)
            basin_date_map[basin] = basin_ds["date"][target_mask].values
        else:
            basin_date_map[basin] = basin_ds["date"].values

    # Now build unique indices for each pairing...
    # Master list of (basin, date) tuples that we use during __getitem__
    sample_list = []
    # Keep track of which indices in the master list belong to which basins for the sampler
    basin_index_map = {}
    index = 0
    for basin, dates in basin_date_map.items():
        basin_index_map[basin] = []
        for date in dates:
            sample_list.append((basin, date))
            basin_index_map[basin].append(index)
            index += 1

    return sample_list, basin_index_map


# --- Helper functions moved outside the class or made static ---
# These are called by initialize_dataset_globals to do the actual I/O.
def _get_basin_list(cfg: Config, subset: DataSubset):
    def read_file(fp):
        with open(fp, "r") as file:
            basin_list = [line.strip() for line in file.readlines()]
        return basin_list

    match subset:
        case DataSubset.train:
            basins = read_file(cfg.train_basin_file)
        case DataSubset.test:
            basins = read_file(cfg.test_basin_file)
        case DataSubset.predict:
            train = read_file(cfg.train_basin_file)
            test = read_file(cfg.test_basin_file)
            basins = list(set(train + test))
        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")

    return basins


def _get_time_slice(cfg: Config, subset: DataSubset):
    match subset:
        case DataSubset.train:
            start, end = tuple(cfg.train_date_range)
        case DataSubset.validate:
            start, end = tuple(cfg.validate_date_range)
        case DataSubset.test:
            start, end = tuple(cfg.test_date_range)
        case DataSubset.predict:
            start, end = tuple(cfg.predict_date_range)
        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")
    return slice(start, end)


def _load_attributes(
    cfg: Config, basins: list[str], features: Features, training_stats: dict
) -> xr.Dataset:
    """
    Loads the basin attributes from a CSV file.

    Returns:
        xr.Dataset: An xarray dataset of attribute data with basin coordinates.
    """
    print("Loading static attributes...", end="")
    df = pd.read_parquet(cfg.attributes_file)

    basin_subbasin_map = {
        basin: list(row["subbasin"]) for basin, row in df.reset_index().groupby("basin")
    }

    # Trim to training subset
    df = df[df.index.get_level_values("basin").isin(basins)]
    df = df.droplevel("basin")

    # Encode and scale the data.
    encoding = training_stats["s_encoding"]
    scale = training_stats["s_scale"]
    ds = df.to_xarray()
    ds, features = _encode_data(ds, "static", features, encoding)
    ds = _normalize_data(ds, scale)

    print("Done!")
    return ds, features, basin_subbasin_map


def _open_zarr_store(
    cfg: Config, basins: list[str], subset: DataSubset, features: Features, training_stats: dict
) -> dict[str, xr.Dataset]:
    """Opens all basin-specific Zarr groups and stores them as lazy datasets."""
    print("Opening dynamic data...")

    time_slice = _get_time_slice(cfg, subset)

    dynamic_columns = set([vv for v in cfg.features.dynamic.values() for vv in v])
    target_columns = set(cfg.features.target)
    all_config_columns = dynamic_columns.union(target_columns)

    encoding = training_stats["d_encoding"]
    scale = training_stats["d_scale"]

    if cfg.in_memory:
        print(f"Loading full dynamic dataset into memory ({cfg.in_memory=}).")
    else:
        print("Lazily loading each basin's dynamic data.")

    basin_datasets = {}
    globally_found_columns = set()

    for basin_id in tqdm(basins, disable=cfg.quiet, desc="Loading basins"):
        ds = xr.open_zarr(cfg.zarr_dir / basin_id)
        ds = ds.sel(date=time_slice)

        # Find and track missing columns
        available_columns = all_config_columns.intersection(set(ds.data_vars))
        globally_found_columns.update(available_columns)

        ds = ds[list(available_columns)]

        # # Remote sensing data are not always observed in every basin
        # available_columns = [col for col in columns if col in ds.data_vars]
        # ds = ds[available_columns]

        if cfg.in_memory:
            ds = ds.compute()

        ds, features = _encode_data(ds, "dynamic", features, encoding)
        ds = _normalize_data(ds, scale)

        basin_datasets[basin_id] = ds

    never_found_columns = all_config_columns - globally_found_columns
    if never_found_columns:
        raise KeyError(
            "The following dynamic columns from the config file were not found in ANY basin file."
            f"This may indicate a configuration error or typo: {sorted(list(never_found_columns))}"
        )

    return basin_datasets, features


def _load_basin_graphs(cfg: Config, basins: list[str]) -> dict[str, np.ndarray]:
    """Loads all basin-specific graphs."""
    print("Loading basin graphs...", end="")

    with open(cfg.graph_network_file) as f:
        graph_json = json.load(f)
    G = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges")

    graphs = {}
    for nodes in nx.weakly_connected_components(G):
        subG = G.subgraph(nodes).copy()

        # Find outlet: node with no outgoing edges
        outlets = [n for n in subG.nodes if subG.out_degree(n) == 0]
        if len(outlets) != 1:
            raise ValueError(f"Expected exactly one outlet per basin, found {outlets}")

        basin_id = outlets[0]
        if basin_id not in basins:
            continue

        # Map subbasin IDs to consecutive indices
        subbasin_map = {node: i for i, node in enumerate(subG.nodes())}

        source_nodes, dest_nodes = [], []
        for source, dest in subG.edges:
            source_nodes.append(subbasin_map[source])
            dest_nodes.append(subbasin_map[dest])
        graphs[basin_id] = np.array([source_nodes, dest_nodes], dtype=np.int32)

    missing_basins = set(basins) - set(graphs.keys())
    if missing_basins:
        raise ValueError(f"Not all basins were found in graph file. {missing_basins=}")

    print("Done!")
    return graphs


def _get_training_stats(cfg: Config) -> dict:
    print("Calculating training statistics for encoding and normalization...")

    train_basins = _get_basin_list(cfg, "train")
    static_df = pd.read_parquet(cfg.attributes_file)
    static_train_df = static_df[static_df.index.get_level_values("basin").isin(train_basins)]

    stats = {}
    stats["s_encoding"] = _get_static_encoding(static_train_df, cfg.static_encoding.model_dump())
    stats["s_scale"] = _calculate_scale_from_data(cfg, static_train_df, stats["s_encoding"])
    stats["d_encoding"] = _get_prescribed_encoding(cfg.dynamic_encoding.model_dump())
    stats["d_scale"] = _get_scale_from_precomp_stats(cfg, train_basins, stats["d_encoding"])

    return stats


def _get_prescribed_encoding(encoding):
    new_cols = []
    # Handle categorical columns
    if encoding["categorical"]:
        for col, categories in encoding["categorical"].items():
            if not categories:
                raise ValueError(
                    f"Dynamic categorical variable '{col}' must either be excluded or have "
                    "its categories prescribed in the config."
                )
            new_cols.extend([f"{col}_{cat}" for cat in categories])
    # Handle bitmask columns
    if encoding["bitmask"]:
        for col, bits in encoding["bitmask"].items():
            if not bits:
                raise ValueError(
                    f"Dynamic bitmask variable '{col}' must either be excluded or have "
                    "its bits prescribed in the config."
                )
            new_cols.extend([f"{col}_bit_{bit}" for bit in bits])

    encoding["encoded_columns"] = new_cols
    return encoding


def _get_scale_from_precomp_stats(cfg: Config, basins: list[str], encoding: dict):
    """
    Calculates mean and std for a subset of basins by aggregating pre-computed stats.
    Uses pre-stored attributes in each basin's Zarr group.
    """
    aggregated_stats = defaultdict(
        lambda: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "log_sum": 0.0,
        }
    )

    for basin_id in basins:
        try:
            basin_path = cfg.zarr_dir / basin_id
            z_group = zarr.open(basin_path, mode="r")
            basin_stats = z_group.attrs["normalization_stats"]

            for var, stats in basin_stats.items():
                agg = aggregated_stats[var]
                agg["count"] += stats["count"]
                agg["sum"] += stats["sum"]
                agg["sum_sq"] += stats["sum_sq"]
                agg["min"] = min(agg["min"], stats["min"])
                agg["max"] = max(agg["max"], stats["max"])
                agg["log_sum"] += stats["log_sum"]
        except Exception:
            raise KeyError(f"Failed to load normalization stats for {basin_path}")

    scale = {}
    all_vars = set(aggregated_stats.keys()).union(set(encoding["encoded_columns"]))

    for var in all_vars:
        scale[var] = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}

        if var in encoding["encoded_columns"]:
            scale[var]["encoded"] = True
            continue

        stats = aggregated_stats[var]
        total_count = stats["count"]

        if var in cfg.log_norm_cols:
            scale[var]["log_norm"] = True
            scale[var]["offset"] = stats["log_sum"] / total_count
        elif var in cfg.range_norm_cols:
            min_val, max_val = stats["min"], stats["max"]
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = stats["sum"] / total_count
            variance = (stats["sum_sq"] / total_count) - (mean**2)
            std = np.sqrt(max(variance, 0))
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


def _get_static_encoding(df: pd.DataFrame, encoding: dict):
    old_cols = []
    new_cols = []
    # Handle categorical columns
    for col, categories in encoding["categorical"].items():
        if not categories:
            categories = df[col].astype("category").unique()
        new_cols.extend([f"{col}_{cat}" for cat in categories])
        old_cols.append(col)
    # Handle bitmask columns
    for col, bits in encoding["bitmask"].items():
        if not bits:
            bit_data = df[col].astype(int)
            max_val = bit_data.max()
            nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
            # Check which bits are active across the dataset
            bits = [n for n in range(nbits) if ((bit_data // 2**n) % 2).any()]
        new_cols.extend([f"{col}_bit_{bit}" for bit in bits])
        old_cols.append(col)

    encoding["encoded_columns"] = new_cols
    encoding["removed_columns"] = old_cols
    return encoding


def _calculate_scale_from_data(cfg: Config, df: pd.DataFrame, encoding: dict):
    """
    Calculate normalization directly from dataset values.
    """
    new_cols = set(encoding["encoded_columns"])
    old_cols = set(encoding["removed_columns"])
    all_cols = set(df.columns).union(new_cols) - old_cols
    scale = {
        k: {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0} for k in all_cols
    }
    for var in all_cols:
        if var in new_cols:
            scale[var]["encoded"] = True
            continue
        if var in cfg.log_norm_cols:
            scale[var]["log_norm"] = True
            x = df[var] + _LOG_PAD
            scale[var]["offset"] = np.nanmean(np.log(x))
        elif var in cfg.range_norm_cols:
            min_val = float(df[var].min())
            max_val = float(df[var].max())
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = float(df[var].mean())
            std = float(df[var].std())
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


def _encode_data(ds: xr.Dataset, feat_group: str, features: Features, encoding: dict):
    assert feat_group in ["dynamic", "static"]

    # columns_in = ds.data_vars
    ds, features = _one_hot_encoding(ds, feat_group, features, encoding["categorical"])
    ds, features = _bitmask_expansion(ds, feat_group, features, encoding["bitmask"])
    # new_columns = set(ds.data_vars) - set(columns_in)
    # updated_encoding = {
    #     "categorical": categorical_enc,
    #     "bitmask": bitmask_enc,
    #     "encoded_columns": list(new_columns),
    # }

    return ds, features


def _one_hot_encoding(
    ds: xr.Dataset, feat_group: str, features: Features, cat_enc: dict[str, list[str]]
):
    for col, prescribed in cat_enc.items():
        if col not in ds.data_vars:
            continue

        # Convert to a lazy Dask DataFrame
        ddf = ds[[col]].to_dask_dataframe(set_index=True).categorize(columns=[col])
        # Perform one-hot encoding lazily using Dask
        encoded_ddf = dd.get_dummies(ddf, prefix=col, columns=[col])

        # Add missing prescribed columns
        for c in prescribed:
            if c not in encoded_ddf.columns:
                encoded_ddf[c] = 0

        # Ensure order and presence of prescribed columns
        encoded_ddf = encoded_ddf[prescribed]

        # Convert the lazy Dask DataFrame back to a lazy xarray.Dataset
        encoded_ds = xr.Dataset.from_dataframe(encoded_ddf)

        ds = ds.drop_vars(col)  # Drop the original column
        ds = xr.merge([ds, encoded_ds])  # Merge the lazy encoded columns

        # Update the feature dicts (logic remains the same)
        encoded_columns = list(encoded_ds.data_vars)
        if feat_group == "dynamic":
            # Loop through the sources
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        else:  # static
            features.static.remove(col)
            features.static.extend(encoded_columns)

    return ds, features


def _bitmask_expansion(
    ds: xr.Dataset, feat_group: str, features: Features, bitmask_enc: dict[str, list[int]]
):
    for col, prescribed_bits in bitmask_enc.items():
        if col not in ds.data_vars:
            continue

        # Operate directly on the lazy xarray.DataArray (which is a Dask array)
        x = ds[col]
        finite_mask = da.isfinite(x.data)
        x_int = da.where(finite_mask, x.data, 0).astype(int)

        new_vars = {}
        for n in prescribed_bits:
            # All these operations are lazy Dask operations
            bit_arr_data = ((x_int // 2**n) % 2).astype(float)
            bit_arr_data = da.where(finite_mask, bit_arr_data, np.nan)

            new_var_name = f"{col}_bit_{n}"
            new_vars[new_var_name] = xr.DataArray(
                bit_arr_data, dims=ds[col].dims, coords=ds[col].coords
            )

        encoded_ds = xr.Dataset(new_vars)
        ds = ds.drop_vars(col)
        ds = xr.merge([ds, encoded_ds], compat="no_conflicts")

        # Update the feature dicts (logic remains the same)
        encoded_columns = list(encoded_ds.data_vars)
        if feat_group == "dynamic":
            # Loop through the sources
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        else:  # static
            features.static.remove(col)
            features.static.extend(encoded_columns)

    return ds, features


def _normalize_data(ds: xr.Dataset, scale=None):
    """
    Normalize the input data using log normalization for specified variables and standard normalization for others.

    Returns:
        ds: the input xarray dataset after normalization
        scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
    """
    for var in set(ds.data_vars).intersection(scale.keys()):
        scl = scale[var]
        if scl["encoded"]:
            continue
        elif scl["log_norm"]:
            ds[var] = np.log(ds[var] + _LOG_PAD) - scl["offset"]
        else:
            if scl["scale"] == 0:
                ds[var] = ds[var] - scl["offset"]
            else:
                ds[var] = (ds[var] - scl["offset"]) / scl["scale"]

    return ds


class LightBasinGraphDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(self, cfg: Config, subset: DataSubset):
        self.cfg = cfg
        self.data_subset = subset

        # Check if globals have been loaded
        if _TRAINING_STATS is None:
            raise RuntimeError(
                "Global data not loaded. You must call initialize_dataset_globals() "
                "from your main script before creating a Dataset instance."
            )
        self.s_encoding = _TRAINING_STATS["s_encoding"]
        self.d_encoding = _TRAINING_STATS["d_encoding"]
        self.s_scale = _TRAINING_STATS["s_scale"]
        self.d_scale = _TRAINING_STATS["d_scale"]

        self.features = _FEATURES
        self.basins = _BASINS
        self.x_s = _STATIC
        self.basin_x_ds = _ZARR_STORES
        self.graphs = _BASIN_GRAPHS
        self.sample_list = _SAMPLE_LIST
        self.basin_subbasin_map = _BASIN_SUBBASIN_MAP
        self.target = cfg.features.target

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict:
        """Generate one batch of data."""
        basin, end_date = self.sample_list[idx]
        start_date = end_date - pd.Timedelta(days=self.cfg.sequence_length - 1)

        # get the basin xarray and slice by time
        date_slice = slice(start_date, end_date)
        ds = self.basin_x_ds[basin].sel(date=date_slice)

        # NaN padding
        # Identify all dynamic and target columns that are required for this sample.
        all_dynamic_cols = [col for cols in self.features.dynamic.values() for col in cols]
        all_needed_cols = set(all_dynamic_cols + self.target)
        missing_cols = all_needed_cols - set(ds.data_vars)
        # If any columns are missing, create NaN placeholders and assign them to the dataset
        # We do it here, instead of during data loading, because we only need to pad this slice.
        if missing_cols:
            nan_arrays_to_add = {}
            for col in missing_cols:
                nan_da = xr.DataArray(
                    data=np.full((len(ds.date), len(ds.subbasin)), np.nan, dtype=np.float32),
                    coords={"date": ds.date, "subbasin": ds.subbasin},
                    dims=["date", "subbasin"],
                )
                nan_arrays_to_add[col] = nan_da
            ds = ds.assign(**nan_arrays_to_add)

        dynamic = {}
        # select the dynamic, static, and target arrays
        for source, columns in self.features.dynamic.items():
            source_cols = [c for c in columns if c in ds.data_vars]
            source_arr = (
                ds[source_cols]
                .to_array(dim="variable")
                .transpose("date", "subbasin", "variable")
                .values
            )
            dynamic[source] = source_arr

        static = None
        if self.x_s is not None and self.features.static:
            subbasins = self.basin_subbasin_map[basin]
            static_ds = self.x_s.sel(subbasin=subbasins)
            static_arr = (
                static_ds[self.features.static]
                .to_array(dim="variable")
                .transpose("subbasin", "variable")
                .values
            )
            static = static_arr

        target = None
        if self.target:
            target_arr = (
                ds[self.target]
                .to_array(dim="variable")
                .transpose("date", "subbasin", "variable")
                .values
            )
            target = target_arr

        sample = GraphBatch(
            dynamic=dynamic, graph_edges=self.graphs[basin], static=static, y=target
        )

        return basin, end_date, sample

    def denormalize(self, x: Array, name: str) -> Array:
        """
        Denormalizes a feature or target by its name.

        Args:
            x (Array): Normalized data.
            name (str): Name of the variable to denormalize.

        Returns:
            np.ndarray or jnp.ndarray: Denormalized data.
        """
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            return jnp.exp(x + offset) - self.log_pad
        else:
            return x * scale + offset

    def denormalize_target(self, y_normalized: Array) -> Array:
        """
        Denormalizes the target variable(s).
        Returns:
            Array: Denormalized target data.
        """
        y = jnp.empty_like(y_normalized)

        for i in range(len(self.target)):
            target_name = self.target[i]
            denorm = self.denormalize(y_normalized[..., i], name=target_name)
            y = y.at[..., i].set(denorm)

        return y
