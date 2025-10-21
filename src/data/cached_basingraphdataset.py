import warnings
import json
import hashlib
import copy
from typing import NamedTuple
from pathlib import Path
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
from jax.tree import map as jt_map
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

    def to_jax(self):
        """Convert all numpy arrays to jax arrays."""
        jax_batch = jt_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, self)
        return jax_batch


_LOG_PAD = 0.001


def get_train_ds_stats(cfg: Config, *, dynamic: bool = True, static: bool = True) -> dict:
    print("Calculating training statistics for encoding and normalization...")

    train_basins = _get_basin_list(cfg, "train")
    static_df = pd.read_parquet(cfg.attributes_file)
    static_train_df = static_df[static_df.index.get_level_values("basin").isin(train_basins)]

    stats = {}
    if static:
        stats["s_encoding"] = _get_static_encoding(
            static_train_df, cfg.static_encoding.model_dump()
        )
        stats["s_scale"] = _calculate_scale_from_data(cfg, static_train_df, stats["s_encoding"])
    if dynamic:
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


def config_hash(cfg: Config) -> str:
    """Return a unique hash based on a subset of fields from a Pydantic config object."""
    fields_to_hash = [
        "data_root",
        "zarr_dir",
        "attributes_file",
        "train_basin_file",
        "test_basin_file",
        "graph_network_file",
        "features ",
        "time_gaps",
        "dynamic_encoding",
        "static_encoding",
        "log_norm_cols",
        "range_norm_cols",
        "train_date_range",
        "validate_date_range",
        "test_date_range",
        "predict_date_range",
        "add_rolling_means",
        "clip_feature_range",
        "value_filter",
    ]
    # Extract selected field values into dict
    selected = {f: getattr(cfg, f) for f in fields_to_hash if hasattr(cfg, f)}
    # Convert to JSON string for deterministic serialization
    serialized = json.dumps(selected, sort_keys=True, default=str)
    # Compute SHA256 hash
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class DynamicCacheManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_dir = cfg.zarr_dir / "_cache" / config_hash(cfg)
        self.train_stats = None  # lazy
        print(f"Caches will be stored at: {self.cache_dir}/<subset>")

    def create_cache(self, subset: str, overwrite: bool = False):
        sub_cache_dir = self.cache_dir / subset
        sub_cache_dir.mkdir(exist_ok=True, parents=True)

        basins = _get_basin_list(self.cfg, subset)
        time_slice = _get_time_slice(self.cfg, subset)
        features = copy.deepcopy(self.cfg.features)

        globally_found_columns = set()
        written = 0

        for basin_id in tqdm(basins, disable=self.cfg.quiet, desc="Loading basins"):
            if not overwrite and (sub_cache_dir / basin_id).is_dir():
                continue

            ds = xr.open_zarr(self.cfg.zarr_dir / basin_id)
            ds = ds.sel(date=time_slice)

            if self.train_stats is None:
                self.train_stats = get_train_ds_stats(self.cfg, static=False)
                encoding = self.train_stats["d_encoding"]
                scale = self.train_stats["d_scale"]
                dynamic_columns = set([vv for v in features.dynamic.values() for vv in v])
                target_columns = set([vv for v in features.target.values() for vv in v])
                all_config_columns = dynamic_columns.union(target_columns)

            # Find and track missing columns
            available_columns = all_config_columns.intersection(set(ds.data_vars))
            globally_found_columns.update(available_columns)
            ds = ds[list(available_columns)]

            ds, features = _encode_data(ds, "dynamic", features, encoding)
            ds = _normalize_data(ds, scale)
            ds = ds.chunk({"date": self.cfg.sequence_length})

            ds.to_zarr(store=sub_cache_dir, mode="w", group=basin_id)
            written += 1

        print(f"Wrote {written} new basin files to cache.")
        if written > 0:
            never_found_columns = all_config_columns - globally_found_columns
            if never_found_columns:
                warnings.warn(
                    "The following dynamic columns from the config file were not found in ANY of the processed basin files."
                    f"This may indicate a configuration error or typo: {sorted(list(never_found_columns))}"
                )

            # self._cache_indices(sub_cache_dir, subset, basins, time_slice, features)

        print("✅ Cached resources are loaded and ready.")
        return sub_cache_dir

    # def _cache_indices(self, sub_cache_dir: Path, subset: str, basins: list[str], time_slice: slice, features: Features):
    #     """Computes and caches the sample_list and basin_index_map."""
    #     seq_len = self.cfg.sequence_length
    #     sample_list_path = sub_cache_dir / f"sample_list_{seq_len}.json"
    #     basin_index_map_path = sub_cache_dir / f"basin_index_map_{seq_len}.json"

    #     # Avoid recalculating if indices for this sequence length already exist
    #     if sample_list_path.exists() and basin_index_map_path.exists():
    #         print(f"Indices for sequence length {seq_len} already exist. Skipping.")
    #         return

    #     seq_len = np.timedelta64(self.cfg.sequence_length, "D")
    #     min_train_date = np.datetime64(time_slice.start) + seq_len

    #     targets = [v for v_lists in features.target.values() for v in v_lists]
    #     def valid_target(ds):
    #         available_targets = [t for t in targets if t in ds.data_vars]
    #         if not available_targets:
    #             return np.zeros(len(ds['date']), dtype=bool) # No targets found, so no valid dates
    #         not_nan_arr = ~np.isnan(ds[available_targets].to_array())
    #         return not_nan_arr.any(dim=["subbasin", "variable"]).values

    #     basin_date_map = {}
    #     for basin in tqdm(basins, disable=self.cfg.quiet, desc="Building sampling index"):
    #         try:
    #             with xr.open_zarr(sub_cache_dir / basin, consolidated=False) as basin_ds:
    #                 valid_seq_mask = basin_ds["date"] >= min_train_date
    #                 basin_ds = basin_ds.sel(date=valid_seq_mask)

    #                 if subset in ["train", "test"]:
    #                     target_mask = valid_target(basin_ds)
    #                     dates = basin_ds["date"][target_mask].values
    #                 else:
    #                     dates = basin_ds["date"].values
    #                 # Convert numpy.datetime64 to string for JSON serialization
    #                 basin_date_map[basin] = [str(d) for d in dates]
    #         except Exception as e:
    #             warnings.warn(f"Could not process basin {basin} for index creation: {e}")

    #     sample_list = []
    #     basin_index_map = {}
    #     index = 0
    #     for basin, dates in basin_date_map.items():
    #         basin_index_map[basin] = []
    #         for date in dates:
    #             sample_list.append([basin, date]) # Use list for JSON
    #             basin_index_map[basin].append(index)
    #             index += 1

    #     # Save to parameterized JSON files
    #     with open(sample_list_path, "w") as f:
    #         json.dump(sample_list, f)
    #     with open(basin_index_map_path, "w") as f:
    #         json.dump(basin_index_map, f)

    #     print(f"✅ Cached {len(sample_list)} indices.")


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


def _encode_data(ds: xr.Dataset, feat_group: str, features: Features, encoding: dict):
    assert feat_group in ["dynamic", "static"]

    ds, features = _one_hot_encoding(ds, feat_group, features, encoding["categorical"])
    ds, features = _bitmask_expansion(ds, feat_group, features, encoding["bitmask"])

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


class CachedBasinGraphDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(self, cfg: Config, cache_dir: Path, subset: DataSubset):
        self.cfg = cfg
        self.cache_dir = cache_dir

        self.data_subset = subset

        self.features = copy.deepcopy(self.cfg.features)
        self.time_slice = _get_time_slice(cfg, subset)
        self.basins = _get_basin_list(cfg, subset)

        train_stats = get_train_ds_stats(cfg)
        self.s_encoding = train_stats["s_encoding"]
        self.s_scale = train_stats["s_scale"]
        self.d_encoding = train_stats["d_encoding"]
        self.d_scale = train_stats["d_scale"]

        self.x_s, self.basin_subbasin_map = self._load_attributes()
        self.graphs = self._load_basin_graphs()

        self.target = [v for v_lists in self.features.target.values() for v in v_lists]
        self.sample_list, self.basin_index_map = self._create_indices()

        # print("Loading sample indices from cache...", end="")
        # try:
        #     with open(self.cache_dir / "sample_list.json", "r") as f:
        #         self.sample_list = json.load(f)
        #     with open(self.cache_dir / "basin_index_map.json", "r") as f:
        #         self.basin_index_map = json.load(f)
        # except FileNotFoundError:
        #     raise FileNotFoundError(
        #         f"Index files not found in {self.cache_dir}. "
        #         "Please run the cache creation process first."
        #     )
        # print("Done!")

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sample_list)

    def _load_basin_graphs(self) -> dict[str, np.ndarray]:
        """Loads all basin-specific graphs."""
        print("Loading basin graphs...", end="")

        with open(self.cfg.graph_network_file) as f:
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
            if basin_id not in self.basins:
                continue

            # Map subbasin IDs to consecutive indices
            subbasin_map = {node: i for i, node in enumerate(subG.nodes())}

            source_nodes, dest_nodes = [], []
            for source, dest in subG.edges:
                source_nodes.append(subbasin_map[source])
                dest_nodes.append(subbasin_map[dest])
            graphs[basin_id] = np.array([source_nodes, dest_nodes], dtype=np.int32)

        missing_basins = set(self.basins) - set(graphs.keys())
        if missing_basins:
            raise ValueError(f"Not all basins were found in graph file. {missing_basins=}")

        print("Done!")
        return graphs

    def _load_attributes(self) -> xr.Dataset:
        """
        Loads the basin attributes from a CSV file.

        Returns:
            xr.Dataset: An xarray dataset of attribute data with basin coordinates.
        """
        print("Loading static attributes...", end="")
        df = pd.read_parquet(self.cfg.attributes_file)
        # Trim to basin subset
        df = df[df.index.get_level_values("basin").isin(self.basins)]

        basin_subbasin_map = {
            basin: list(row["subbasin"]) for basin, row in df.reset_index().groupby("basin")
        }

        df = df.droplevel("basin")

        # Encode and scale the data.
        ds = df.to_xarray()
        ds, self.features = _encode_data(ds, "static", self.features, self.s_encoding)
        ds = _normalize_data(ds, self.s_scale)

        print("Done!")
        return ds, basin_subbasin_map

    def _create_indices(self):
        # First get the dates that can build a complete sequence
        seq_len = np.timedelta64(self.cfg.sequence_length, "D")
        min_train_date = np.datetime64(self.time_slice.start) + seq_len

        # For each basin, we will need to identify valid dates based on valid observations.
        def valid_target(ds):
            not_nan_arr = ~np.isnan(ds[self.target]).to_array()
            return not_nan_arr.any(dim=["subbasin", "variable"]).values

        # Loop through the basins and get a list of dates
        basin_date_map = {}
        for basin in tqdm(self.basins, disable=self.cfg.quiet, desc="Building sampling index"):
            basin_ds = xr.open_zarr(self.cache_dir / basin, consolidated=False)

            valid_seq_mask = basin_ds["date"] >= min_train_date
            basin_ds = basin_ds.sel(date=valid_seq_mask)

            # Mask out dates without valid data if we need it.
            if self.data_subset in ["train", "test"]:
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

    def __getitem__(self, idx: int) -> dict:
        """Generate one batch of data."""
        basin, end_date = self.sample_list[idx]
        start_date = end_date - pd.Timedelta(days=self.cfg.sequence_length - 1)

        # basin_ds = self.basin_ds[basin]
        with xr.open_zarr(self.cache_dir / basin, consolidated=False) as basin_ds:
            date_slice = slice(start_date, end_date)
            ds = basin_ds.sel(date=date_slice)

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
            for source, columns in self.features.dynamic.items():
                source_cols = [c for c in columns if c in ds.data_vars]
                source_arr = (
                    ds[source_cols]
                    .to_array(dim="variable")
                    .transpose("date", "subbasin", "variable")
                    .compute()  # Explicitly compute
                )
                dynamic[source] = np.asarray(source_arr)  # Ensure numpy, not dask

        target = None
        if self.target:
            target_arr = (
                ds[self.target]
                .to_array(dim="variable")
                .transpose("date", "subbasin", "variable")
                .compute()
            )
            target = np.asarray(target_arr)

        static = None
        if self.x_s is not None and self.features.static:
            subbasins = self.basin_subbasin_map[basin]
            static_ds = self.x_s.sel(subbasin=subbasins)
            static_arr = (
                static_ds[self.features.static]
                .to_array(dim="variable")
                .transpose("subbasin", "variable")
                .compute()
            )
            static = np.asarray(static_arr)

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
            np.ndarray: Denormalized data.
        """
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            return np.exp(x + offset) - _LOG_PAD
        else:
            return x * scale + offset

    def denormalize_target(self, y_normalized: Array) -> Array:
        """
        Denormalizes the target variable(s).
        Returns:
            Array: Denormalized target data.
        """
        y = np.empty_like(y_normalized)

        for i in range(len(self.target)):
            target_name = self.target[i]
            denorm = self.denormalize(y_normalized[..., i], name=target_name)
            y[..., i] = denorm

        return y
