import copy
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple
import concurrent.futures

import dask.array as da
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from jax.tree import map as jt_map
from jaxtyping import Array
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config, DataSubset, Features


class GraphBatch(NamedTuple):
    dynamic: dict[str, Array]
    graph_edges: Array
    graph_idx: Array = None
    node_mask: Array = None
    edge_mask: Array = None
    static: Array = None
    y: dict[str, Array] = {}

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

    train_basin_dict = _get_basin_subbasin_dict(cfg, "train")
    train_basins = list(train_basin_dict.keys())
    train_subbasins = {x for s in train_basin_dict.values() for x in s}

    static_df = pd.read_parquet(cfg.attributes_file)
    subbasin_mask = static_df.index.get_level_values("subbasin").isin(train_subbasins)
    static_train_df = static_df[subbasin_mask]

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
    new_cols = {}
    # Handle categorical columns
    if encoding["categorical"]:
        for col, categories in encoding["categorical"].items():
            if not categories:
                raise ValueError(
                    f"Dynamic categorical variable '{col}' must either be excluded or have "
                    "its categories prescribed in the config."
                )
            new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    if encoding["bitmask"]:
        for col, bits in encoding["bitmask"].items():
            if not bits:
                raise ValueError(
                    f"Dynamic bitmask variable '{col}' must either be excluded or have "
                    "its bits prescribed in the config."
                )
            new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

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
            "log_sum_sq": 0.0,
            "positive_count": 0,
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
                agg["log_sum_sq"] += stats["log_sum_sq"]
                agg["positive_count"] += stats["positive_count"]
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

        if var in cfg.log_norm_cols:
            log_mean = stats["log_sum"] / stats["positive_count"]
            log_var = (stats["log_sum_sq"] / stats["positive_count"]) - log_mean**2
            log_std = np.sqrt(max(log_var, 1e-12))
            scale[var]["log_norm"] = True
            scale[var]["offset"] = log_mean
            scale[var]["scale"] = log_std
        elif var in cfg.range_norm_cols:
            min_val, max_val = stats["min"], stats["max"]
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = stats["sum"] / stats["count"]
            variance = (stats["sum_sq"] / stats["count"]) - (mean**2)
            std = np.sqrt(max(variance, 0))
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


def _get_static_encoding(df: pd.DataFrame, encoding: dict):
    new_cols = {}
    # Handle categorical columns
    for col, categories in encoding["categorical"].items():
        if not categories:
            categories = df[col].astype("category").unique()
            encoding["categorical"][col] = categories  # Add inferred categories to encoding
        new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    for col, bits in encoding["bitmask"].items():
        if not bits:
            bit_data = df[col].astype(int)
            max_val = bit_data.max()
            nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
            # Check which bits are active across the dataset
            bits = [n for n in range(nbits) if ((bit_data // 2**n) % 2).any()]
            encoding["bitmask"][col] = bits  # Add inferred bits to encoding
        new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

    encoding["encoded_columns"] = new_cols
    encoding["removed_columns"] = list(new_cols.keys())
    return encoding


def _calculate_scale_from_data(cfg: Config, df: pd.DataFrame, encoding: dict):
    """
    Calculate normalization directly from dataset values.
    """
    encoded_cols = set([c for cl in encoding["encoded_columns"] for c in cl])
    removed_cols = set(encoding["removed_columns"])
    all_cols = set(df.columns).union(encoded_cols) - removed_cols
    scale = {
        k: {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0} for k in all_cols
    }
    for var in all_cols:
        if var in encoded_cols:
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


# TODO Allow different sets of features, but check what is already in cache and append as needed.
# Each var is normalized separately, so there is no conflict with extra vars.
def config_hash(cfg: Config) -> str:
    """Return a unique hash based on a subset of fields from a Pydantic config object."""
    fields_to_hash = [
        "data_root",
        "zarr_dir",
        "attributes_file",
        "train_basin_file",
        "features",
        "dynamic_encoding",
        "static_encoding",
        "log_norm_cols",
        "range_norm_cols",
        "train_date_range",
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
        self.cache_dir = cfg.zarr_dir / "_cache4" / config_hash(cfg)
        self.train_stats = None
        print(f"Caches will be stored at: {self.cache_dir}")

    def create_cache(self, subset: str | DataSubset, overwrite: bool = False, num_workers: int = 16):
        """
        Create cache with parallel processing, combining normalization,
        target identification, and index generation in one pass.
        """
        if isinstance(subset, str):
            subset = DataSubset(subset)

        subset_cache_dir = self.cache_dir / subset.name
        subset_cache_dir.mkdir(exist_ok=True, parents=True)

        # Pre-compute normalization stats once
        if self.train_stats is None:
            self.train_stats = get_train_ds_stats(self.cfg, static=False)

        features = copy.deepcopy(self.cfg.features)
        encoding = self.train_stats["d_encoding"]
        scale = self.train_stats["d_scale"]
        dynamic_columns = set([vv for v in features.dynamic.values() for vv in v])
        all_config_columns = dynamic_columns.union(set(features.target))

        self._process_subset(
            subset=subset,
            subset_cache_dir=subset_cache_dir,
            overwrite=overwrite,
            num_workers=num_workers,
            features=features,
            encoding=encoding,
            scale=scale,
            all_config_columns=all_config_columns,
        )

        print("✅ All caches created and indexed.")

        return subset_cache_dir
        
    def _process_subset(
        self,
        subset: DataSubset,
        subset_cache_dir: Path,
        overwrite: bool,
        num_workers: int,
        features,
        encoding,
        scale,
        all_config_columns,
    ):
        """Process a single subset: cache data and generate indices in one pass."""

        basin_dict = _get_basin_subbasin_dict(self.cfg, subset)
        basins = sorted(list(basin_dict.keys()))

        # Filter out already cached basins if not overwriting
        if not overwrite:
            basins_to_process = [b for b in basins if not (subset_cache_dir / b).is_dir()]
        else:
            basins_to_process = basins

        if len(basins_to_process) == 0:
            return

        time_slice = _get_time_slice(self.cfg, subset)
        seq_len = np.timedelta64(self.cfg.sequence_length, "D")
        min_valid_date = np.datetime64(time_slice.start) + seq_len
        target_vars = self.cfg.features.target

        # Process basins in parallel, returning both cache status and index data
        globally_found_columns = set()
        all_basin_indices = {}  # basin -> list of (date_int,)
        written = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_basin_with_indices,
                    basin_id=basin_id,
                    cache_dir=subset_cache_dir,
                    time_slice=time_slice,
                    features=features,
                    encoding=encoding,
                    scale=scale,
                    all_config_columns=all_config_columns,
                    min_valid_date=min_valid_date,
                    target_vars=target_vars,
                    subset=subset,
                ): basin_id
                for basin_id in basins_to_process
            }

            with tqdm(
                total=len(futures), desc=f"Processing {subset.name}", disable=self.cfg.quiet
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    basin_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            found_columns, valid_date_ints = result
                            globally_found_columns.update(found_columns)
                            if valid_date_ints is not None and len(valid_date_ints) > 0:
                                all_basin_indices[basin_id] = valid_date_ints
                            written += 1
                    except Exception as e:
                        warnings.warn(f"Failed to process basin {basin_id}: {e}")
                    pbar.update(1)

        print(f"Wrote {written} new basin files to {subset.name} cache.")

        if written > 0:
            never_found_columns = all_config_columns - globally_found_columns
            if never_found_columns:
                warnings.warn(f"Columns not found in any basin: {sorted(never_found_columns)}")

        # For basins that were already cached, load their index data
        already_cached = set(basins) - set(basins_to_process)
        if already_cached:
            existing_indices = self._load_existing_indices(
                already_cached, subset_cache_dir, min_valid_date, target_vars, subset, all_basin_indices
            )
            all_basin_indices.update(existing_indices)

        # Save indices
        self._save_indices(subset_cache_dir, basins, all_basin_indices)

        return

    def _process_single_basin_with_indices(
        self,
        basin_id: str,
        cache_dir: Path,
        time_slice,
        features,
        encoding,
        scale,
        all_config_columns,
        min_valid_date,
        target_vars,
        subset: DataSubset,
    ):
        """
        Process a single basin: normalize, cache, and identify valid sample dates.
        Returns (found_columns, valid_date_ints) or None on failure.
        """
        try:
            # Load and normalize data (your existing logic)
            ds = self._load_and_normalize_basin(
                basin_id, time_slice, features, encoding, scale, all_config_columns
            )
            if ds is None:
                return None

            found_columns = set(ds.data_vars)

            # Write to cache
            basin_cache_path = cache_dir / basin_id
            ds.to_zarr(basin_cache_path, mode="w", consolidated=True)

            # Identify valid dates for indexing (while data is still in memory)
            valid_date_ints = self._extract_valid_dates(
                ds, min_valid_date, target_vars, subset
            )

            return found_columns, valid_date_ints

        except Exception as e:
            warnings.warn(f"Error processing {basin_id}: {e}")
            return None
        

    def _load_and_normalize_basin(
        self, 
        basin_id: str,
        time_slice: slice,
        features: Features,
        encoding: dict,
        scale: dict,
        all_config_columns: set) -> xr.Dataset:
        
        # Open with consolidation for faster metadata reads
        ds = xr.open_zarr(
            self.cfg.zarr_dir / basin_id,
            consolidated=True if (self.cfg.zarr_dir / basin_id / ".zmetadata").exists() else False
        )
        
        # Select time slice efficiently
        ds = ds.sel(date=time_slice)
        
        # Find available columns
        available_columns = all_config_columns.intersection(set(ds.data_vars))
        ds = ds[list(available_columns)]
        
        # Apply encoding and normalization
        ds, features = self._encode_data_optimized(ds, features, encoding)
        ds = _normalize_data(ds, scale)
        
        # Optimize chunking for the sequence length
        ds = ds.chunk({
            "date": self.cfg.sequence_length,
            "subbasin": -1  # Keep subbasin dimension in single chunk
        })
        
        return ds        


    def _encode_data_optimized(self, ds: xr.Dataset, features: Features, encoding: dict):
        """Optimized encoding without unnecessary conversions."""
        
        # Process categorical encodings
        for col, prescribed_cats in encoding["categorical"].items():
            if col not in ds.data_vars:
                continue
                
            prescribed_cols = [f"{col}_{cat}" for cat in prescribed_cats]
            
            # Direct numpy operations instead of dask dataframe conversion
            col_data = ds[col].values
            
            # Create one-hot encoded arrays directly
            encoded_arrays = {}
            for cat_col in prescribed_cols:
                cat_name = cat_col.replace(f"{col}_", "")
                encoded_arrays[cat_col] = xr.DataArray(
                    (col_data == cat_name).astype(float),
                    dims=ds[col].dims,
                    coords=ds[col].coords
                )
            
            # Update dataset
            ds = ds.drop_vars(col)
            for name, arr in encoded_arrays.items():
                ds[name] = arr
            
            # Update features
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(prescribed_cols)
        
        # Process bitmask encodings (keeping your efficient implementation)
        for col, prescribed_bits in encoding["bitmask"].items():
            if col not in ds.data_vars:
                continue
                
            x = ds[col]
            finite_mask = da.isfinite(x.data) if isinstance(x.data, da.Array) else np.isfinite(x.data)
            x_int = np.where(finite_mask, x.data, 0).astype(int)
            
            new_vars = {}
            for n in prescribed_bits:
                bit_arr_data = ((x_int // 2**n) % 2).astype(float)
                bit_arr_data = np.where(finite_mask, bit_arr_data, np.nan)
                
                new_var_name = f"{col}_bit_{n}"
                new_vars[new_var_name] = xr.DataArray(
                    bit_arr_data, 
                    dims=ds[col].dims, 
                    coords=ds[col].coords
                )
            
            # Update dataset and features
            encoded_columns = list(new_vars.keys())
            ds = ds.drop_vars(col)
            ds = ds.assign(**new_vars)
            
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        
        return ds, features

    def _extract_valid_dates(
        self,
        ds: xr.Dataset,
        min_valid_date: np.datetime64,
        target_vars: list,
        subset: DataSubset,
    ) -> np.ndarray:
        """Extract valid sample dates from an in-memory dataset."""
        # Filter by minimum date
        valid_seq_mask = ds["date"] >= min_valid_date
        ds_filtered = ds.sel(date=valid_seq_mask)

        if subset in [DataSubset.train, DataSubset.test]:
            # Check target availability
            missing = [t for t in target_vars if t not in ds_filtered]
            if missing:
                return np.array([], dtype=np.int32)

            not_nan_arr = ~np.isnan(ds_filtered[target_vars]).to_array()
            target_mask = not_nan_arr.any(dim=["subbasin", "variable"]).values
            valid_dates = ds_filtered["date"][target_mask].values
        else:
            valid_dates = ds_filtered["date"].values

        if len(valid_dates) == 0:
            return np.array([], dtype=np.int32)

        # Convert to integer days since epoch
        return valid_dates.astype("datetime64[D]").astype(np.int32)

    def _load_existing_indices(
        self,
        basins: set,
        cache_dir: Path,
        min_valid_date,
        target_vars,
        subset,
    ) -> dict:
        """Load index data for already-cached basins."""

        loaded_indices = {}
        for basin in tqdm(basins, desc="Loading existing indices", disable=self.cfg.quiet):
            try:
                ds = xr.open_zarr(cache_dir / basin, consolidated=True)
                valid_date_ints = self._extract_valid_dates(
                    ds, min_valid_date, target_vars, subset
                )
                if len(valid_date_ints) > 0:
                    loaded_indices[basin] = valid_date_ints
            except Exception:
                continue
        return loaded_indices

    def _save_indices(self, subset_cache_dir: Path, basins: list, all_basin_indices: dict):
        """Save the computed indices to disk."""
        index_dir = subset_cache_dir / '_indices' / f"sequence={str(self.cfg.sequence_length)}"
        index_dir.mkdir(exist_ok=True, parents=True)

        out_list_path = index_dir / "sample_list.npy"
        out_map_path = index_dir / "basin_index_map.json"
        out_lookup_path = index_dir / "basin_lookup.json"

        # Create basin -> int mapping
        basin_to_int = {b: i for i, b in enumerate(basins)}

        # Build flattened sample list and per-basin index map
        sample_list_ints = []
        basin_index_map = defaultdict(list)
        global_idx = 0

        for basin in basins:
            if basin not in all_basin_indices:
                continue
            basin_int = basin_to_int[basin]
            for d_int in all_basin_indices[basin]:
                sample_list_ints.append([basin_int, d_int])
                basin_index_map[basin].append(global_idx)
                global_idx += 1

        # Save files
        with open(out_lookup_path, "w") as f:
            json.dump(basins, f)

        with open(out_map_path, "w") as f:
            json.dump(basin_index_map, f)

        np.save(out_list_path, np.array(sample_list_ints, dtype=np.int32))

        print(f"  Saved {global_idx} samples to {subset_cache_dir}")


# --- Helper functions moved outside the class or made static ---
# These are called by initialize_dataset_globals to do the actual I/O.

def _get_basin_subbasin_dict(cfg: Config, subset: DataSubset):
    def read_file(fp) -> dict:
        df = pd.read_csv(fp, dtype=str)
        if "basin" not in df.columns or "subbasin" not in df.columns:
            raise ValueError(f"Subbasin file {fp} must contain 'basin' and 'subbasin' columns.")
            
        return df.groupby("basin")["subbasin"].apply(set).to_dict()
    
    match subset:
        case DataSubset.train:
            basin_dict = read_file(cfg.train_basin_file)
        case DataSubset.test:
            basin_dict = read_file(cfg.test_basin_file)
        case DataSubset.predict:
            train = read_file(cfg.train_basin_file)
            test = read_file(cfg.test_basin_file)
            basin_dict = {
                b: train.get(b, set()) | test.get(b, set())
                for b in train.keys() | test.keys()
            }
        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")
        
    return basin_dict


def _get_time_slice(cfg: Config, subset: DataSubset):
    match subset:
        case DataSubset.train:
            start, end = tuple(cfg.train_date_range)
        case DataSubset.validate:
            start, end = tuple(cfg.validate_date_range)
        case DataSubset.test:
            start, end = tuple(cfg.test_date_range)
        case DataSubset.predict:
            ranges = [
                cfg.train_date_range,
                cfg.validate_date_range,
                cfg.test_date_range,
            ]
            start = min(r[0] for r in ranges)
            end = max(r[1] for r in ranges)
        
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
    for col, prescribed_cats in cat_enc.items():
        if col not in ds.data_vars:
            continue

        # Get the underlying dask array
        da_col = ds[col].data
        
        # Create a dictionary of new variables
        new_vars = {}
        for cat in prescribed_cats:
            # Broadcast comparison (lazy dask operation)
            # Use strict equality; assumes categories match exactly
            if isinstance(cat, str):
                 # handle string categories if necessary, though usually encoded as int/float in zarr
                 # Ideally, your input Zarr already has these as numerics or strict types
                 pass 
            
            mask = (da_col == cat)
            
            # Convert boolean mask to float (or int)
            new_var_name = f"{col}_{cat}"
            new_vars[new_var_name] = xr.DataArray(
                mask.astype(np.float32), 
                dims=ds[col].dims, 
                coords=ds[col].coords
            )

        # Merge new variables
        encoded_ds = xr.Dataset(new_vars)
        ds = ds.drop_vars(col)
        ds = xr.merge([ds, encoded_ds], compat="no_conflicts")

        # Update feature tracking (same logic as before)
        encoded_columns = list(encoded_ds.data_vars)
        if feat_group == "dynamic":
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        else:
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
        # print(f"BME: Removing {col}, adding {encoded_columns}")
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
            ds[var] = (np.log1p(ds[var]) - scl["offset"]) / scl['scale']
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

    def __init__(self, cfg: Config, cache_dir: Path, subset: str | DataSubset):
        self.cfg = cfg
        self.cache_dir = cache_dir

        if isinstance(subset, str):
            subset = DataSubset(subset)
        self.data_subset = subset

        self.features = copy.deepcopy(self.cfg.features)
        self.time_slice = _get_time_slice(cfg, subset)

        self.basin_dict = _get_basin_subbasin_dict(cfg, subset)
        self.basins = list(self.basin_dict.keys())
        self.subbasins = {x for s in self.basin_dict.values() for x in s}

        train_stats = get_train_ds_stats(cfg)
        self.s_encoding = train_stats["s_encoding"]
        self.s_scale = train_stats["s_scale"]
        self.d_encoding = train_stats["d_encoding"]
        self.d_scale = train_stats["d_scale"]
        self._update_encoded_features()

        self.x_s, self.basin_subbasin_map = self._load_attributes()
        self.graphs = self._load_basin_graphs()

        self.target = self.features.target
        self.basin_lookup, self.sample_list, self.basin_index_map = self._load_indices()


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

    def _load_indices(self):
        subset_name = self.data_subset.name
        index_dir = self.cache_dir / '_indices' / f"sequence={str(self.cfg.sequence_length)}"

        lookup_path = index_dir / "basin_lookup.json"
        list_path = index_dir / "sample_list.npy"
        map_path = index_dir / "basin_index_map.json"

        if not list_path.exists():
            raise FileNotFoundError(f"Index cache not found for {subset_name}. Run DynamicCacheManager.")

        print(f"Loading optimized indices for {subset_name}...", end="")
        
        # 1. Load Basin Lookup (List of strings)
        with open(lookup_path, "r") as f:
            basin_lookup = json.load(f)
            
        # 2. Load Sample List (Memory Mapped Integer Array)
        # mmap_mode='r' means it stays on disk and loads pages into RAM only when accessed.
        # This reduces initial RAM usage to near zero for this object.
        sample_list = np.load(list_path, mmap_mode='r')
        
        # 3. Load Index Map (for Sampler)
        with open(map_path, "r") as f:
            basin_index_map = json.load(f)

        print("Done!")
        return basin_lookup, sample_list, basin_index_map

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sample_list)

    def _update_encoded_features(self):
        for old_col, new_cols in self.d_encoding["encoded_columns"].items():
            # Have to comb through the sources
            for source, source_cols in self.features.dynamic.items():
                if old_col in source_cols:
                    self.features.dynamic[source].remove(old_col)
                    self.features.dynamic[source].extend(new_cols)
                    break

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

    def __getitem__(self, idx: int) -> dict:
        """Generate one batch of data."""

        # 1. Retrieve integers (returns numpy.int32 scalars)
        basin_idx, date_int = self.sample_list[idx]

        # 2. Decode Basin ID
        basin = self.basin_lookup[basin_idx]

        # 3. Decode Date
        # FIXED: Cast date_int to native python int first
        end_date = np.datetime64(int(date_int), 'D') 

        start_date = end_date - np.timedelta64(self.cfg.sequence_length - 1, 'D')

        # basin, end_date = self.sample_list[idx]
        # start_date = end_date - pd.Timedelta(days=self.cfg.sequence_length - 1)

        # basin_ds = self.basin_ds[basin]
        with xr.open_zarr(self.cache_dir / basin, consolidated=True) as basin_ds:
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

            target = {}
            if not self.cfg.model_args.seq2seq:
                ds = ds.sel(date=[end_date])

            for t_name in self.target:
                target_arr = (
                    ds[self.target]
                    .to_array(dim="variable")
                    .transpose("date", "subbasin", "variable")
                    .compute()
                )
                target[t_name] = np.asarray(target_arr)

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
            return np.expm1(x * scale + offset)
        else:
            return x * scale + offset

    def denormalize_std(self, mu_normalized: Array, sigma_normalized: Array, name: str) -> Array:
        """
        Denormalizes standard deviation. For log-normal, requires both mu and sigma.
        """
        offset = self.d_scale[name]["offset"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            # mu and sigma are in log-space (normalized)
            mu_log = mu_normalized + offset
            sigma_log = sigma_normalized

            # Log-normal variance in original space
            # https://en.wikipedia.org/wiki/Log-normal_distribution
            var_original = (np.exp(sigma_log**2) - 1) * np.exp(2 * mu_log + sigma_log**2)
            return np.sqrt(var_original)
        else:
            # Z-score and range-norm case
            scale = self.d_scale[name]["scale"]
            return sigma_normalized * scale

    def denormalize_targets(self, y_normalized: dict[str, Array]) -> dict[str, Array]:
        """
        Denormalizes the target variable(s).
        Returns:
            dict[str, Array]: Denormalized target data.
        """
        y_denorm = {}
        for t_name, t_norm in y_normalized.items():
            y_denorm[t_name] = self.denormalize(t_norm, name=t_name)

        return y_denorm

    def normalize(self, x: Array, name: str) -> Array:
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]
        encoded = self.d_scale[name]["encoded"]

        if encoded:
            raise ValueError(f"Cannot normalize an encoded column: {name}")

        if log_norm:
            return np.log(x + _LOG_PAD) - offset
        elif scale == 0:
            return (x - offset) / scale
        else:
            return x - offset
