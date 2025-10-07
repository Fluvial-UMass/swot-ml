import warnings
import json
import copy
from typing import NamedTuple
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import xarray as xr
import dask.dataframe as dd
import dask.array as da
import zarr
from zarr.errors import ZarrUserWarning
import networkx as nx
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from jaxtyping import Array

from config import Config, DataSubset


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


class BasinGraphDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(self, cfg: Config, subset: DataSubset, *, for_stats: bool = False):
        self.cfg = cfg
        self.log_pad = 0.001
        self.data_subset = subset

        if subset == DataSubset.train:
            self.s_encoding = self.cfg.static_encoding.model_dump()
            self.d_encoding = self.cfg.dynamic_encoding.model_dump()
            self.s_scale = self.d_scale = None
        elif not for_stats:
            train_stats = self.get_training_stats(cfg)
            self.s_encoding = train_stats["s_encoding"]
            self.s_scale = train_stats["s_scale"]
            self.d_encoding = train_stats["d_encoding"]
            self.d_scale = train_stats["d_scale"]
        else:
            # This prevents infinite recursion from the get_training_stats method
            raise ValueError(f"{for_stats=} only allowed with subset='train'.")

        self.features = self.cfg.features.model_dump()  # dump to dict
        self.target = [v for v_lists in self.features["target"].values() for v in v_lists]

        dynamic_sources = set(self.features["dynamic"].keys())
        target_sources = set(self.features["target"].keys())
        self.sources = dynamic_sources.union(target_sources)

        self.time_slice = self._get_time_slice()
        self.basins = self._read_basin_files()
        self.x_s = self._load_attributes()

        if not for_stats:
            self.basin_x_ds = self._open_zarr_store()
            self.graphs = self._load_basin_graphs()
            self.update_indices()
        else:
            self.d_encoding = self._get_dummy_dynamic_encoding()
            self.d_scale = self._get_scale_from_precomp_stats(self.d_encoding)

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sequence_indices)

    def _get_time_slice(self):
        match self.data_subset:
            case DataSubset.train:
                start, end = tuple(self.cfg.train_date_range)
            case DataSubset.validate:
                start, end = tuple(self.cfg.validate_date_range)
            case DataSubset.test:
                start, end = tuple(self.cfg.test_date_range)
            case DataSubset.predict:
                start, end = tuple(self.cfg.predict_date_range)
            case _:
                raise ValueError(f"This data_subset ({self.data_subset}) is not implemented.")

        return slice(start, end)

    def _read_basin_files(self):
        def read_file(fp):
            with open(fp, "r") as file:
                lines = file.readlines()
                basin_list = [line.strip() for line in lines]
            return basin_list

        match self.data_subset:
            case DataSubset.train:
                basins = read_file(self.cfg.train_basin_file)
            case DataSubset.test:
                basins = read_file(self.cfg.test_basin_file)
            case DataSubset.predict:
                train = read_file(self.cfg.train_basin_file)
                test = read_file(self.cfg.test_basin_file)
                basins = list(set(train + test))
            case _:
                raise ValueError(f"This data_subset ({self.data_subset}) is not implemented.")

        return basins

    def _open_zarr_store(self) -> xr.Dataset:
        """Opens all basin-specific Zarr groups and stores them as lazy datasets."""
        print("Opening dynamic data...")
        warnings.filterwarnings("ignore", category=ZarrUserWarning)

        dynamic_columns = set([vv for v in self.features["dynamic"].values() for vv in v])
        target_columns = set([vv for v in self.features["target"].values() for vv in v])
        all_config_columns = dynamic_columns.union(target_columns)

        if self.cfg.in_memory:
            print(f"Loading full dynamic dataset into memory ({self.cfg.in_memory=}).")
        else:
            print("Lazily loading each basin's dynamic data.")

        basin_datasets = {}
        self.basin_subbasin_map = {}
        globally_found_columns = set()

        for basin_id in tqdm(self.basins, disable=self.cfg.quiet, desc="Loading basins"):
            ds = xr.open_zarr(self.cfg.zarr_dir / basin_id)
            ds = ds.sel(date=self.time_slice)

            # Find and track missing columns
            available_columns = all_config_columns.intersection(set(ds.data_vars))
            globally_found_columns.update(available_columns)

            ds = ds[list(available_columns)]

            # # Remote sensing data are not always observed in every basin
            # available_columns = [col for col in columns if col in ds.data_vars]
            # ds = ds[available_columns]

            if self.cfg.in_memory:
                ds = ds.compute()

            # Encoding is fully prescribed in the config or from the
            ds, updated_enc = self._encode_data(ds, "dynamic", self.d_encoding)
            self.d_encoding = updated_enc
            # During training, this will calculate the global training normalization stats based
            # on the metadata in the training basin zarr files the first time it is called,
            # then reuse the scaling in the repeated basins.
            ds, self.d_scale = self._normalize_data(ds, "dynamic", self.d_encoding, self.d_scale)

            basin_datasets[basin_id] = ds

            subbasin_ids = ds.subbasin.values.tolist()  # This is small, OK to compute
            self.basin_subbasin_map[basin_id] = subbasin_ids

        self.basin_subbasin_counts = {
            basin: len(subbasins) for basin, subbasins in self.basin_subbasin_map.items()
        }

        never_found_columns = all_config_columns - globally_found_columns
        if never_found_columns:
            raise KeyError(
                "The following dynamic columns from the config file were not found in ANY basin file."
                f"This may indicate a configuration error or typo: {sorted(list(never_found_columns))}"
            )

        warnings.resetwarnings()
        return basin_datasets

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

        basin_mask = df.index.get_level_values("basin").isin(self.basins)
        df = df[basin_mask]

        # 2. Extract the basin-to-subbasin mapping into a separate Series
        # This Series will have 'subbasin' as its index and 'basin' as its values
        subbasin_to_basin_map = df.index.to_frame(index=False).set_index("subbasin")["basin"]

        # 3. Drop the 'basin' level to proceed with data-only processing
        df = df.droplevel("basin")

        if self.data_subset != DataSubset.train:
            unencoded_cols = [k for k, v in self.s_scale.items() if not v["encoded"]]
            categorical_cols = list((self.s_encoding["categorical"] or {}).keys())
            bitmask_cols = list((self.s_encoding["bitmask"] or {}).keys())
            feat = unencoded_cols + categorical_cols + bitmask_cols
            df = df[feat]

        else:
            # Trim the dataset to the config'd list.
            feat = self.features["static"]
            if isinstance(feat, list) and (len(feat) == 0):
                self.s_scale = None
                return None
            df = df[feat] if feat else df

            # Remove numerical columns with zero variance or NaN values
            nan_cols = list(df.columns[df.isna().any()])
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            zero_var_cols = list(df[numeric_cols].columns[df[numeric_cols].std(ddof=0) == 0])
            cols_to_drop = list(set(zero_var_cols + nan_cols))
            if cols_to_drop:
                print(
                    f"Dropping numerical attributes with 0 variance or NaN values: {cols_to_drop}"
                )
                df.drop(columns=cols_to_drop, inplace=True)

        # Update the static feature list, excluding 'basin' which becomes a coordinate.
        self.features["static"] = [col for col in df.columns if col != "basin"]

        # Encode and scale the data.
        ds = df.to_xarray()

        ds, self.s_encoding = self._encode_data(ds, "static", self.s_encoding)
        x_s, self.s_scale = self._normalize_data(ds, "static", self.s_encoding, self.s_scale)

        # Reindex the map ensure alignment with the subbasins in the final dataset.
        if "subbasin" in x_s.coords:
            aligned_basins = subbasin_to_basin_map.reindex(x_s.subbasin.values)
            x_s = x_s.assign_coords(basin=("subbasin", aligned_basins.values))

        print("Done!")
        return x_s

    def _create_sparse_graph_from_nx(self) -> Array:
        """
        Converts a networkx graph into a sparse edge index and edge features.
        """
        G = self.nx_graph

        # build edge index
        node_to_int_index = {node: i for i, node in enumerate(G.nodes())}
        source_nodes, dest_nodes = [], []

        for source_id, dest_id in G.edges:
            source_nodes.append(node_to_int_index[source_id])
            dest_nodes.append(node_to_int_index[dest_id])
        edge_index = np.array([source_nodes, dest_nodes], dtype=np.int32)

        return edge_index

    def __getitem__(self, idx: int) -> dict:
        """Generate one batch of data."""
        basin, end_date = self.sample_list[idx]
        start_date = end_date - pd.Timedelta(days=self.cfg.sequence_length - 1)

        # get the basin xarray and slice by time
        date_slice = slice(start_date, end_date)
        ds = self.basin_x_ds[basin].sel(date=date_slice)

        # NaN padding
        # Identify all dynamic and target columns that are required for this sample.
        all_dynamic_cols = [col for cols in self.features["dynamic"].values() for col in cols]
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
        for source, columns in self.features["dynamic"].items():
            source_cols = [c for c in columns if c in ds.data_vars]
            source_arr = (
                ds[source_cols]
                .to_array(dim="variable")
                .transpose("date", "subbasin", "variable")
                .values
            )
            dynamic[source] = source_arr

        static = None
        if self.x_s is not None and self.features["static"]:
            static_ds = self.x_s.where(self.x_s.basin == basin, drop=True)
            static_arr = (
                static_ds[self.features["static"]]
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

    def _apply_normalization(self, data: np.ndarray, scale_info: dict) -> np.ndarray:
        """Helper to apply normalization to a numpy array."""
        if scale_info["encoded"]:
            return data

        offset = scale_info["offset"]
        scale = scale_info["scale"]

        if scale_info["log_norm"]:
            return np.log(data + self.log_pad) - offset
        else:
            # Handle zero variance
            if scale == 0:
                return data - offset
            else:
                return (data - offset) / scale

    def _encode_data(self, ds: xr.Dataset, feat_group: str, encoding: dict):
        assert feat_group in ["dynamic", "static"]

        columns_in = ds.data_vars
        encoding_copy = copy.deepcopy(encoding)
        ds, categorical_enc = self._one_hot_encoding(
            ds, feat_group, encoding_copy.get("categorical", {})
        )
        ds, bitmask_enc = self._bitmask_expansion(ds, feat_group, encoding_copy.get("bitmask", {}))

        new_columns = set(ds.data_vars) - set(columns_in)

        updated_encoding = {
            "categorical": categorical_enc,
            "bitmask": bitmask_enc,
            "encoded_columns": list(new_columns),
        }

        return ds, updated_encoding

    def _one_hot_encoding(self, ds: xr.Dataset, feat_group: str, cat_enc: dict[str, list[str]]):
        for col, prescribed in cat_enc.items():
            if col not in ds.data_vars:
                continue

            # Convert to a lazy Dask DataFrame instead of a Pandas DataFrame
            ddf = ds[[col]].to_dask_dataframe(set_index=True).categorize(columns=[col])

            # Perform one-hot encoding lazily using Dask
            encoded_ddf = dd.get_dummies(ddf, prefix=col, columns=[col])

            if prescribed:
                # Add missing prescribed columns lazily
                for c in prescribed:
                    if c not in encoded_ddf.columns:
                        encoded_ddf[c] = 0
                # Ensure order and presence of prescribed columns
                encoded_ddf = encoded_ddf[prescribed]
            else:
                if feat_group == "static":
                    # Discovery for static features requires computing the columns.
                    # This is a one-time, metadata-only operation and should be acceptable.
                    discovered_cols = encoded_ddf.columns.tolist()
                    cat_enc[col] = discovered_cols
                else:  # feat_group == "dynamic"
                    raise ValueError(
                        f"Dynamic categorical variable '{col}' must have its categories "
                        "prescribed in the config. Discovery is not supported for dynamic data."
                    )

            # Convert the lazy Dask DataFrame back to a lazy xarray.Dataset
            encoded_ds = xr.Dataset.from_dataframe(encoded_ddf)

            ds = ds.drop_vars(col)  # Drop the original column
            ds = xr.merge([ds, encoded_ds])  # Merge the lazy encoded columns

            # Update the feature dicts (logic remains the same)
            encoded_columns = list(encoded_ds.data_vars)
            if feat_group == "dynamic":
                for feats in self.features["dynamic"].values():
                    if col in feats:
                        feats.remove(col)
                        feats.extend(encoded_columns)
            else:  # static
                if col in self.features["static"]:
                    self.features["static"].remove(col)
                self.features["static"].extend(encoded_columns)

        return ds, cat_enc

    def _bitmask_expansion(
        self, ds: xr.Dataset, feat_group: str, bitmask_enc: dict[str, list[int]]
    ):
        for col, prescribed_bits in bitmask_enc.items():
            if col not in ds.data_vars:
                continue

            # Operate directly on the lazy xarray.DataArray (which is a Dask array)
            x = ds[col]
            finite_mask = da.isfinite(x.data)
            x_int = da.where(finite_mask, x.data, 0).astype(int)

            if not prescribed_bits:
                if feat_group == "static":
                    # Discovery requires computing max value. This is a one-time cost.
                    max_val = x_int.max().compute()
                    nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0

                    # Check which bits are active across the dataset
                    # This part still requires computation to discover bits
                    active_bits = []
                    for n in range(nbits):
                        if ((x_int // 2**n) % 2)[finite_mask].sum().compute() > 0:
                            active_bits.append(n)
                    prescribed_bits = active_bits
                    bitmask_enc[col] = prescribed_bits
                else:  # feat_group == 'dynamic'
                    raise ValueError(
                        f"Dynamic bitmask variable '{col}' must have its bits "
                        "prescribed in the config. Discovery is not supported for dynamic data."
                    )

            new_vars = {}
            if prescribed_bits:
                for n in prescribed_bits:
                    # All these operations are lazy Dask operations
                    bit_arr_data = ((x_int // 2**n) % 2).astype(float)
                    bit_arr_data = da.where(finite_mask, bit_arr_data, np.nan)

                    new_var_name = f"{col}_bit_{n}"
                    new_vars[new_var_name] = xr.DataArray(
                        bit_arr_data, dims=ds[col].dims, coords=ds[col].coords
                    )
                ds = ds.drop_vars(col)

            if new_vars:
                # Merge the new lazy variables
                ds = xr.merge([ds, xr.Dataset(new_vars)], compat="no_conflicts")

                # Update the feature dicts (logic remains the same)
                if feat_group == "dynamic":
                    for source, feats in self.features["dynamic"].items():
                        if col in feats:
                            feats.remove(col)
                            feats.extend(new_vars.keys())
                else:  # static
                    if col in self.features["static"]:
                        self.features["static"].remove(col)
                    self.features["static"].extend(new_vars.keys())

        return ds, bitmask_enc

    def _get_scale_from_precomp_stats(self, encoding: dict):
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

        for basin_id in self.basins:
            try:
                basin_path = self.cfg.zarr_dir / basin_id
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

            if var in self.cfg.log_norm_cols:
                scale[var]["log_norm"] = True
                scale[var]["offset"] = stats["log_sum"] / total_count

            elif var in self.cfg.range_norm_cols:
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

    def _calculate_scale_from_data(self, ds: xr.Dataset, encoding: dict):
        """
        Calculate normalization directly from dataset values.
        """
        scale = {
            k: {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}
            for k in ds.data_vars
        }

        for var in ds.data_vars:
            if var in encoding["encoded_columns"]:
                scale[var]["encoded"] = True
                continue

            if var in self.cfg.log_norm_cols:
                scale[var]["log_norm"] = True
                x = ds[var] + self.log_pad
                scale[var]["offset"] = np.nanmean(np.log(x))

            elif var in self.cfg.range_norm_cols:
                min_val = float(ds[var].min())
                max_val = float(ds[var].max())
                scale[var]["offset"] = min_val
                scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0

            else:  # z-score
                mean = float(ds[var].mean())
                std = float(ds[var].std())
                scale[var]["offset"] = mean
                scale[var]["scale"] = std if std > 1e-9 else 1.0

        return scale

    def _normalize_data(self, ds: xr.Dataset, feat_group: str, encoding: dict, scale=None):
        """
        Normalize the input data using log normalization for specified variables and standard normalization for others.

        Returns:
            ds: the input xarray dataset after normalization
            scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
        """
        assert feat_group in ["dynamic", "static"]

        if scale is None:
            if feat_group == "dynamic" and self.cfg.precomp_scaling:
                scale = self._get_scale_from_precomp_stats(encoding)
            else:
                scale = self._calculate_scale_from_data(ds, encoding)

        for var in set(ds.data_vars).intersection(scale.keys()):
            scl = scale[var]
            if scl["encoded"]:
                continue
            elif scl["log_norm"]:
                ds[var] = np.log(ds[var] + self.log_pad) - scl["offset"]
            else:
                if scl["scale"] == 0:
                    ds[var] = ds[var] - scl["offset"]
                else:
                    ds[var] = (ds[var] - scl["offset"]) / scl["scale"]

        return ds, scale

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

    def _basin_date_batching(self):
        # First get the dates that can build a complete sequence
        seq_len = np.timedelta64(self.cfg.sequence_length, "D")
        min_train_date = np.datetime64(self.time_slice.start) + seq_len

        # For each basin, we will need to identify valid dates based on valid observations.
        def valid_target(ds):
            not_nan_arr = ~np.isnan(ds[self.targets_to_index]).to_array()
            return not_nan_arr.any(dim=["subbasin", "variable"]).values

        # Loop through the basins and get a list of dates
        basin_date_map = {}
        for basin in tqdm(self.basins_to_index, disable=self.cfg.quiet, desc="Updating Indices"):
            basin_ds = self.basin_x_ds[basin]

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
        self.sample_list = []
        # Keep track of which indices in the master list belong to which basins for the sampler
        self.basin_index_map = {}
        index = 0
        for basin, dates in basin_date_map.items():
            self.basin_index_map[basin] = []
            for date in dates:
                self.sample_list.append((basin, date))
                self.basin_index_map[basin].append(index)
                index += 1

    def update_indices(self, basin_subset: list[str] | str = None):
        # Set the basin subset.
        if basin_subset is None:
            # If none, will default to the basins defined by the data subset.
            self.basins_to_index = self.basins
        else:
            # Cast as list if needed (sometimes we use a single basin).
            if isinstance(basin_subset, str):
                basin_subset = [basin_subset]
            self.basins_to_index = basin_subset

        # Get a list of target variables to check when indexing valid training targets.
        exclude_target = self.cfg.exclude_target_from_index
        if exclude_target is None:
            self.targets_to_index = self.target
        else:
            # Filter the tuples if the target name matches.
            # Does not support identical target names across different sources.
            self.targets_to_index = [
                target for target in self.target if target not in exclude_target
            ]

        self._basin_date_batching()

    def load_dynamic_in_memory(self):
        for basin, ds in tqdm(
            self.basin_x_ds.items(), disable=self.cfg.quiet, desc="Loading dynamic data into memory"
        ):
            self.basin_x_ds[basin] = ds.compute()

    @classmethod
    def get_training_stats(cls, cfg: Config) -> dict:
        """
        Calculates normalization and encoding statistics from the training data subset
        in a memory-efficient way.

        Args:
            cfg: The configuration object pointing to the training data.

        Returns:
            A dictionary containing the calculated 's_encoding', 's_scale',
            'd_encoding', and 'd_scale'.
        """
        print("Calculating training statistics for encoding and normalization...")

        # Force lazy loading for this operation to ensure low memory usage
        temp_cfg = copy.deepcopy(cfg)
        temp_cfg.in_memory = False

        # Create a 'lightweight' instance. This initializes the dataset, calculates
        # all stats via _load_attributes and _open_zarr_store, but does not
        # build the sampler indices, saving time and memory.
        temp_train_ds = cls(temp_cfg, DataSubset.train, for_stats=True)

        return temp_train_ds.stats_dict()

    def _get_dummy_dynamic_encoding(self):
        encoding = copy.deepcopy(self.d_encoding)
        encoded_columns = []
        # Handle categorical columns
        if encoding.get("categorical"):
            for col, categories in encoding["categorical"].items():
                if not categories:
                    raise ValueError(
                        f"Dynamic categorical variable '{col}' must either be excluded or have "
                        "its categories prescribed in the config."
                    )
                encoded_columns.extend([f"{col}_{cat}" for cat in categories])
        # Handle bitmask columns
        if encoding.get("bitmask"):
            for col, bits in encoding["bitmask"].items():
                if not bits:
                    raise ValueError(
                        f"Dynamic bitmask variable '{col}' must either be excluded or have "
                        "its bits prescribed in the config."
                    )
                encoded_columns.extend([f"{col}_bit_{bit}" for bit in bits])
        encoding["encoded_columns"] = encoded_columns
        return encoding

    def stats_dict(self):
        stats = {
            "s_encoding": self.s_encoding,
            "s_scale": self.s_scale,
            "d_encoding": self.d_encoding,
            "d_scale": self.d_scale,
        }
        return stats
