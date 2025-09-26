import hashlib
import warnings
import json
import pickle
from typing import NamedTuple

import yaml
from tqdm import tqdm
import pandas as pd
import xarray as xr
import zarr
import networkx as nx
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from jaxtyping import Array

from config import Config, DataSubset


class GraphBatch(NamedTuple):
    dynamic: dict[str, Array]
    graph_edges: Array
    static: Array = None
    y: Array = None

    def __getitem__(self, key):
        warnings.warn(
            f"Batch: dict-style access ('batch[\"{key}\"]') is deprecated. Use attribute access ('batch.{key}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)

    @classmethod
    def in_axes(cls):
        # TODO: If we start training with mixes of different basins we will need assign graph_edges to 0 as well.
        return cls(
            dynamic=0,
            static=0,
            graph_edges=0,
            y=0,
        )


class BasinGraphDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(
        self,
        cfg: Config,
        subset: DataSubset,
        *,
        train_ds: "BasinGraphDataset" = None,
        use_cache=True,
    ):
        self.cfg = cfg
        self.log_pad = 0.001
        self.data_subset = subset

        self.inference_mode = isinstance(train_ds, self.__class__)
        if self.inference_mode:
            self.s_encoding = train_ds.s_encoding
            self.s_scale = train_ds.s_scale
            self.d_encoding = train_ds.d_encoding
            self.d_scale = train_ds.d_scale
        else:
            self.s_encoding = self.s_scale = self.d_encoding = self.d_scale = None

        self.features = self.cfg.features.model_dump()  # dump to dict
        self.target = [v for v_lists in self.features["target"].values() for v in v_lists]

        self.time_slice = self._get_time_slice()
        self.basins = self._read_basin_files()
        self.graphs = self._load_basin_graphs()
        self.x_s = self._load_attributes()
        self.x_d = self._open_zarr_store(use_cache)
        self.update_indices()

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

    def _open_zarr_store(self, use_cache) -> xr.Dataset:
        """Opens all basin-specific Zarr groups and stores them as lazy datasets."""
        print("Opening dyanmic zarr store (lazy)...")

        dynamic_sources = set(self.features["dynamic"].keys())
        target_sources = set(self.features["target"].keys())
        sources = dynamic_sources.union(target_sources)

        # Unpack dict of lists into sets
        dynamic_columns = set([vv for v in self.features["dynamic"].values() for vv in v])
        target_columns = set([vv for v in self.features["target"].values() for vv in v])
        columns = dynamic_columns.union(target_columns)

        basin_datasets = []
        self.basin_subbasin_map = {}
        for basin_id in self.basins:
            basin_paths = []

            for source in sources:
                path = self.cfg.zarr_dir / source / basin_id
                if path.exists():
                    self._ensure_consolidated_metadata(path)
                    basin_paths.append(str(path))

            if not basin_paths:
                continue  # Skip if this basin has no data files

            # Open all sources for this single basin
            ds = xr.open_mfdataset(
                basin_paths,
                engine="zarr",
                combine="by_coords",
                data_vars=list(columns),
                join="outer",
            )
            ds = ds.assign_coords(basin=("subbasin", [basin_id] * len(ds.subbasin)))
            basin_datasets.append(ds)
            self.basin_subbasin_map[basin_id] = list(ds.subbasin.values)

        # Build full dataset and rechunk
        ds = xr.concat(basin_datasets, dim="subbasin", join="outer")
        basin_chunks = self.create_basin_aware_chunks(ds)
        ds = ds.chunk(basin_chunks)
        ds = ds.sel(date=self.time_slice)

        self.time_gaps = {
            source: ds[columns].to_array().isnull().any().compute().item()
            for source, columns in self.features["dynamic"].items()
        }

        x_d = self._cached_encode_dynamic(ds, use_cache)

        if self.cfg.in_memory:
            x_d = x_d.compute()

        return x_d

    def _cached_encode_dynamic(self, ds, use_cache):
        from_cache = False
        norm_config_hash = self._get_normalization_hash()
        cache_dir = self.cfg.data_root / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / (norm_config_hash + ".pkl")
        if use_cache and cache_file.is_file():
            with open(cache_file, "rb") as f:
                cache_dict = pickle.load(f)
            self.d_encoding = cache_dict.get("d_encoding")
            self.d_scale = cache_dict.get("d_scale")
            from_cache = True

        # Encode and scale the data.
        ds, self.d_encoding = self._encode_data(ds, "dynamic", self.d_encoding)
        x_d, self.d_scale = self._normalize_data(ds, "dynamic", self.d_encoding, self.d_scale)

        # Save encoding and scale dicts to pickle file, unless we just read it in above...
        if use_cache and not from_cache:
            with open(cache_file, "wb") as f:
                pickle.dump({"d_encoding": self.d_encoding, "d_scale": self.d_scale}, f)

        return x_d

    def _load_basin_graphs(self) -> dict[str, np.ndarray]:
        """Loads all basin-specific graphs."""
        print("Loading basin graphs...")

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

        return graphs

    def _load_attributes(self) -> xr.Dataset:
        """
        Loads the basin attributes from a CSV file.

        Returns:
            xr.Dataset: An xarray dataset of attribute data with basin coordinates.
        """
        print("Loading static attributes")
        df = pd.read_parquet(self.cfg.attributes_file)

        basin_mask = df.index.get_level_values("basin").isin(self.basins)
        df = df[basin_mask].droplevel("basin")

        if self.inference_mode:
            unencoded_cols = [k for k, v in self.s_scale.items() if not v["encoded"]]
            one_hot_cols = list((self.s_encoding["one_hot"] or {}).keys())
            bitmask_cols = list((self.s_encoding["bitmask"] or {}).keys())
            feat = unencoded_cols + one_hot_cols + bitmask_cols
            df = df[feat]

        else:
            # Trim the dataset to the config'd list.
            feat = self.features["static"]
            if isinstance(feat, list) and (len(feat) == 0):
                self.s_scale = None
                return None
            df = df[feat] if feat else df

            # Remove columns with zero variance or NaN values
            nan_cols = list(df.columns[df.isna().any()])
            zero_var_cols = list(df.columns[df.std(ddof=0) == 0])
            cols_to_drop = list(set(zero_var_cols + nan_cols))
            if cols_to_drop:
                print(
                    f"Dropping numerical attributes with 0 variance or NaN values: {cols_to_drop}"
                )
                df.drop(columns=cols_to_drop, inplace=True)

        # Update or set the static feature list.
        self.features["static"] = list(df.columns)

        # Encode and scale the data.
        ds = df.to_xarray()
        ds, self.s_encoding = self._encode_data(ds, "static", self.s_encoding)
        x_s, self.s_scale = self._normalize_data(ds, "static", self.s_encoding, self.s_scale)

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

        # Slice into the dynamic xarray dataset
        subbasins = self.basin_subbasin_map[basin]
        date_slice = slice(start_date, end_date)
        ds = self.x_d.sel(subbasin=subbasins, date=date_slice)

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
            static_ds = self.x_s.sel(subbasin=subbasins)
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
        one_hot_enc = encoding.get("one_hot") if encoding else None
        bitmask_enc = encoding.get("bitmask") if encoding else None

        ds, one_hot = self._one_hot_encoding(ds, feat_group, one_hot_enc)
        ds, bitmask = self._bitmask_expansion(ds, feat_group, bitmask_enc)

        new_columns = set(ds.data_vars) - set(columns_in)

        encoding = {"one_hot": one_hot, "bitmask": bitmask, "encoded_columns": list(new_columns)}

        return ds, encoding

    def _one_hot_encoding(self, ds: xr.Dataset, feat_group, onehot_enc: dict | None):
        assert feat_group in ["dynamic", "static"]

        # Apply one-hot encoding to categorical columns
        if not onehot_enc:
            # Use flattened categorical_cols from config, filter for columns in ds
            categorical_cols = [col for col in self.cfg.categorical_cols if col in ds.data_vars]
            if not categorical_cols:
                return ds, None
            onehot_enc = {col: None for col in categorical_cols}

        for col, prescribed_cols in onehot_enc.items():
            if col in ds.data_vars:
                df = ds[col].to_dataframe()
                encoded = pd.get_dummies(df[col].astype(str), prefix=col)
                # Remove the original categorical column
                ds = ds.drop_vars(col)
            else:
                # Create an empty DataFrame with the same index as ds
                encoded = pd.DataFrame(index=ds.basin)

            if prescribed_cols is not None and len(prescribed_cols) > 0:
                # Add missing categories as columns filled with zeros
                for c in prescribed_cols:
                    if c not in encoded.columns:
                        encoded[c] = 0
                # Filter out columns not in the prescribed encoding
                encoded = encoded[prescribed_cols]
            else:
                onehot_enc[col] = encoded.columns

            # Add encoded data
            ds = xr.merge([ds, encoded.to_xarray()])

            # Locate the col inside the features dict, remove and replace.
            # This is kind of ugly but deals with the 2 level feature dict.
            if feat_group == "dynamic":
                for source, source_features in self.features[feat_group].items():
                    if col in source_features:
                        self.features[feat_group][source].remove(col)
                        self.features[feat_group][source].extend(encoded.columns)
            elif feat_group == "static":
                self.features[feat_group].extend(encoded.columns)
                if col in self.features[feat_group]:
                    self.features[feat_group].remove(col)
                else:
                    print(f"{col} not found in {feat_group} features. Encoded as 0s.")

        return ds, onehot_enc

    def _bitmask_expansion(self, ds: xr.Dataset, feat_group: str, bitmask_enc: dict | None):
        assert feat_group in ["dynamic", "static"]

        if not bitmask_enc:
            # Use flattened bitmask_cols from config, filter for columns present in the dataset
            bitmask_cols = [col for col in self.cfg.bitmask_cols if col in ds.data_vars]
            if not bitmask_cols:
                return ds, None
            # Initialize encoding. The value for each column will be the list of used bit indices.
            bitmask_enc = {k: None for k in bitmask_cols}

        for col, bits_to_expand in bitmask_enc.items():
            new_vars = {}
            if col in ds.data_vars:
                # Get the bitmask integers
                original_da = ds[col]
                x = original_da.values
                finite_mask = np.isfinite(x)

                if not np.any(finite_mask):
                    print(
                        f"Warning: No finite values found in column '{col}' for bitmask expansion. "
                        "Skipping bitmask encoding for this column."
                    )
                    # During training, record that no bits were used for this column.
                    if bits_to_expand is None:
                        bitmask_enc[col] = []
                    continue

                # Temporarily set NaN to 0 for bit operations
                x_int = np.where(finite_mask, x, 0).astype(int)

                if bits_to_expand is None:  # Training mode: determine which bits to expand
                    max_val = x_int.max()
                    num_bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0

                    # Determine which bits are actually used in this dataset
                    used_bits = []
                    for n in range(num_bits):
                        bit_arr = (x_int // 2**n) % 2
                        if bit_arr[finite_mask].sum() > 0:
                            used_bits.append(n)

                    bitmask_enc[col] = used_bits  # Save the list of used bits
                    bits_to_expand = used_bits  # Use this list for the current expansion

                # Expand only the determined/prescribed bits
                for n in bits_to_expand:
                    bit_arr = (x_int // 2**n) % 2
                    bit_arr = bit_arr.astype(float)  # So we can assign np.nan
                    bit_arr[~finite_mask] = np.nan  # Restore NaNs

                    new_vars[f"{col}_bit_{n}"] = xr.DataArray(
                        data=bit_arr,
                        dims=original_da.dims,
                        coords=original_da.coords,
                    )
                # Remove the original categorical column
                ds = ds.drop_vars(col)

            else:  # Column is not in the current dataset
                if bits_to_expand is None:  # Training mode, but column is missing from data.
                    bitmask_enc[col] = []  # Record that no bits were used.
                    continue

                # Inference mode, column is missing. Create zero-filled columns for all prescribed bits.
                for n in bits_to_expand:
                    # Infer dims/coords from the dataset's coordinates
                    if feat_group == "dynamic":
                        dims = ("subbasin", "date")
                        coords = {"subbasin": ds.coords["subbasin"], "date": ds.coords["date"]}
                        shape = (len(ds.coords["subbasin"]), len(ds.coords["date"]))
                    elif feat_group == "static":
                        dims = ("subbasin",)
                        coords = {"subbasin": ds.coords["subbasin"]}
                        shape = (len(ds.coords["subbasin"]),)

                    new_vars[f"{col}_bit_{n}"] = xr.DataArray(
                        data=np.zeros(shape),
                        dims=dims,
                        coords=coords,
                    )

            if new_vars:
                ds = xr.merge([ds, xr.Dataset(new_vars)], compat="no_conflicts")

            # Update the features list with the new bit columns
            if feat_group == "dynamic":
                for source, source_features in self.features["dynamic"].items():
                    if col in source_features:
                        self.features["dynamic"][source].remove(col)
                        self.features["dynamic"][source].extend(new_vars.keys())
            elif feat_group == "static":
                if col in self.features[feat_group]:
                    self.features[feat_group].remove(col)
                    self.features[feat_group].extend(new_vars.keys())
                elif new_vars:
                    # Only print warning if we actually added columns for a missing feature
                    print(f"{col} not found in {feat_group} features. Encoded as 0s.")

        return ds, bitmask_enc

    def _normalize_data(self, ds: xr.Dataset, feat_group, encoding, scale=None):
        """
        Normalize the input data using log normalization for specified variables and standard normalization for others.

        Returns:
            ds: the input xarray dataset after normalization
            scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
        """
        assert feat_group in ["dynamic", "static"]

        if scale is None:
            # Initialize
            scale = {
                k: {
                    "encoded": False,
                    "log_norm": False,
                    "offset": 0,
                    "scale": 1,
                }
                for k in ds.data_vars
            }

            # Iterate over each variable in the dataset and calculate scaler
            for var in ds.data_vars:
                log_norm_cols = self.cfg.log_norm_cols
                range_norm_cols = self.cfg.range_norm_cols

                if var in encoding["encoded_columns"]:
                    # One-hot encoded columns don't need normalization
                    scale[var]["encoded"] = True

                elif log_norm_cols is not None and var in log_norm_cols:
                    # Log normalization
                    scale[var]["log_norm"] = True
                    x = ds[var] + self.log_pad
                    scale[var]["offset"] = np.nanmean(np.log(x))

                elif range_norm_cols is not None and var in range_norm_cols:
                    # Min-max scaling
                    min_val = ds[var].min().values.item()
                    max_val = ds[var].max().values.item()
                    scale[var]["offset"] = min_val
                    scale[var]["scale"] = max_val - min_val

                else:
                    # Standard normalization
                    scale[var]["offset"] = ds[var].mean().values.item()
                    scale[var]["scale"] = ds[var].std().values.item()

        for var in set(ds.data_vars).intersection(scale.keys()):
            scl = scale[var]
            if scl["encoded"]:
                continue
            elif scl["log_norm"]:
                ds[var] = np.log(ds[var] + self.log_pad) - scl["offset"]
            else:
                # Handle 0 variance here
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
        valid_seq_mask = self.x_d["date"] >= min_train_date
        valid_dates = self.x_d["date"][valid_seq_mask].values

        # For each basin, we will need to identify valid dates based on valid observations.
        def valid_target(ds):
            not_nan_arr = ~np.isnan(ds[self.targets_to_index]).to_array()
            return not_nan_arr.any(dim=["subbasin", "variable"]).values

        # Loop through the basins and get a list of dates
        basin_date_map = {}
        ds_valid_dates = self.x_d.sel(date=valid_dates)
        for basin in tqdm(self.basins_to_index, disable=self.cfg.quiet, desc="Updating Indices"):
            ds_basin = ds_valid_dates.where(ds_valid_dates["basin"] == basin)

            # Mask out dates without valid data if we need it.
            if self.data_subset in ["train", "test"]:
                target_mask = valid_target(ds_basin)
            else:
                target_mask = True

            basin_date_map[basin] = ds_basin["date"][target_mask].values

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

    def _get_normalization_hash(self):
        """Generates a hash based on config parameters that affect normalization."""
        cfg_keys = [
            "train_date_range",
            "log_norm_cols",
            "range_norm_cols",
        ]
        data_config = {k: getattr(self.cfg, k) for k in cfg_keys}
        data_config["dynamic_feat"] = self.features["dynamic"]
        data_config["train_basins"] = sorted(self.basins)
        dict_str = yaml.dump(data_config, sort_keys=True)
        return hashlib.sha256(dict_str.encode("utf-8")).hexdigest()

    def _ensure_consolidated_metadata(self, zarr_path):
        """Consolidate metadata if not already consolidated."""
        try:
            # Try to open with consolidated metadata
            zarr.open_consolidated(str(zarr_path))
        except ValueError:
            # Not consolidated, so consolidate it
            print(f"Consolidating metadata for {zarr_path}")
            zarr.consolidate_metadata(str(zarr_path))

    def create_basin_aware_chunks(self, ds):
        basin_coord = ds.coords["basin"].values
        basin_boundaries = [0]
        current_basin = basin_coord[0]

        for i, basin in enumerate(basin_coord[1:], 1):
            if basin != current_basin:
                basin_boundaries.append(i)
                current_basin = basin
        basin_boundaries.append(len(basin_coord))

        subbasin_chunks = []
        for i in range(len(basin_boundaries) - 1):
            chunk_size = basin_boundaries[i + 1] - basin_boundaries[i]
            subbasin_chunks.append(chunk_size)

        return {"date": self.cfg.sequence_length, "subbasin": tuple(subbasin_chunks)}
