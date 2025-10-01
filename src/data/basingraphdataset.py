import warnings
import json
from typing import NamedTuple
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import xarray as xr
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

    def __init__(
        self,
        cfg: Config,
        subset: DataSubset,
        *,
        train_ds: "BasinGraphDataset" = None,
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
            self.s_encoding = self.cfg.static_encoding.model_dump()
            self.d_encoding = self.cfg.dynamic_encoding.model_dump()
            self.s_scale = self.d_scale = None

        self.features = self.cfg.features.model_dump()  # dump to dict
        self.target = [v for v_lists in self.features["target"].values() for v in v_lists]

        self.time_slice = self._get_time_slice()
        self.basins = self._read_basin_files()
        self.graphs = self._load_basin_graphs()
        self.x_s = self._load_attributes()
        self.basin_x_ds = self._open_zarr_store()
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

    def _open_zarr_store(self) -> xr.Dataset:
        """Opens all basin-specific Zarr groups and stores them as lazy datasets."""
        print("Opening dynamic data...")
        warnings.filterwarnings("ignore", category=ZarrUserWarning)

        dynamic_sources = set(self.features["dynamic"].keys())
        target_sources = set(self.features["target"].keys())
        self.sources = dynamic_sources.union(target_sources)

        dynamic_columns = set([vv for v in self.features["dynamic"].values() for vv in v])
        target_columns = set([vv for v in self.features["target"].values() for vv in v])
        columns = dynamic_columns.union(target_columns)

        if self.cfg.in_memory:
            print(f"Loading full dynamic dataset into memory ({self.cfg.in_memory=}).")
        else:
            print("Lazily loading each basin's dynamic data.")

        basin_datasets = {}
        self.basin_subbasin_map = {}
        for basin_id in tqdm(self.basins, disable=self.cfg.quiet, desc="Basins"):
            basin_paths = []
            for source in self.sources:
                path = self.cfg.zarr_dir / source / basin_id
                if path.exists():
                    basin_paths.append(str(path))

            ds = xr.open_mfdataset(
                basin_paths,
                engine="zarr",
                combine="by_coords",
                data_vars=list(columns),
                join="outer",
                chunks={"date": 365},
            )
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
        ds, categorical_enc = self._one_hot_encoding(ds, feat_group, encoding["categorical"])
        ds, bitmask_enc = self._bitmask_expansion(ds, feat_group, encoding["bitmask"])

        new_columns = set(ds.data_vars) - set(columns_in)

        updated_encoding = {
            "categorical": categorical_enc,
            "bitmask": bitmask_enc,
            "encoded_columns": list(new_columns),
        }

        return ds, updated_encoding

    def _one_hot_encoding(self, ds: xr.Dataset, feat_group: str, cat_enc: dict[str, list[str]]):
        is_train = self.data_subset == "train"

        for col, prescribed in cat_enc.items():
            if col not in ds.data_vars:
                # During inference we add zero-filled columns even if the column is missing.
                if not is_train and prescribed:
                    zero_df = pd.DataFrame(
                        0, index=ds.indexes["subbasin"], columns=[f"{col}_{c}" for c in prescribed]
                    )
                    ds = xr.merge([ds, zero_df.to_xarray()])
                continue

            df = ds[col].to_dataframe()
            encoded = pd.get_dummies(df[col].astype(str), prefix=col)

            ds = ds.drop_vars(col)

            if is_train and not prescribed:
                # discover categories
                cat_enc[col] = encoded.columns.tolist()
            else:
                # enforce prescribed ordering
                for c in prescribed:
                    if c not in encoded.columns:
                        encoded[c] = 0
                encoded = encoded[prescribed]

            ds = xr.merge([ds, encoded.to_xarray()])

            # update features dict
            if feat_group == "dynamic":
                for source, feats in self.features["dynamic"].items():
                    if col in feats:
                        feats.remove(col)
                        feats.extend(encoded.columns)
            else:
                if col in self.features["static"]:
                    self.features["static"].remove(col)
                self.features["static"].extend(encoded.columns)

        return ds, cat_enc

    def _bitmask_expansion(
        self, ds: xr.Dataset, feat_group: str, bitmask_enc: dict[str, list[int]]
    ):
        is_train = self.data_subset == "train"

        for col, prescribed_bits in bitmask_enc.items():
            new_vars = {}

            if col in ds.data_vars:
                x = ds[col].values
                finite_mask = np.isfinite(x)
                x_int = np.where(finite_mask, x, 0).astype(int)

                if is_train and not prescribed_bits:
                    # infer used bits
                    max_val = x_int.max()
                    nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
                    prescribed_bits = [
                        n for n in range(nbits) if ((x_int // 2**n) % 2)[finite_mask].sum() > 0
                    ]
                    bitmask_enc[col] = prescribed_bits

                for n in prescribed_bits:
                    bit_arr = ((x_int // 2**n) % 2).astype(float)
                    bit_arr[~finite_mask] = np.nan
                    new_vars[f"{col}_bit_{n}"] = xr.DataArray(
                        bit_arr, dims=ds[col].dims, coords=ds[col].coords
                    )

                ds = ds.drop_vars(col)

            else:
                # col missing
                if not is_train and prescribed_bits:
                    if feat_group == "dynamic":
                        dims = ("subbasin", "date")
                        coords = {"subbasin": ds.subbasin, "date": ds.date}
                        shape = (len(ds.subbasin), len(ds.date))
                    else:
                        dims = ("subbasin",)
                        coords = {"subbasin": ds.subbasin}
                        shape = (len(ds.subbasin),)
                    for n in prescribed_bits:
                        new_vars[f"{col}_bit_{n}"] = xr.DataArray(
                            np.zeros(shape), dims=dims, coords=coords
                        )

            if new_vars:
                ds = xr.merge([ds, xr.Dataset(new_vars)], compat="no_conflicts")

                # update features
                if feat_group == "dynamic":
                    for source, feats in self.features["dynamic"].items():
                        if col in feats:
                            feats.remove(col)
                            feats.extend(new_vars.keys())
                else:
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
            for source in self.sources:
                basin_path = self.cfg.zarr_dir / source / str(basin_id)
                try:
                    z_group = zarr.open(str(basin_path), mode="r")
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
            if feat_group == "dynamic":
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
            else:
                target_mask = True

            basin_date_map[basin] = basin_ds["date"][target_mask].values

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
