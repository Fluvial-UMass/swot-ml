# ruff: noqa: E402
import warnings
from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning

warnings.filterwarnings("ignore", category=ZarrUserWarning)
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

from pathlib import Path

import zarr
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from dask.distributed import Client, LocalCluster


class ZarrBasinStore:
    def __init__(self, store_path, start_date="1980-01-01", end_date="2025-12-31"):
        self.store_path = Path(store_path)
        self.date_range = pd.date_range(start_date, end_date, freq="D").tz_localize(None)

        if not self.store_path.is_dir():
            store_path.mkdir()

    def write_batch_data(self, basin_id, batch_df, all_basin_subs, batch_subs, init_vars=True):
        """
        all_basin_subs: The full list of all subbasins in the Zarr store (for indexing).
        batch_subs: The specific contiguous list of subbasins we are writing NOW.
        """
        basin_path = self.store_path / basin_id

        self.init_basin_coords(basin_path, all_basin_subs)
        if init_vars:
            self.init_missing_vars(basin_path, all_basin_subs, batch_df)

        ds_write = batch_df.reset_index().set_index(["date", "subbasin"]).to_xarray()

        # Strip timezone information to match the naive format of self.date_range
        # Otherwise the reindexing below will silently remove all data.
        if ds_write.indexes["date"].tz is not None:
            ds_write.coords["date"] = ds_write.indexes["date"].tz_localize(None)

        # reindex to pad date and time
        ds_write = ds_write.reindex(subbasin=batch_subs)
        ds_write = ds_write.reindex(date=self.date_range)

        # Transpose to match Zarr layout (date, subbasin)
        ds_write = ds_write.transpose("date", "subbasin")

        # Calculate subbasin indices
        start_idx = all_basin_subs.index(batch_subs[0])
        end_idx = all_basin_subs.index(batch_subs[-1])

        # Sanity check: Ensure the slice length matches our dataset length
        expected_len = end_idx - start_idx + 1
        if len(ds_write.subbasin) != expected_len:
            raise ValueError(
                f"Batch continuity error. Expected {expected_len} items, got {len(ds_write.subbasin)}"
            )

        region = {"date": slice(0, len(self.date_range)), "subbasin": slice(start_idx, end_idx + 1)}

        # --- Write ---
        ds_write.drop_vars(["date", "subbasin"]).to_zarr(
            basin_path,
            region=region,
            mode="r+",
        )

    def write_subbasin_data(self, basin_id, subbasin, all_basin_subs, data_df):
        """
        Writes a single subbasin df into the zarr store
        """
        basin_path = self.store_path / basin_id

        self.init_basin_coords(basin_path, all_basin_subs)
        self.init_missing_vars(basin_path, all_basin_subs, data_df)

        # 3. Prepare Data for Region Write
        # Convert DataFrame to Xarray
        ds_write = data_df.to_xarray()

        # CRITICAL: Expand dims to make it 2D (date, subbasin) to match Zarr shape
        # We select [subbasin] list to ensure the coordinate is preserved as a list/array
        ds_write = ds_write.expand_dims(subbasin=[subbasin])

        # Ensure dimensions are in the correct order: (date, subbasin)
        ds_write = ds_write.transpose("date", "subbasin")

        # Reindex time to ensure alignment with the store's master clock
        ds_write = ds_write.reindex(date=self.date_range, fill_value=np.nan)

        # 4. Calculate Integer Region
        # We must find where this specific subbasin lives in the global list
        try:
            subbasin_idx = all_basin_subs.index(subbasin)
        except ValueError:
            raise ValueError(f"Subbasin {subbasin} not found in the provided all_basin_subs.")

        # Define the slice for the region write
        # region = {"dim_name": slice(start_index, end_index)}
        region = {
            "date": slice(0, len(self.date_range)),  # Write all dates
            "subbasin": slice(subbasin_idx, subbasin_idx + 1),  # Write specific subbasin index
        }

        # 5. Write to Region
        # We drop the coordinates because we are writing into a known region
        # and do not want to attempt to rewrite coordinate variables
        ds_write.drop_vars(["date", "subbasin"]).to_zarr(
            basin_path,
            region=region,
            mode="r+",
        )

        return ds_write

    def compute_and_store_stats(
        self,
        basins: str | list[str] = None,
        var_names: str | list[str] = None,
        overwrite: bool = False,
        n_workers: int = 8,
        memory_limit: str = 'auto' # or '8GB' for example

    ):
        """
        Iterates through all basin Zarr groups to compute and store normalization statistics.
        This is a standalone script to be run once after data is exported to Zarr.
        """
        basins = validate_list_or_str(basins)
        var_names = validate_list_or_str(var_names)
        basin_paths = [p for p in self.store_path.iterdir() if p.is_dir()]

        if basins:
            basin_paths = [p for p in basin_paths if p.stem in basins]

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit
        )
        with Client(cluster) as client:
            print(f"Dask client started. Dashboard at: {client.dashboard_link}")

            for basin_path in tqdm(basin_paths, desc="Basins"):
                ds = None  # initialize ds for the 'finally' block
                try:
                    # Catch errors for non-Zarr directories (e.g., _cache).
                    try:
                        z_group = zarr.open(str(basin_path), mode="r")
                        existing_stats = z_group.attrs.get("normalization_stats", {})
                    except Exception:
                        # Silently skip directories that are not valid Zarr groups
                        continue

                    # If specific vars requested, check overlap BEFORE opening xarray
                    # This avoids metadata reads for fully completed basins.
                    if var_names and not overwrite:
                        missing_vars = [v for v in var_names if v not in existing_stats]
                        if not missing_vars:
                            continue

                    ds = xr.open_zarr(basin_path, consolidated=True)
                    all_numeric_vars = [v for v in ds.data_vars if ds[v].dtype.kind in "fi"]
                    target_vars = var_names if var_names else all_numeric_vars

                    # Determine which variables need stats computed
                    if overwrite:
                        # process all target variables
                        vars_to_scale = target_vars
                    else:
                        # process only target variables that don't have existing stats
                        vars_to_scale = [v for v in target_vars if v not in existing_stats]

                    # Go to next basin if no variables to process
                    if not vars_to_scale:
                        continue
                    
                    all_computations = {}
                    for var_name in vars_to_scale:
                        # Cast and mask once
                        da = ds[var_name].astype(np.float64)
                        # Define masks
                        is_valid = da.notnull()
                        pos_mask = da > 0
                        da_pos = da.where(pos_mask)
                        log_da_pos = np.log1p(da_pos)

                        # Aggregate moments into the computation graph
                        all_computations[var_name] = {
                            "count": is_valid.sum(),
                            "sum": da.sum(),
                            "sum_sq": (da**2).sum(),
                            "min": da.min(),
                            "max": da.max(),
                            "log_sum": log_da_pos.sum(),
                            "log_sum_sq": (log_da_pos**2).sum(),
                            "positive_count": pos_mask.sum(),
                        }

                    # Single compute call for the entire basin
                    # This maximizes parallelization across all variables and all chunks
                    results = client.compute(all_computations)
                    processed_results = client.gather(results)


                    # Convert NumPy scalars to Python primitives for Zarr metadata compatibility
                    serializable_stats = {
                        var_name: {stat_key: val.item() for stat_key, val in metrics.items()}
                        for var_name, metrics in processed_results.items()
                        if metrics["count"] > 0
                    }

                    if serializable_stats:
                        existing_stats.update(serializable_stats)
                        z_group_write = zarr.open(str(basin_path), mode="r+")
                        z_group_write.attrs["normalization_stats"] = existing_stats

                finally:
                    # Ensure the xarray dataset is closed
                    if ds is not None:
                        ds.close()

    def init_basin_coords(self, basin_path, subbasin_list):
        """Initializes basin zarr with coordinates if needed."""
        if basin_path.is_dir():
            return

        coords = {"date": self.date_range, "subbasin": subbasin_list}
        # Create the template dataset (lazy)
        template_ds = xr.Dataset(coords=coords)

        # Define chunks: Time is chunked, subbasin is one chunk (or -1 for all)
        # Note: Adjust chunking strategy based on typical read patterns
        template_ds = template_ds.chunk({"date": 365, "subbasin": -1})

        # Compute nothing, just write metadata and coordinate arrays
        template_ds.to_zarr(basin_path, mode="w", compute=False, consolidated=True)

    def init_missing_vars(self, basin_path, subbasin_list, data_df):
        """If any variables are missing, we need to initiailize them for all subbasins with dummy data."""

        with xr.open_dataset(basin_path, engine="zarr", consolidated=True) as ds_ondisk:
            existing_vars = set(ds_ondisk.data_vars).union(set(ds_ondisk.coords))
            # Capture existing attributes (normalization data)
            existing_attrs = ds_ondisk.attrs.copy()

        new_vars = [v for v in data_df.columns if v not in existing_vars]

        if new_vars:
            # Create dummy variables filled with NaN for the new columns
            chunks = (365, len(subbasin_list))
            dummy_shape = (len(self.date_range), len(subbasin_list))

            new_vars_dict = {}
            for var_name in new_vars:
                # Create a lazy dask array filled with NaNs
                darr = da.full(dummy_shape, np.nan, chunks=chunks)
                new_vars_dict[var_name] = (["date", "subbasin"], darr)

            dummy_ds = xr.Dataset(
                data_vars=new_vars_dict,
                coords={"date": self.date_range, "subbasin": subbasin_list},
                attrs=existing_attrs,  # Re-attach normalization data
            )

            # Append new variables. Dask handles the streaming write.
            dummy_ds.to_zarr(basin_path, mode="a", consolidated=True)


def validate_list_or_str(arg: list[str] | str | None) -> list[str]:
    if arg is None:
        return []
    elif isinstance(arg, str):
        return [arg]
    elif isinstance(arg, list):
        return arg
    else:
        raise TypeError(f"Argument must be list, str, or None. Received {type(arg)=} ({arg=}).")
