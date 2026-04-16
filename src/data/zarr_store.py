# ruff: noqa: E402
import warnings
from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning

warnings.filterwarnings("ignore", category=ZarrUserWarning)
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

from pathlib import Path

import zarr
import dask
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from dask.distributed import Client, LocalCluster, as_completed


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
        memory_limit: str = "8GB",
    ):
        basins = validate_list_or_str(basins)
        var_names = validate_list_or_str(var_names)
        basin_paths = [p for p in self.store_path.iterdir() if p.is_dir()]

        if basins:
            basin_paths = [p for p in basin_paths if p.stem in basins]

        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=2, memory_limit=memory_limit)

        with Client(cluster) as client:
            print(f"Dask Dashboard: {client.dashboard_link}")

            @dask.delayed
            def get_basin_task(path, vars_to_compute):
                """Returns a nested dictionary of lazy Dask computations."""
                ds = xr.open_zarr(path, consolidated=True, chunks={})

                # Intersection of requested and available variables
                available_vars = [v for v in vars_to_compute if v in ds.data_vars]

                computations = {}
                for v in available_vars:
                    da = ds[v].astype(np.float64)
                    # Filter valid and positive values
                    valid = da.where(da.notnull())
                    pos = valid.where(valid >= 0)

                    computations[v] = {
                        "count": valid.count(),
                        "sum": valid.sum(),
                        "sum_sq": (valid**2).sum(),
                        "min": valid.min(),
                        "max": valid.max(),
                        "log_sum": np.log1p(pos).sum(),
                        "log_sum_sq": (np.log1p(pos) ** 2).sum(),
                        "positive_count": pos.count(),
                    }
                return computations

            # Step 1: Submit tasks
            futures = []
            path_map = {}

            for p in basin_paths:
                try:
                    z_group = zarr.open(str(p), mode="r")
                    existing = z_group.attrs.get("normalization_stats", {})
                except Exception:
                    continue

                if var_names and not overwrite:
                    vars_to_scale = [v for v in var_names if v not in existing]
                else:
                    with xr.open_zarr(p, consolidated=True) as temp_ds:
                        all_vars = [v for v in temp_ds.data_vars if temp_ds[v].dtype.kind in "fi"]
                        vars_to_scale = var_names if var_names else all_vars
                    if not overwrite:
                        vars_to_scale = [v for v in vars_to_scale if v not in existing]

                if vars_to_scale:
                    future = client.compute(get_basin_task(p, vars_to_scale))
                    futures.append(future)
                    path_map[future.key] = (p, existing)

            # Step 2: Monitor and Eagerly Compute Nested Results
            with tqdm(total=len(futures), desc="Processing Basins") as pbar:
                for future in as_completed(futures):
                    path, existing_stats = path_map[future.key]
                    try:
                        # 'raw_dict' is a dict of dicts of lazy Dask Arrays
                        raw_dict = future.result()

                        # Compute the entire nested dictionary into NumPy scalars in one go
                        basin_results = client.gather(client.compute(raw_dict))

                        # Now 'val' is a NumPy scalar, so .item() works perfectly
                        serializable = {
                            v: {k: val.item() for k, val in metrics.items()}
                            for v, metrics in basin_results.items()
                            if metrics["count"] > 0
                        }

                        if serializable:
                            existing_stats.update(serializable)
                            z_group_write = zarr.open(str(path), mode="r+")
                            z_group_write.attrs["normalization_stats"] = existing_stats

                    except Exception as e:
                        print(f"Error in {path.stem}: {e}")
                        raise e
                    finally:
                        pbar.set_postfix({"basin": path.stem})
                        pbar.update(1)

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
