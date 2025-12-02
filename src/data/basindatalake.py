# ruff: noqa: E402
import warnings
import time
import random
from functools import wraps

from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning

warnings.filterwarnings("ignore", category=ZarrUserWarning)
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Suppress 'warn' and 'info' messages from the Rust backend.
# deltalake is receiving deprecation warnings about future updates to DataFusion.
# https://github.com/delta-io/delta-rs/issues/3709
# Will probably go away with updates, but we are locking versions anyway.
# Has to be set before deltalakes is imported
os.environ["RUST_LOG"] = "warn,datafusion_datasource_parquet=error"

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
import zarr
from dask.distributed import Client
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError
from tqdm import tqdm

LOG_PAD = 1e-6  # IMPORTANT: This must match the log_pad value the dataloader class


def validate_list_or_str(arg: list[str] | str | None) -> list[str]:
    if arg is None:
        return []
    elif isinstance(arg, str):
        return [arg]
    elif isinstance(arg, list):
        return arg
    else:
        raise TypeError(f"Argument must be list, str, or None. Received {type(arg)=} ({arg=}).")
    
def retry_on_concurrent_write(func=None, *, max_retries=15, base_delay=0.5):
    """
    Decorator for retrying on concurrent write conflicts.
    Can be used as @retry_on_concurrent_write or @retry_on_concurrent_write(max_retries=20)
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    if "concurrent transaction" in error_msg or "conflicting concurrent" in error_msg:
                        if attempt < max_retries - 1:
                            delay = base_delay * (1.5 ** attempt) + random.uniform(0, 2)
                            delay = min(delay, 30)
                            
                            print(f"Concurrent write conflict (attempt {attempt + 1}/{max_retries}). "
                                  f"Retrying in {delay:.2f}s...")
                            time.sleep(delay)
                            continue
                    raise
            
            raise Exception(f"Failed after {max_retries} retries. Last error: {last_exception}")
        
        return wrapper
    
    if func is None:
        # Called with arguments: @retry_on_concurrent_write(max_retries=20)
        return decorator
    else:
        # Called without arguments: @retry_on_concurrent_write
        return decorator(func)


class BasinDataLake:
    def __init__(self, root_dir: str):
        """
        Initialize or open a hydrological dataset using Delta Lake with separate tables per source.

        Parameters
        ----------
        root_dir : str
            The root directory where the dataset will be stored.
        """
        self.root = Path(root_dir)
        self.dynamic_data_root = self.root / "dynamic_data"
        self.metadata_uri = str(self.root / "processing_metadata")
        self.static_file = self.root / "static.parquet"

        self.root.mkdir(parents=True, exist_ok=True)
        self.dynamic_data_root.mkdir(parents=True, exist_ok=True)

    def _get_source_table_uri(self, source: str) -> str:
        """Get the table URI for a specific source."""
        return str(self.dynamic_data_root / source)

    def _discover_sources(self) -> list[str]:
        """Discover all sources that have been written to the data lake."""
        if not self.dynamic_data_root.exists():
            return []

        sources = []
        for source_dir in self.dynamic_data_root.iterdir():
            if source_dir.is_dir():
                try:
                    # Check if it's a valid Delta table
                    DeltaTable(str(source_dir))
                    sources.append(source_dir.name)
                except TableNotFoundError:
                    # Skip directories that aren't Delta tables
                    continue
        return sources

    # -------------------------------
    # Static attributes (vanilla parquet)
    # -------------------------------
    def write_static(self, df: pd.DataFrame):
        """Write or overwrite the static attributes file."""
        if "subbasin" not in df.columns:
            raise ValueError("Static dataframe must include a 'subbasin' column.")
        df.to_parquet(self.static_file, index=False)

    def read_static(self) -> pd.DataFrame:
        """Read all static basin-level attributes."""
        if not self.static_file.exists():
            raise ValueError("Static dataframe not found in dataset.")
        return pd.read_parquet(self.static_file)

    # -------------------------------
    # Dynamic data (using separate DeltaTables per source)
    # -------------------------------
    def _prepare_dynamic_df(
        self, basin: str, source: str, data: dict[str, pd.DataFrame | None]
    ) -> pd.DataFrame | None:
        """Pre-processes input data dictionary into a single DataFrame for writing."""
        if not isinstance(data, dict):
            raise TypeError("'data' must be a dictionary mapping subbasin IDs to DataFrames.")

        dfs_to_concat = []
        for subbasin_id, sub_df in data.items():
            if sub_df is not None and not sub_df.empty:
                df_copy = sub_df.copy()
                df_copy["subbasin"] = subbasin_id
                dfs_to_concat.append(df_copy)

        if not dfs_to_concat:
            return None

        df = pd.concat(dfs_to_concat)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must be indexed by a pandas.DatetimeIndex.")
        df.index = (
            df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        )

        df.index.name = "date"
        df.reset_index(inplace=True)
        df["year"] = df["date"].dt.year
        df["basin"] = basin
        return df

    @retry_on_concurrent_write
    def write_dynamic(
        self,
        basin: str,
        source: str,
        data: dict[str, pd.DataFrame | None],
        mode: str = "upsert",
    ):
        """
        Writes dynamic time series data to source-specific table using either 'upsert' or 'append' mode.

        Parameters
        ----------
        basin : str
            The basin identifier.
        source : str
            The data source identifier.
        data : dict[str, Optional[pd.DataFrame]]
            Dictionary mapping subbasin IDs to DataFrames.
        mode : str, optional
            Write mode:
            - 'upsert': (default) performs a merge based on 'subbasin' and 'date'.
            - 'append': performs a fast append without checking for duplicates.
            - 'overwrite': atomically replaces all data for the specified 'basin'.
        """
        if mode not in ["upsert", "append", "overwrite"]:
            raise ValueError("Mode must be either 'upsert', 'append', or 'overwrite'.")

        df = self._prepare_dynamic_df(basin, source, data)

        # If df is None, only update metadata and exit.
        if df is None:
            if mode == "append":
                self._append_processing_record(basin, source, data)
            else:
                self._upsert_processing_record(basin, source, data)
            return

        table_uri = self._get_source_table_uri(source)
        partition_cols = ["basin", "year"]

        try:
            if mode == "append":
                write_deltalake(
                    table_uri,
                    df,
                    mode="append",
                    partition_by=partition_cols,
                    schema_mode="merge",
                )
                self._upsert_processing_record(basin, source, data)

            elif mode == "overwrite":
                partition_filters = [("basin", "=", basin)]
                write_deltalake(
                    table_uri,
                    df,
                    mode="overwrite",
                    partition_by=partition_cols,
                    partition_filters=partition_filters,
                    schema_mode="merge",
                )
                # Use upsert to update the metadata to reflect the new state
                self._upsert_processing_record(basin, source, data)

            else:  # mode == "upsert"
                try:
                    dt = DeltaTable(table_uri)
                    for _, df_group in df.groupby(partition_cols):
                        source_table = pa.Table.from_pandas(df_group)
                        predicate = "t.subbasin = s.subbasin AND t.date = s.date"
                        dt.merge(
                            source=source_table,
                            predicate=predicate,
                            source_alias="s",
                            target_alias="t",
                        ).when_matched_update_all().when_not_matched_insert_all().execute()
                    self._upsert_processing_record(basin, source, data)
                except TableNotFoundError:
                    print(
                        f"Table not found at {table_uri}. Creating new Delta table for source '{source}'."
                    )
                    write_deltalake(table_uri, df, mode="append", partition_by=partition_cols)
                    self._upsert_processing_record(basin, source, data)
        except Exception as e:
            print(f"Failed to {mode} data for basin {basin}, source {source}. Error: {e}")
            raise

    def read_dynamic(
        self,
        basin: str,
        subbasin: list[str] | str = None,
        source: list[str] | str = None,
        start_date: str | pd.Timestamp = None,
        end_date: str | pd.Timestamp = None,
        concat_sources: bool = True,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Read dynamic data from one or more source tables.

        Parameters
        ----------
        basin : str
            The basin identifier to filter by.
        subbasin : list[str], optional
            List of subbasin IDs to filter by.
        sources : list[str], optional
            List of source identifiers. If None, reads from all available sources.
        start_date : str, optional
            Start date for filtering (ISO format).
        end_date : str, optional
            End date for filtering (ISO format).
        concat_sources : bool, default True
            If True, concatenate data from all sources into a single DataFrame.
            If False, return a dictionary mapping source names to DataFrames.

        Returns
        -------
        Union[pd.DataFrame, dict[str, pd.DataFrame]]
            If concat_sources=True, returns concatenated DataFrame.
            If concat_sources=False, returns dict of {source: DataFrame}.
        """

        # Discover available sources if not specified
        available_sources = self._discover_sources()
        if not available_sources:
            print("No dynamic data sources found. Returning empty result.")
            return pd.DataFrame() if concat_sources else {}

        # Filter to requested sources
        if source is not None:
            source = [source] if isinstance(source, str) else source
            missing_sources = set(source) - set(available_sources)
            if missing_sources:
                print(f"Warning: Requested sources not found: {missing_sources}")
            sources_to_read = [s for s in source if s in available_sources]
        else:
            sources_to_read = available_sources

        if not sources_to_read:
            return pd.DataFrame() if concat_sources else {}

        def validate_date(date):
            if date is None:
                pass
            elif isinstance(date, str):
                date = pd.Timestamp(date, tz="UTC")
            elif isinstance(date, pd.Timestamp):
                if date.tz is None:
                    date = date.tz_localize("UTC")
            else:
                raise ValueError(f"Dates must be either string or timestamp, received {date=}")
            return date

        start_date = validate_date(start_date)
        end_date = validate_date(end_date)

        def read_source(source: str) -> pd.DataFrame:
            """Read data from a single source table."""
            table_uri = self._get_source_table_uri(source)
            try:
                dt = DeltaTable(table_uri)

                filters = []
                filters.append(("basin", "=", basin))
                if subbasin:
                    if isinstance(subbasin, str):
                        filters.append(("subbasin", "=", subbasin))
                    elif isinstance(subbasin, list):
                        filters.append(("subbasin", "in", subbasin))
                if start_date:
                    filters.append(("date", ">=", start_date))
                if end_date:
                    filters.append(("date", "<=", end_date))

                df = dt.to_pandas(filters=filters if filters else None)
                df.drop(columns=["basin", "year"], inplace=True)

                df.set_index(["subbasin", "date"], inplace=True)
                return df

            except TableNotFoundError:
                print(f"Table not found for source '{source}'. Skipping.")
                return pd.DataFrame()

        source_data = {s: read_source(s) for s in sources_to_read}
        if not concat_sources:
            return source_data

        # Concatenate all sources
        if not source_data:
            return pd.DataFrame()

        # Build multi-index: [source, original column]
        result = pd.concat(source_data.values(), keys=source_data.keys(), axis=1)

        # Optionally give the levels names
        result.columns.set_names(["source", "variable"], inplace=True)

        # Drop the source column
        # result = result.drop(columns=["source"], errors="ignore")

        return result  # .sort_index()

    # -------------------------------
    # Processing metadata table
    # -------------------------------
    def _prepare_metadata_df(
        self, basin: str, source: str, data: dict[str, pd.DataFrame | None]
    ) -> pd.DataFrame | None:
        """Creates a DataFrame of metadata records from the input data dict."""
        subbasins = list(data.keys())
        if not subbasins:
            return None

        has_data_map = {sb: (df is not None and not df.empty) for sb, df in data.items()}
        now = pd.Timestamp.now(tz="UTC")
        metadata_rows = [
            {
                "basin": basin,
                "subbasin": sb,
                "source": source,
                "processed_at": now,
                "has_data": has_data_map.get(sb, False),
            }
            for sb in subbasins
        ]
        return pd.DataFrame(metadata_rows)

    def _append_processing_record(self, basin: str, source: str, data: dict):
        """Naively appends records to the processing metadata table."""
        metadata_df = self._prepare_metadata_df(basin, source, data)
        if metadata_df is None:
            return

        write_deltalake(
            self.metadata_uri, metadata_df, mode="append", partition_by=["basin", "source"]
        )

    def _upsert_processing_record(self, basin: str, source: str, data: dict):
        """Merges records into the processing metadata table."""
        metadata_df = self._prepare_metadata_df(basin, source, data)
        if metadata_df is None:
            return

        try:
            dt = DeltaTable(self.metadata_uri)
            source_table = pa.Table.from_pandas(metadata_df)
            predicate = "t.basin = s.basin AND t.subbasin = s.subbasin AND t.source = s.source"
            dt.merge(
                source=source_table, predicate=predicate, source_alias="s", target_alias="t"
            ).when_matched_update_all().when_not_matched_insert_all().execute()
        except TableNotFoundError:
            write_deltalake(
                self.metadata_uri, metadata_df, mode="append", partition_by=["basin", "source"]
            )

    def get_processing_status(self, basin: str = None, source: str = None) -> pd.DataFrame:
        try:
            dt = DeltaTable(self.metadata_uri)
            df = dt.to_pandas()

            if source is not None:
                df = df[df["source"] == source]

            if df.empty:
                return pd.DataFrame()

            # Always normalize datetimes (avoid tz issues)
            if "processed_at" in df.columns:
                df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")
                df["processed_at"] = df["processed_at"].dt.tz_localize(None)

            # Pivot both has_data and processed_at, then concat
            pivots = {}
            if "has_data" in df.columns:
                pivots["has_data"] = df.pivot_table(
                    index=["basin", "subbasin"], columns="source", values="has_data"
                )
            if "processed_at" in df.columns:
                pivots["processed_at"] = df.pivot_table(
                    index=["basin", "subbasin"], columns="source", values="processed_at"
                )

            if not pivots:
                return pd.DataFrame()

            return pd.concat(pivots.values(), axis=1, keys=pivots.keys())

        except TableNotFoundError:
            # Metadata table doesn’t exist yet
            return pd.DataFrame()

    def list_sources(self) -> list[str]:
        """List all available data sources."""
        return self._discover_sources()

    def list_processed_basins(self, status: pd.DataFrame = None) -> list[str]:
        # Unprocessed subbasin/source combinations are left as NaN in 'has_data' field.
        def grp_not_na(grp):
            return (~grp["has_data"].isna()).all(axis=None)

        status = self.get_processing_status() if status is None else status
        is_proc = status.groupby("basin").apply(grp_not_na)
        return is_proc[is_proc].index

    def list_source_basin_data(self, status: pd.DataFrame = None) -> pd.DataFrame:
        def grp_is_true(grp):
            return (grp["has_data"] == 1).any(axis=0)

        status = self.get_processing_status() if status is None else status
        has_data = status.groupby("basin").apply(grp_is_true)
        return has_data

    def get_basin_subbasin_map(self, status: pd.DataFrame = None) -> dict:
        def get_subbasin_list(row):
            return list(row.index.get_level_values("subbasin"))

        status = self.get_processing_status() if status is None else status
        return status.groupby("basin").apply(get_subbasin_list).to_dict()

    # -------------------------------
    # Maintenance Operations
    # -------------------------------
    def optimize(self, retention_hours: int = None, sources: list[str] = None):
        """
        Compact small files within partitions into larger ones for specified sources.

        Parameters
        ----------
        sources : list[str], optional
            List of sources to optimize. If None, optimizes all sources.
        """

        def print_stats(stats, name):
            n_added = stats["numFilesAdded"]
            n_removed = stats["numFilesRemoved"]
            print(
                f"\t{name}: Reduced file count by {n_removed - n_added} ({n_added=}, {n_removed=})."
            )

        # Optimize metadata table
        dt = DeltaTable(self.metadata_uri)
        print("Compacting processing metadata...")
        stats = dt.optimize.compact()
        print_stats(stats, "Metadata")

        # Optimize source tables
        sources = validate_list_or_str(sources)
        sources_to_optimize = sources if sources else self._discover_sources()

        for source in sources_to_optimize:
            table_uri = self._get_source_table_uri(source)
            dt = DeltaTable(table_uri)
            print(f"Compacting and ordering {source} data...")
            stats = dt.optimize.z_order(["date"])
            print_stats(stats, source)

        self.vacuum(retention_hours = retention_hours, sources=sources)

    def vacuum(self, retention_hours: int = None, sources: list[str] = None):
        """
        Physically delete files that are no longer referenced by the tables.

        Parameters
        ----------
        retention_hours : int, optional
            Retention period in hours. If 0, immediately deletes (unsafe for production).
        sources : list[str], optional
            List of sources to vacuum. If None, vacuums all sources.
        """

        def _vac(uri, name):
            dt = DeltaTable(uri)
            print(f"Vacuuming {name}...")

            if retention_hours == 0:
                print("WARNING: Overriding retention period check for immediate deletion.")
                dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
            else:
                dt.vacuum(retention_hours=retention_hours, dry_run=False)

        # Vacuum metadata
        _vac(self.metadata_uri, "processing metadata")

        # Vacuum source tables
        available_sources = self._discover_sources()
        if sources is not None:
            sources_to_vacuum = [s for s in sources if s in available_sources]
        else:
            sources_to_vacuum = available_sources

        for source in sources_to_vacuum:
            table_uri = self._get_source_table_uri(source)
            _vac(table_uri, f"{source} data")

    def export_to_zarr(
        self,
        zarr_path: Path | str,
        workers: int = 1,
        basins: list[str] | None = None,
        sources: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        chunk_days: int = 365,
        overwrite: bool = False,
    ):
        zarr_path = Path(zarr_path)

        start_date = pd.Timestamp(start_date, tz="UTC") if start_date else None
        end_date = pd.Timestamp(end_date, tz="UTC") if end_date else None

        status = self.get_processing_status()
        basin_subbasin_map = self.get_basin_subbasin_map(status)
        basin_source_df = self.list_source_basin_data(status)
        basin_source_map = {
            basin: list(row.index[row]) for basin, row in basin_source_df.iterrows()
        }
        processed_basins = self.list_processed_basins(status).to_list()

        if sources is None:
            sources = self.list_sources()
        elif isinstance(sources, str):
            sources = [sources]
        sources = set(sources)

        if basins:
            # Use the basins argument if it exists
            basins_to_export = basins
        elif overwrite:
            # Get all basins with data
            basins_to_export = set(processed_basins)
        else:
            # Filter the basins based on what has already been exported.
            if zarr_path.exists():
                exported_basins = {d.name for d in zarr_path.iterdir() if d.is_dir()}
                basins_to_export = set(processed_basins) - exported_basins
            else:
                basins_to_export = set(processed_basins)

        if not basins_to_export:
            print("No new basins to process.")
            return

        count = 0
        if workers > 1:
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        self.export_basin_to_zarr,
                        basin=basin,
                        subbasins=basin_subbasin_map[basin],
                        sources=sources.intersection(basin_source_map[basin]),
                        zarr_path=zarr_path,
                        start_date=start_date,
                        end_date=end_date,
                        chunk_days=chunk_days,
                    ): basin
                    for basin in basins_to_export
                }

                with tqdm(total=len(basins_to_export), desc="Processing basins") as pbar:
                    for future in as_completed(futures):
                        basin = futures[future]
                        try:
                            written = future.result()
                            if written:
                                count += 1
                            pbar.set_postfix({"completed": count, "basin": basin})
                        except Exception as e:
                            print(f"\nError in basin {basin}: {e}")
                        finally:
                            pbar.update(1)

        else:
            for basin in tqdm(basins_to_export, desc="Iterating through basins"):
                try:
                    written = self.export_basin_to_zarr(
                        basin=basin,
                        subbasins=basin_subbasin_map[basin],
                        sources=sources,
                        zarr_path=zarr_path,
                        start_date=start_date,
                        end_date=end_date,
                        chunk_days=chunk_days,
                    )
                    count += 1 if written else 0
                except Exception as e:
                    print(f"{basin=}")
                    raise e
            print(f"Wrote data for {count} basins.")

    def export_basin_to_zarr(
        self,
        basin: str,
        subbasins: list[str],
        sources: list[str],
        zarr_path: Path | str,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
        chunk_days: int,
    ):
        """
        Export one basin's dynamic data from DeltaLake -> Zarr for fast ML training.

        Parameters
        ----------
        basin : str
            Basin ID to export.
        zarr_path : str
            Path to the zarr store.
        sources : list[str] | None
            Sources to include. If None, exports all available.
        start_date, end_date : str | None
            Optional date filters.
        chunk_days : int
            Chunk length along time dimension (e.g., 90 days).
        """
        # One of the offending basins = ABOM-213983010

        zarr_group_path = Path(zarr_path) / basin
        global_calendar = pd.date_range(start_date, end_date, freq="D").tz_localize(None)
        subbasin_idx = pd.Index(subbasins)  # Faster indexing than list of str

        # Determine the time chunks to iterate over for reading from Delta Lake
        if len(subbasins) <= 500:
            year_step = 50
        elif len(subbasins) <= 1000:
            year_step = 10
        else:
            year_step = 1
        # Create the date range for iteration
        iter_dates = pd.date_range(start_date, end_date, freq=f"{year_step}YS-JAN")
        if end_date not in iter_dates:
            iter_dates = iter_dates.append(pd.DatetimeIndex([end_date]))

        def write_coords(dates, subbasins):
            template_ds = xr.Dataset(coords={"date": dates, "subbasin": subbasins})
            template_ds = template_ds.chunk({"date": chunk_days, "subbasin": -1})

            # Write the empty template to disk. This establishes the coordinate grid and chunks
            template_ds.to_zarr(zarr_group_path, mode="w", consolidated=True)

        def init_empty_var_arrays(var_names, dates, subbasins):
            ds = xr.open_dataset(zarr_group_path, engine="zarr")
            existing_vars = list(ds.data_vars)

            for var_name in var_names:
                if var_name not in existing_vars:
                    # Create a dummy DataArray with the correct structure
                    dummy_data = xr.DataArray(
                        np.full((len(dates), len(subbasins)), np.nan),
                        coords={"date": dates, "subbasin": subbasins},
                        dims=["date", "subbasin"],
                        name=var_name,
                    )
                    # Chunk and write to the existing zarr store
                    dummy_data = dummy_data.chunk({"date": chunk_days, "subbasin": -1})
                    dummy_data.to_zarr(zarr_group_path, mode="a")

        def write_df_to_zarr(df: pd.DataFrame):
            chunk_data = {}
            for var_name in df.columns:
                pivoted_data = df[var_name].unstack(level="subbasin")
                chunk_data[var_name] = (["date", "subbasin"], pivoted_data.values)

            chunk_ds = xr.Dataset(
                data_vars=chunk_data,
                coords={
                    "date": pivoted_data.index.tz_localize(None),
                    "subbasin": pivoted_data.columns,
                },
            )

            chunk_dates = df.index.get_level_values("date").unique().tz_localize(None)
            all_chunk_dates = pd.date_range(chunk_dates.min(), chunk_dates.max(), freq="D")

            chunk_ds = chunk_ds.reindex(date=all_chunk_dates, subbasin=subbasins, fill_value=np.nan)

            # Write to the specific region
            # xarray will automatically align based on coordinate values
            chunk_ds.to_zarr(
                zarr_group_path,
                region="auto",  # automatically determines the region based on coordinates
                mode="r+",  # read-write mode (must exist)
                consolidated=False,
            )

        written = False
        for i in range(len(iter_dates) - 1):
            chunk_start, chunk_end = iter_dates[i], iter_dates[i + 1]

            df_chunk = self.read_dynamic(
                basin=basin, source=sources, start_date=chunk_start, end_date=chunk_end
            ).droplevel(level="source", axis=1)

            if df_chunk.empty:
                continue

            if not written and not zarr_group_path.is_dir():
                write_coords(global_calendar, subbasin_idx)

            init_empty_var_arrays(list(df_chunk), global_calendar, subbasin_idx)
            write_df_to_zarr(df_chunk)

        return True

    def compute_and_store_stats(
        self, zarr_root: Path, vars: str | list[str] = None, overwrite: bool = False
    ):
        """
        Iterates through all basin Zarr groups to compute and store normalization statistics.
        This is a standalone script to be run once after data is exported to Zarr.
        """
        zarr_root = Path(zarr_root)
        vars = validate_list_or_str(vars)
        basin_paths = [p for p in zarr_root.iterdir() if p.is_dir()]

        with Client() as client:
            print(f"Dask client started. Dashboard at: {client.dashboard_link}")

            for basin_path in tqdm(basin_paths, desc="Basins"):
                ds = None  # initialize ds for the 'finally' block
                try:
                    # Open with Zarr to read existing attributes without loading the dataset
                    z_group = zarr.open(str(basin_path), mode="r")
                    existing_stats = z_group.attrs.get("normalization_stats", {})

                    ds = xr.open_zarr(basin_path, consolidated=True)
                    all_numeric_vars = [v for v in ds.data_vars if ds[v].dtype.kind in "fi"]
                    target_vars = vars if vars else all_numeric_vars

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

                    stats_dict = {}
                    computations = {}
                    # Prepare all Dask computations for the necessary variables
                    for var_name in vars_to_scale:
                        variable = ds[var_name].astype(np.float64)
                        valid_values = variable.where(variable.notnull())
                        positive_values = valid_values.where(valid_values > 0)

                        # Store the delayed computation objects
                        computations[var_name] = {
                            "count": valid_values.count(),
                            "sum": valid_values.sum(),
                            "sum_sq": (valid_values**2).sum(),
                            "min": valid_values.min(),
                            "max": valid_values.max(),
                            "log_sum": np.log(positive_values + LOG_PAD).sum(),
                            "positive_count": positive_values.count(),
                        }

                    # Trigger all computations in parallel
                    results = client.compute(computations)
                    processed_results = client.gather(results)

                    # Process the computed results
                    for var_name, stats in processed_results.items():
                        count = stats["count"].item()
                        if count == 0:
                            continue

                        stats_dict[var_name] = {
                            "count": count,
                            "sum": stats["sum"].item(),
                            "sum_sq": stats["sum_sq"].item(),
                            "min": stats["min"].item(),
                            "max": stats["max"].item(),
                            "log_sum": stats["log_sum"].item()
                            if stats["positive_count"].item() > 0
                            else 0.0,
                        }

                    if stats_dict:
                        # Merge new stats with existing stats and write back
                        existing_stats.update(stats_dict)

                        # Re-open with zarr library to write attributes
                        z_group_write = zarr.open(str(basin_path), mode="r+")
                        z_group_write.attrs["normalization_stats"] = existing_stats

                finally:
                    # Ensure the xarray dataset is closed
                    if ds is not None:
                        ds.close()



