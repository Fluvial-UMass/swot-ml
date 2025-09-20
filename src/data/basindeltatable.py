import os
from pathlib import Path

# Suppress 'warn' and 'info' messages from the Rust backend.
# deltalake is receiving a deprecation warning about future updates to DataFusion.
# https://github.com/delta-io/delta-rs/issues/3709
# Will probably go away with updates, but we are locking versions anyway.
# Has to be set before deltalakes is imported
os.environ["RUST_LOG"] = "warn,datafusion_datasource_parquet=error"

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError


class BasinDeltaTable:
    def __init__(self, root_dir: str):
        """
        Initialize or open a hydrological dataset using Delta Lake.

        Parameters
        ----------
        root_dir : str
            The root directory where the dataset will be stored.
        overwrite : bool
            If True, an existing dataset at the root_dir will be deleted.
        """
        self.root = Path(root_dir)
        self.table_uri = str(self.root / "dynamic_data")
        self.metadata_uri = str(self.root / "processing_metadata")
        self.static_file = self.root / "static.parquet"

        self.root.mkdir(parents=True, exist_ok=True)

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
    # Dynamic data (using DeltaTable)
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
        df["source"] = source
        return df

    def write_dynamic(
        self,
        basin: str,
        source: str,
        data: dict[str, pd.DataFrame | None],
        mode: str = "upsert",
    ):
        """
        Writes dynamic time series data using either 'upsert' or 'append' mode.

        Parameters
        ----------
        basin : str
            The basin identifier.
        source : str
            The data source identifier.
        data : dict[str, pd.DataFrame | None]
            Dictionary mapping subbasin IDs to DataFrames.
        mode : str, optional
            Write mode: 'upsert' (default) performs a merge, 'append' performs a
            fast append without checking for duplicates.
        """
        if mode not in ["upsert", "append"]:
            raise ValueError("Mode must be either 'upsert' or 'append'.")

        df = self._prepare_dynamic_df(basin, source, data)

        # If df is None, only update metadata and exit.
        if df is None:
            if mode == "append":
                self._append_processed_record(basin, source, data)
            else:
                self._upsert_processed_record(basin, source, data)
            return

        partition_cols = ["basin", "year", "source"]

        try:
            if mode == "append":
                write_deltalake(
                    self.table_uri,
                    df,
                    mode="append",
                    partition_by=partition_cols,
                    schema_mode="merge",
                )
                self._append_processed_record(basin, source, data)

            else:  # mode == "upsert"
                try:
                    dt = DeltaTable(self.table_uri)
                    for _, df_group in df.groupby(partition_cols):
                        source_table = pa.Table.from_pandas(df_group)
                        predicate = (
                            "t.subbasin = s.subbasin AND t.date = s.date AND t.source = s.source"
                        )
                        dt.merge(
                            source=source_table,
                            predicate=predicate,
                            source_alias="s",
                            target_alias="t",
                        ).when_matched_update_all().when_not_matched_insert_all().execute()
                    self._upsert_processed_record(basin, source, data)
                except TableNotFoundError:
                    print(f"Table not found at {self.table_uri}. Creating new Delta table.")
                    write_deltalake(self.table_uri, df, mode="append", partition_by=partition_cols)
                    self._upsert_processed_record(basin, source, data)
        except Exception as e:
            print(f"Failed to {mode} data for basin {basin}, source {source}. Error: {e}")
            raise

    def read_dynamic(
        self,
        basin: str | None = None,
        subbasins: list[str] | None = None,
        sources: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        # This method remains unchanged
        try:
            dt = DeltaTable(self.table_uri)
        except TableNotFoundError:
            print("Dynamic data table not found. Returning empty DataFrame.")
            return pd.DataFrame()

        filters = []
        if basin:
            filters.append(("basin", "=", basin))
        if subbasins:
            filters.append(("subbasin", "in", subbasins))
        if sources:
            filters.append(("source", "in", sources))
        if start_date:
            filters.append(("date", ">=", pd.Timestamp(start_date, tz="UTC")))
        if end_date:
            filters.append(("date", "<=", pd.Timestamp(end_date, tz="UTC")))

        df = dt.to_pandas(filters=filters if filters else None)
        return df if df.empty else df.set_index("date").sort_index()

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

    def _append_processed_record(self, basin: str, source: str, data: dict):
        """Naively appends records to the processing metadata table."""
        metadata_df = self._prepare_metadata_df(basin, source, data)
        if metadata_df is None:
            return

        write_deltalake(
            self.metadata_uri, metadata_df, mode="append", partition_by=["basin", "source"]
        )

    def _upsert_processed_record(self, basin: str, source: str, data: dict):
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

    def is_processed(self, basin: str, subbasin: str, source: str) -> bool:
        try:
            dt = DeltaTable(self.metadata_uri)
            df = dt.to_pandas(
                filters=[
                    ("basin", "=", basin),
                    ("subbasin", "=", subbasin),
                    ("source", "=", source),
                ]
            )
            return not df.empty
        except TableNotFoundError:
            return False

    def get_processing_status(self, basin: str = None, source: str = None) -> pd.DataFrame:
        try:
            dt = DeltaTable(self.metadata_uri)
            filters = []
            if basin:
                filters.append(("basin", "=", basin))
            if source:
                filters.append(("source", "=", source))
            return dt.to_pandas(filters=filters if filters else None)
        except TableNotFoundError:
            return pd.DataFrame(columns=["basin", "subbasin", "source", "processed_at", "has_data"])

    # -------------------------------
    # Maintenance Operations
    # -------------------------------
    def optimize(self):
        """
        Compact small files within partitions into larger ones.

        This is the solution to the "small file problem" and should be run
        periodically to optimize read performance.
        """
        def print_stats(stats):
            n_added = stats["numFilesAdded"]
            n_removed = stats["numFilesRemoved"]
            print(f"\tReduced file count by: {n_removed - n_added} ({n_added=}, {n_removed=}).")

        dt = DeltaTable(self.metadata_uri)
        print("Compacting processing metadata...")
        stats = dt.optimize.compact()
        print_stats(stats)

        # z_order optimization co-locates related data in the same files,
        # dramatically speeding up queries that filter by the specified columns.
        dt = DeltaTable(self.table_uri)
        print("Compacting and ordering dynamic data...")
        stats = dt.optimize.z_order(["subbasin", "date"])
        print_stats(stats)

        self.vacuum()

    def vacuum(self, retention_hours: int | None = None):
        """
        Physically delete files that are no longer referenced by the table.

        This is a destructive action. It should be run after compaction to clean up
        storage. By default, Delta Lake has a retention period of 7 days
        to prevent accidental deletion of data needed for time travel.
        """

        def _vac(uri, name):
            dt = DeltaTable(uri)
            print(f"Vacuuming {name}...")

            # --- IMPORTANT ---
            # Overriding the retention hours to 0 immediately deletes the unlinked files.
            # This is UNSAFE in production but useful for development.
            if retention_hours == 0:
                print("WARNING: Overriding retention period check for immediate deletion.")
                dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
            else:
                dt.vacuum(retention_hours=retention_hours, dry_run=False)

        _vac(self.metadata_uri, "processing metadata")
        _vac(self.table_uri, "dynamic data")
