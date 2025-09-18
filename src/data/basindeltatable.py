import os
import shutil
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
    def __init__(self, root_dir: str, overwrite: bool = False):
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

        if self.root.is_dir() and overwrite:
            print(f"Overwriting. Deleting directory: {self.root}")
            shutil.rmtree(self.root)

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
    def upsert_dynamic(self, basin: str, subbasin: str, source: str, data: pd.DataFrame | None):
        """
        Updates/inserts dynamic time series data into the Delta table.

        This method uses a merge operation to prevent duplicate records. It inserts
        new rows and updates existing rows if a match is found based on the
        unique combination of subbasin, date, and source.
        """
        if data is None or data.empty:
            self._mark_processed(basin, subbasin, source, False)
            return

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must be indexed by a pandas.DatetimeIndex.")

        # Prepare the source DataFrame with necessary columns
        df = data.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df["date"] = df.index
        df["year"] = df["date"].dt.year
        df["basin"] = basin
        df["subbasin"] = subbasin
        df["source"] = source

        partition_cols = ["basin", "year", "source"]
        try:
            # 1. Attempt to load the table and merge
            dt = DeltaTable(self.table_uri)
            source_table = pa.Table.from_pandas(df.reset_index(drop=True))

            predicate = "t.subbasin = s.subbasin AND t.date = s.date AND t.source = s.source"

            (
                dt.merge(
                    source=source_table,
                    predicate=predicate,
                    source_alias="s",
                    target_alias="t",
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )

        except TableNotFoundError:
            # 2. If the table doesn't exist, create it with the first write
            print(f"Table not found at {self.table_uri}. Creating new Delta table.")
            write_deltalake(
                self.table_uri,
                df.reset_index(drop=True),
                mode="append",  # 'append' is fine here since the table is new
                partition_by=partition_cols,
            )
        self._mark_processed(basin, subbasin, source, True)

    def read_dynamic(
        self,
        basin: str | None = None,
        subbasins: list[str] | None = None,
        sources: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Read dynamic data from the Delta table with filters."""
        try:
            dt = DeltaTable(self.table_uri)
        except Exception:  # TableNotFoundError is not yet in the public API
            print("Dynamic data table not found. Returning empty DataFrame.")
            return pd.DataFrame()

        # Build filters for efficient predicate pushdown
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
        if df.empty:
            return df

        return df.set_index("date").sort_index()

    # -------------------------------
    # Processing metadata table
    # -------------------------------
    def _mark_processed(self, basin: str, subbasin: str, source: str, has_data: bool = True):
        """
        Mark a basin/subbasin/source combination as processed.

        Parameters
        ----------
        basin : str
            Basin identifier
        subbasin : str
            Subbasin identifier
        source : str
            Data source identifier
        has_data : bool
            Whether this combination actually contains data
        """
        metadata_row = pd.DataFrame(
            [
                {
                    "basin": basin,
                    "subbasin": subbasin,
                    "source": source,
                    "processed_at": pd.Timestamp.now(tz="UTC"),
                    "has_data": has_data,
                }
            ]
        )

        try:
            # Try to upsert into existing metadata table
            dt = DeltaTable(self.metadata_uri)
            source_table = pa.Table.from_pandas(metadata_row)

            predicate = "t.basin = s.basin AND t.subbasin = s.subbasin AND t.source = s.source"

            (
                dt.merge(
                    source=source_table,
                    predicate=predicate,
                    source_alias="s",
                    target_alias="t",
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )
        except TableNotFoundError:
            # Create new metadata table
            write_deltalake(
                self.metadata_uri, metadata_row, mode="append", partition_by=["basin", "source"]
            )

    def is_processed(self, basin: str, subbasin: str, source: str) -> bool:
        """
        Check if a basin/subbasin/source combination has been processed.

        Returns
        -------
        bool
            True if already processed, False otherwise
        """
        try:
            dt = DeltaTable(self.metadata_uri)
            df = dt.to_pandas(
                filters=[
                    ("basin", "=", basin),
                    ("subbasin", "=", subbasin),
                    ("source", "=", source),
                ]
            )
            return len(df) > 0
        except TableNotFoundError:
            return False

    def get_processing_status(self, basin: str = None, source: str = None) -> pd.DataFrame:
        """
        Get processing status for basin/source combinations.

        Parameters
        ----------
        basin : str, optional
            Filter by basin
        source : str, optional
            Filter by source

        Returns
        -------
        pd.DataFrame
            Processing status information
        """
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
    def compact(self):
        """
        Compact small files within partitions into larger ones.

        This is the solution to the "small file problem" and should be run
        periodically to optimize read performance.
        """

        def _compact(uri, name):
            dt = DeltaTable(uri)
            print(f"Compacting {name}...")
            stats = dt.optimize.compact()

            n_removed = stats["numFilesAdded"]
            n_added = stats["numFilesRemoved"]
            print(
                f"\tReduced file count by: {n_added - n_removed} (files added: {n_added}, files removed: {n_removed})."
            )
            return dt

        _compact(self.metadata_uri, "processing metadata")
        dt = _compact(self.table_uri, "dynamic_data")
        # z_order optimization co-locates related data in the same files,
        # dramatically speeding up queries that filter by the specified columns.
        dt.optimize.z_order(["subbasin", "date"])

        self.vacuum()

    def vacuum(self, retention_hours: int | None = None):
        """
        Physically delete files that are no longer referenced by the table.

        This is a destructive action. It should be run after compaction to clean up
        storage. By default, Delta Lake has a retention period (usually 7 days)
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
