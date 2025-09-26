import warnings

import numpy as np
import xarray as xr

from config import Config


def apply_filters(cfg: Config, features: dict, ds: xr.Dataset):
    """
    Apply filters specified in the configuration to the dataset.
    """
    for fspec in cfg.value_filters:
        column = fspec.column
        operation = fspec.operation
        value = fspec.value

        if fspec.feature_list:
            col_list = fspec.feature_list
        elif fspec.feature_group:
            col_list = features["dynamic"][fspec.feature_group]
        else:
            col_list = ds.data_vars

        if column not in ds.data_vars:
            raise RuntimeError(
                f"Column '{column}' not found in dataset. This column is required for filtering."
            )

        missing_cols = [col for col in col_list if col not in ds.data_vars]
        if missing_cols:
            warnings.warn(
                f"{missing_cols} specified by the filter are missing from the dataset. Skipping these columns."
            )

        # Operations are inverted here to match ds.where args.
        if operation == "less_than":
            mask = ds[column] > value
        elif operation == "greater_than":
            mask = ds[column] < value
        elif operation == "equals":
            mask = ds[column] != value
        else:
            raise ValueError(f"Unsupported operation '{operation}' in filter spec.")
        ds[col_list] = ds[col_list].where(mask, np.nan)

    return ds
