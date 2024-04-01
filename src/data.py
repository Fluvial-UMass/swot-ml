import pandas as pd
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt
import xarray as xr
import os
from pathlib import Path


def get_dataloaders(train_df, test_df, **kwargs):
    train = DataLoader(train_df, train=True, shuffle=True, **kwargs)
    test = DataLoader(test_df, train=False, scale=train.scale, **kwargs)
    return train, test

import pandas as pd
import xarray as xr
import numpy as np


class DataLoader:
    def __init__(self, 
                 data_dir: Path, 
                 basins: list,
                 features: list,
                 target: str,
                 time_slice: slice,
                 split_time: np.datetime64,
                 batch_size: int,
                 sequence_length: int,
                 train: bool,
                 discharge_col: str = None,
                 scale: pd.Series = None,
                 fill_and_weight: bool = False,
                 calc_dt: bool = False):
        
        self.data_dir = data_dir
        self.basins = basins
        self.features = features
        self.target = target
        self.time_slice = time_slice
        self.split_time = split_time
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.fill_and_weight = fill_and_weight
        self.calc_dt = calc_dt

        self.train_indices = {}
        self.test_indices = {}
        self.xd = {}
        self.xs = {}
        self.dt = {}
        self.y = {}
        
        self._load_data()

    def __len__(self):
        if self.train:
            return (self.n_train_samples + self.batch_size - 1) // self.batch_size
        return
    
    def _load_basin_data(self, basin):
        file_path = f"{self.data_dir}/time_series/{basin}.nc"
        ds = xr.open_dataset(file_path).sel(date=self.time_slice)
        df = ds.to_dataframe()[self.features+[self.target]]

        self.xd[basin] = df[self.features].values
        self.y[basin] = df[[self.target]].values

        split_idx = np.where(ds['date']==self.split_time)[0][0]
        train_ids = np.where(~np.isnan(self.y[basin][:split_idx]))[0]
        self.train_indices[basin] = train_ids[train_ids >= self.sequence_length]
        self.n_train_samples += len(self.train_indices[basin])
        
        self.test_indices[basin] = np.arange(split_idx+self.sequence_length,len(df))
        self.n_test_samples += len(self.test_indices[basin])

        ds.close()
        return 
    
    def _load_attributes(self):
        file_path = f"{self.data_dir}/attributes/attributes.csv"
        return pd.read_csv(file_path, index_col="index")
    
    def _load_data(self):
        attributes_df = self._load_attributes()

        self.n_train_samples = 0
        self.n_test_samples = 0
        for basin in self.basins:
            self._load_basin_data(basin)
            
            if basin in attributes_df.index:
                self.xs[basin] = attributes_df.loc[basin].values
    
    def __iter__(self):
        if self.train:
            np.random.shuffle(self.basins)
            idx_list = self.train_indices
        else:
            idx_list = self.test_indices

        xd, xs, y = [], [], []
        for basin in self.basins:
            for idx in idx_list[basin]:
                sequence_ids =  np.array(np.arange(idx - self.sequence_length + 1, idx + 1))

                xd.append(self.xd[basin][sequence_ids])
                y.append(self.y[basin][sequence_ids])
                xs.append(self.xs[basin])

                if len(xd) == self.batch_size:
                    yield _create_batch_dict(xd,xs,y)
                    xd, xs, y = [], [], []
                    
        #Yield the final partial batch 
        if len(xd)>0:
            yield _create_batch_dict(xd,xs,y)


def _create_batch_dict(xd, xs, y):
    batch_dict = {'xd': np.stack(xd),
                  'y': np.stack(y),
                  'xs': np.stack(xs)}
    return batch_dict

def _combine_slices(slice_list):
        start_list = [s.start for s in slice_list]
        if None in start_list: start = None
        else: start = min(start_list)
            
        stop_list = [s.stop for s in slice_list]
        if None in stop_list: stop = None
        else: stop = min(stop_list)

        return slice(start, stop)

def _distance_to_closest_obs(arr):
    nan_mask = np.isnan(arr)
    distances = np.full(arr.shape, np.inf) # All nan columns will be inf. Weights -> 0
    
    for feature_idx in range(arr.shape[1]):
        # Calculate the distance transform for the column
        distances[:,feature_idx] = distance_transform_edt(nan_mask[:,feature_idx])
    return distances+1
    
def _fill_nan_obs(arr):
    for feature_idx in range(arr.shape[1]):
        is_nan = jnp.isnan(arr[:,feature_idx])
        if np.sum(~is_nan)==0:
            # Can't interpolate without observations, but distances will be inf and weights 0.
            arr[:,feature_idx][is_nan] = 0
            continue
        #Interpolate based on closest measurement. 
        arr[is_nan,feature_idx] = np.interp(np.flatnonzero(is_nan),
                                            np.flatnonzero(~is_nan),
                                            arr[~is_nan,feature_idx])
    return arr


def _normalize_data(df, scale=None, min_max_columns=None):
    """
    Normalize the input data using the provided scale or calculate the scale if not provided.
    Allows for min-max normalization of specified columns.

    Args:
        df (pd.DataFrame): The input data to be normalized.
        scale (Dict[str, float], optional): A dictionary containing the 'offset' and 'scale'
            for each column. For standard normalization, 'offset' is the mean and 'scale' is the standard deviation.
            For min-max normalization, 'offset' is the minimum and 'scale' is the range (max - min).
            If not provided, these values will be calculated from the data.
        min_max_columns (List[str], optional): A list of column names to be normalized using min-max normalization.
            Other columns will be normalized using standard normalization.

    Returns:
        pd.DataFrame: The normalized data.
        Dict[str, float]: A dictionary containing the 'offset' and 'scale' for each column.
    """
    if scale is None:
        scale = {}
        scale['offset'] = {}
        scale['scale'] = {}

    normalized_df = df.copy()

    if min_max_columns is not None:
        for col in min_max_columns:
            scale['offset'][col] = 0#df[col].min()
            scale['scale'][col] = df[col].max() - df[col].min()
            normalized_df[col] = (df[col] - scale['offset'][col]) / scale['scale'][col]

    # Standard normalization for other columns
    std_cols = [col for col in df.columns if col not in min_max_columns] if min_max_columns is not None else df.columns
    for col in std_cols:
        if col not in scale['offset']:
            scale['offset'][col] = df[col].mean()
            scale['scale'][col] = df[col].std()
        normalized_df[col] = (df[col] - scale['offset'][col]) / scale['scale'][col]

    return normalized_df, scale   