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
                 features_dict: dict,
                 target: str,
                 time_slice: slice,
                 split_time: np.datetime64,
                 batch_size: int,
                 sequence_length: int,
                 train: bool,
                 discharge_col: str = None,
                 zero_min_cols: list = None):
        
        self.data_dir = data_dir
        self.basins = basins
        self.target = target
        self.time_slice = time_slice
        self.split_time = split_time
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.zero_min_cols = zero_min_cols

        self.all_features =  [value for sublist in features_dict.values() for value in sublist]
        self.daily_features = features_dict['daily']
        self.discharge_idx = features_dict['daily'].index(discharge_col)
        if len(features_dict)>1:
            self.irregular_features = features_dict['irregular']

        self.train_ids = {}
        self.test_ids = {}
        self.split_idx = {}
        self.x_dd = {}
        self.x_di = {}
        self.x_di_ids = {}
        self.x_di_attn_dt = {}
        self.x_s = {}
        self.y = {}
        self.scale = {}
        
        self._load_data()

    def __len__(self):
        if self.train:
            return (self.n_train_samples + self.batch_size - 1) // self.batch_size
        return
    
    def _load_basin_data(self, basin):
        file_path = f"{self.data_dir}/time_series/{basin}.nc"
        
        ds = xr.open_dataset(file_path).sel(date=self.time_slice)
        self.split_idx[basin] = np.where(ds['date']==self.split_time)[0][0]
        df = ds.to_dataframe()[self.all_features+[self.target]]

        # df_norm, scale = _normalize_data
        self.x_dd[basin] = df[self.daily_features].values
        self.y[basin] = df[[self.target]].values

        if self.irregular_features is not None:
            df_di = df[self.irregular_features].dropna()
            self.x_di[basin] = df_di.values
            self.x_di_ids[basin] = df_di.reset_index().index.values
            self.x_di_attn_dt[basin] = self._calc_attn_dt(basin)
        
        train_ids = np.where(~np.isnan(self.y[basin][:self.split_idx[basin]]))[0]
        self.train_ids[basin] = train_ids[train_ids >= self.sequence_length]
        self.n_train_samples += len(self.train_ids[basin])
        
        self.test_ids[basin] = np.arange(self.split_idx[basin]+self.sequence_length,len(df))
        self.n_test_samples += len(self.test_ids[basin])

        ds.close()
    
    def _load_attributes(self):
        file_path = f"{self.data_dir}/attributes/attributes.csv"
        return pd.read_csv(file_path, index_col="index")
    
    def _load_data(self):
        attributes_df = self._load_attributes()

        self.n_train_samples = 0
        self.n_test_samples = 0
        for basin in self.basins:
            self._load_basin_data(basin)
            
            if basin not in attributes_df.index:
                raise KeyError(f"Basin '{basin}' not found in attributes DataFrame.")
            self.x_s[basin] = attributes_df.loc[basin].values
    
    def __iter__(self):
        if self.train:
            np.random.shuffle(self.basins)
            idx_list = self.train_ids
        else:
            idx_list = self.test_ids

        def init_batch_dict():
            batch = {'x_dd': [],
                     'x_di': [],
                     'x_di_attn_dt': [],
                     'x_di_ids': [],
                     'x_di_decay_dt': [],
                     'x_s': [],
                     'y': []}
            return batch

        def stack_batch(batch):
            stacked = {}
            for key, value in batch.items():
                stacked[key] = jnp.stack(value)
            return stacked

        batch = init_batch_dict()
        for basin in self.basins:
            for idx in idx_list[basin]:
                seq_daily_ids = jnp.array(jnp.arange(idx - self.sequence_length + 1, idx + 1))
                batch['x_dd'].append(self.x_dd[basin][seq_daily_ids])
                batch['x_di_attn_dt'].append(self.x_di_attn_dt[basin][seq_daily_ids])
                 
                irregular_ids_mask = jnp.isin(self.x_di_ids[basin], seq_daily_ids)
                batch['x_di'].append(self.x_di[basin][irregular_ids_mask])
                x_di_ids = self.x_di_ids[basin][irregular_ids_mask]
                batch['x_di_ids'].append(x_di_ids)
                batch['x_di_decay_dt'].append(np.diff(x_di_ids, prepend=1))

                batch['x_s'].append(self.x_s[basin])
                batch['y'].append(self.y[basin][seq_daily_ids])
                
                if len(batch['x_dd']) == self.batch_size:
                    yield stack_batch(batch)
                    batch = init_batch_dict()
                    
        #Yield the final partial batch 
        if len(xd)>0:
            yield stack_batch(batch)

    def _normalize_data(self):
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
        scale['offset'] = {}
        scale['scale'] = {}
    
        normalized_df = df.copy()
        for col in df.columns:
            if col in self.zero_min_columns:
                scale['offset'][col] = 0
                scale['scale'][col] = df[col].max()
            else:
                scale['offset'][col] = df[col].mean()
                scale['scale'][col] = df[col].std()
                    
            normalized_df[col] = (df[col] - scale['offset'][col]) / scale['scale'][col]
                    
        return

    def _calc_attn_dt(self, basin):
        # Initialize with 0s and leading Infs before first observation.
        # Infinite forces the attn mechanism to 0 for the irregular data
        attn_dt = np.zeros(len(self.x_dd[basin]), dtype=float)
        if self.x_di_ids[basin][0] > 0:
            attn_dt[:self.x_di_ids[basin][0]] = np.inf
        
        # Loop through the non-NaN indices and fill in the result array
        for i in range(len(self.x_di_ids[basin]) - 1):
            start = self.x_di_ids[basin][i]
            end = self.x_di_ids[basin][i + 1]
            attn_dt[start + 1:end] = np.arange(1, end - start)

        return attn_dt


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


 






