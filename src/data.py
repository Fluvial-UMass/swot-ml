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
                 train: bool = True,
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
        self.attn_dt = {}
        self.decay_dt = {}
        self.x_s = {}
        self.y = {}
        self.scale = {}
        
        self._load_data()

    def __len__(self):
        if self.train:
            return (self.n_train_samples + self.batch_size - 1) // self.batch_size
        else: 
            return (self.n_test_samples + self.batch_size - 1) // self.batch_size
    
    def _load_basin_data(self, basin):
        file_path = f"{self.data_dir}/time_series/{basin}.nc"
        
        ds = xr.open_dataset(file_path).sel(date=self.time_slice)
        self.split_idx[basin] = np.where(ds['date']==self.split_time)[0][0]
        df = ds.to_dataframe()[self.all_features+[self.target]]

        # df_norm, scale = _normalize_data
        self.x_dd[basin] = df[self.daily_features].values
        self.y[basin] = df[[self.target]].values

        if self.irregular_features is not None:
            df_di = df[self.irregular_features]
            self.x_di[basin] = df_di.values
            non_nan_ids = df_di.reset_index().dropna().index.values
            self.attn_dt[basin], self.decay_dt[basin]= _calc_attn_decay_dt(len(df_di), non_nan_ids)
        
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
                     'attn_dt': [],
                     'decay_dt': [],
                     'x_s': [],
                     'y': []}
            return batch

        def stack_batch_dict(batch):
            stacked = {}
            for key, value in batch.items():
                stacked[key] = jnp.stack(value)
            return stacked

        batch = init_batch_dict()
        for basin in self.basins:
            for idx in idx_list[basin]:
                sequence_ids = jnp.array(jnp.arange(idx - self.sequence_length + 1, idx + 1))
                batch['x_dd'].append(self.x_dd[basin][sequence_ids])
                batch['x_di'].append(self.x_di[basin][sequence_ids])
                batch['attn_dt'].append(self.attn_dt[basin][sequence_ids])
                batch['decay_dt'].append(self.decay_dt[basin][sequence_ids])
                batch['x_s'].append(self.x_s[basin])
                batch['y'].append(self.y[basin][sequence_ids])
                
                if len(batch['x_dd']) == self.batch_size:
                    yield stack_batch_dict(batch)
                    batch = init_batch_dict()
                    
        #Yield the final partial batch 
        if len(batch['x_dd'])>0:
            yield stack_batch_dict(batch)

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

def _calc_attn_decay_dt(data_len, non_nan_ids):
    # Initialize attention and decay dts.
    attn = np.zeros(data_len)
    decay = np.full(data_len, np.nan)
    if len(non_nan_ids) == 0:
        return attn, decay

    # Special treatment when there are leading missing values.
    # Infinite forces the attn mechanism to 0 for the irregular data.
    # 1 causes no decay for first obs. 
    if non_nan_ids[0] > 0:
        attn[:non_nan_ids[0]] = np.inf
        decay[non_nan_ids[0]] = 1
    # Loop through the non-NaN indices and fill in the result array
    for i in range(len(non_nan_ids) - 1):
        start = non_nan_ids[i]
        end = non_nan_ids[i + 1]
        attn[start + 1:end] = np.arange(1, end - start)
        decay[end] = end-start
        
    return attn, decay

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


 






