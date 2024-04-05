import pandas as pd
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt
import xarray as xr
import os
from pathlib import Path

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
                 log_norm_cols: list = [],
                 range_norm_cols: list = []):

        # Validate the feature dict
        if not isinstance(features_dict.get('daily'), list):
            raise ValueError("features_dict must contain a list of daily features at a minimum.")

        self.data_dir = data_dir
        self.basins = basins
        self.target = target
        self.time_slice = time_slice
        self.split_time = split_time
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.log_norm_cols = log_norm_cols
        self.range_norm_cols = range_norm_cols
            
        self.all_features =  [value for sublist in features_dict.values() for value in sublist] # Unpack all features
        self.daily_features = features_dict['daily']
        self.discharge_idx = features_dict['daily'].index(discharge_col)
        self.irregular_features = features_dict.get('irregular') # is None if key doesn't exist. 

        # Create an xarray dataset with 'basin' and 'time' dimensions
        # time_range = pd.date_range(start=self.time_slice.start, end=self.time_slice.stop, freq='D')
        # self.ds = xr.Dataset(coords={'basin': basins, 'date': time_range}) 
        
        self.train_ids = {}
        self.test_ids = {}

        self.x_s = self._load_attributes()
        self.x_d = self._load_basin_data()
        self.scale = self._normalize_data()

    def __len__(self):
        if self.train:
            return (self.n_train_samples + self.batch_size - 1) // self.batch_size
        else: 
            return (self.n_test_samples + self.batch_size - 1) // self.batch_size
    
    def _load_basin_data(self):
        self.n_train_samples = 0
        self.n_test_samples = 0
        ds_list = []
        for basin in self.basins:
            file_path = f"{self.data_dir}/time_series/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.time_slice)
    
            # Filter to keep only the necessary features and the target variable
            ds = ds[[*self.all_features, self.target]]
    
            # Create dict of train and test dates
            train_mask = (ds['date'] < self.split_time) & (~np.isnan(ds[self.target]))
            train_dates = ds['date'].where(train_mask, drop=True).values
            self.train_ids[basin] = train_dates
            self.n_train_samples += len(train_dates)
    
            test_mask = ds['date'] >= self.split_time
            test_dates = ds['date'].where(test_mask, drop=True).values
            self.test_ids[basin] = test_dates
            self.n_test_samples += len(test_dates)
    
            ds = ds.assign_coords({'basin': basin})
            ds_list.append(ds)
    
        return xr.concat(ds_list, dim="basin")

    def _load_attributes(self):
        file_path = f"{self.data_dir}/attributes/attributes.csv"
        attributes_df = pd.read_csv(file_path, index_col="index")
        x_s = {}
        for basin in self.basins:
            if basin not in attributes_df.index:
                raise KeyError(f"Basin '{basin}' not found in attributes DataFrame.")
            x_s[basin] = attributes_df.loc[basin].values
        return x_s
 
    
    def __iter__(self):
        # Create a list of (basin, index) pairs
        if self.train:
            basin_index_pairs = [(basin, date) for basin, dates in self.train_ids.items() for date in dates]
            np.random.shuffle(basin_index_pairs)
        else:
            basin_index_pairs = [(basin, date) for basin, dates in self.test_ids.items() for date in dates]

        def init_batch_dict():
            batch = {'x_dd': [],
                     'x_di': [],
                     'y': [],
                     'x_s': []}
            return batch

        def stack_batch_dict(batch):
            for key, value in batch.items():
                batch[key] = jnp.stack(value)
            return batch

        basins = []
        dates = []
        batch = init_batch_dict()
        for basin, date in basin_index_pairs:
            basins.append(basin)
            dates.append(date)
            sequence_dates = pd.date_range(end=date, periods=self.sequence_length, freq='D')
            batch['x_dd'].append(self.x_d.sel(basin=basin, date=sequence_dates)[self.daily_features].to_array().values.T)
            batch['x_di'].append(self.x_d.sel(basin=basin, date=sequence_dates)[self.irregular_features].to_array().values.T)
            batch['y'].append(self.x_d.sel(basin=basin, date=sequence_dates)[self.target].values)
            batch['x_s'].append(self.x_s[basin])
            
            if len(batch['x_dd']) == self.batch_size:
                yield (basins, dates, stack_batch_dict(batch))
                batch = init_batch_dict()
                basins = []
                dates = []
                    
        #Yield the final partial batch 
        if len(batch['x_dd'])>0:
            yield (basins, dates, stack_batch_dict(batch))

    def _normalize_data(self):
        """
        Normalize the input data using log normalization for specified variables and standard normalization for others.
        Updates the self.x_d dataset in place with the normalized data.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
        """
        # Initialize
        scale = {k: {'offset': 0, 'scale': 1, 'log_norm': False} for k in self.x_d.data_vars}
    
        # Subset the dataset to the training time period
        training_ds = self.x_d.sel(date=slice(None, self.split_time))
    
        # Iterate over each variable in the dataset
        for var in self.x_d.data_vars:
            if var in self.log_norm_cols:
                # Perform log normalization
                scale[var]['log_norm'] = True
                self.x_d[var] = np.log(self.x_d[var] + 0.001)
            elif var in self.range_norm_cols:
                # Perform min-max scaling
                scale[var]['scale'] = training_ds[var].max().values.item()
                self.x_d[var] = self.x_d[var] / scale[var]['scale']
            else:
                # Perform standard normalization
                scale[var]['offset'] = training_ds[var].mean().values.item()
                scale[var]['scale'] = training_ds[var].std().values.item()
                self.x_d[var] = (self.x_d[var] - scale[var]['offset']) / scale[var]['scale']
                
        return scale



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


 






