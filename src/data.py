import os
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jutil
import jax.sharding as jshard
from torch.utils.data import Dataset, DataLoader


from concurrent.futures import ProcessPoolExecutor, as_completed

class TAPDataLoader(DataLoader):
    def __init__(self, cfg, dataset):    
        num_workers = cfg.get('num_workers', 1)
        persistent_workers = False if num_workers==0 else cfg.get('persistent_workers',True)
        
        super().__init__(dataset,
                         collate_fn = self.collate_fn,
                         shuffle=cfg.get('shuffle', True),
                         batch_size=cfg.get('batch_size', 1),
                         num_workers=num_workers,
                         pin_memory=cfg.get('pin_memory', True),
                         drop_last=cfg.get('drop_last', False),
                         persistent_workers=persistent_workers)     
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        # Dataset index params
        self.data_subset = cfg.get('data_subset','train')
        self.basin_subset = cfg.get('basin_subset',None)
        self.set_mode()

        # Data sharding params
        backend = cfg.get('backend', None)
        num_devices = cfg.get('num_devices', None)
        self.set_jax_sharding(backend, num_devices)
        
    @staticmethod
    def collate_fn(sample):
        # I can't figure out how to just not collate. Can't even use lambdas because of multiprocessing.
        return sample

    def set_mode(self, *, data_subset:str=None, basin_subset:list=None):
        # Use existing values if None provided.
        if data_subset is not None:
            self.data_subset = data_subset    
        if basin_subset is not None:
            self.basin_subset = basin_subset

        self.dataset.update_indices(data_subset=self.data_subset,
                                    basin_subset=self.basin_subset)
    
    def set_jax_sharding(self, backend=None, num_devices=None):
        """
        Updates the jax device sharding of data. 
    
        Args:
        backend (str): XLA backend to use (cpu, gpu, or tpu). If None is passed, select GPU if available.
        num_devices (int): Number of devices to use. If None is passed, use all available devices for 'backend'.
        """
        available_devices = _get_available_devices()
        # Default use GPU if available
        if backend is None:
            backend = 'gpu' if 'gpu' in available_devices.keys() else 'cpu'
        else:
            backend = backend.lower()
        
        # Validate the requested number of devices. 
        if num_devices is None:
            self.num_devices = available_devices[backend]
        elif num_devices > available_devices[backend]:
            raise ValueError(f"requested devices ({backend}: {num_devices}) cannot be greater than available backend devices: {available_devices}.")
        elif num_devices <= 0:
            raise ValueError(f"num_devices {num_devices} cannot be <= 0.")
        else:
            self.num_devices = num_devices
    
        if self.batch_size % self.num_devices != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be a multiple of the num_devices ({self.num_devices}).")
    
        print(f"Batch sharding set to {self.num_devices} {backend}(s)")
        devices = jax.local_devices(backend=backend)[:self.num_devices]
        mesh = jshard.Mesh(devices, ('batch',))
        pspec = jshard.PartitionSpec('batch',)
        self.sharding = jshard.NamedSharding(mesh, pspec)

    def shard_batch(self, batch):
        batch = jutil.tree_map(lambda x: jax.device_put(x, self.sharding), batch)
        return batch


class TAPDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Validate the feature dict
        features = cfg['features']
        for key, value in features.items():
            if not isinstance(value, list) and value is not None:
                raise ValueError(f"All feature dicts must contain a list. {key} is not a list.")
        
        # These are required for all models.
        self.target = features['target']
        self.daily_features = features['daily']
        # These are not and can pass None
        self.irregular_features = features.get('irregular')
        self.static_features = features.get('static') 
        
        with open(self.cfg['basin_file'], 'r') as file:
            basin_list = file.readlines()
            basin_list = [basin.strip() for basin in basin_list]
        self.basins = basin_list

        self.log_pad = 0.001

        self.x_s, self.attributes_scale = self._load_attributes()
        self.x_d, self.scale = self._load_basin_data()
        self.date_ranges = self._precompute_date_ranges()
        self.update_indices(data_subset='train')

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.basin_index_pairs)
    
    def _load_basin_data(self):
        """
        Loads the basin data from NetCDF files and applies the time slice.

        Returns:
            xr.Dataset: An xarray dataset of time series data with time and basin coordinates.
        """
        # Minimum date for sequenced training data
        min_train_date = (np.datetime64(self.cfg['time_slice'].start) +
                          np.timedelta64(self.cfg['sequence_length'], 'D'))  

        self.indices = defaultdict(dict)
        ds_list = []
        for basin in tqdm(self.basins, disable=self.cfg['quiet'], desc="Loading Basins"):
            file_path = f"{self.cfg['data_dir']}/time_series/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.cfg['time_slice'])
            ds['date'] = ds['date'].astype('datetime64[ns]')
    
            # Filter to keep only the necessary features and the target variable
            ds = ds[[*self.daily_features, *self.irregular_features, *self.target]]

            # Replace negative values with NaN in specific columns without explicit loop
            for col in self.cfg['log_norm_cols']:
                ds[col] = ds[col].where(ds[col] >= 0, np.nan)
            # Repetitive if target is in log_norm_cols, but not a problem. 
            if self.cfg['clip_target_to_zero']:
                ds[self.target] = ds[self.target].where(ds[self.target] >= 0, np.nan)

            # Testing if this nans are causing an issue
            for col in self.daily_features:
                ds[col] = ds[col].where(~np.isnan(ds[col]), 0)
                
            # Apply rolling means at 1 or more intervals.    
            window_sizes = self.cfg.get('add_rolling_means')
            if window_sizes is not None:
                ds = self.add_smoothed_features(ds, window_sizes)
                
            # Component masks for creating data indices
            is_train = ds['date'] < self.cfg['split_time']
            valid_sequence = ds['date'] >= min_train_date
            valid_irregular = (~np.isnan(ds[self.irregular_features])).to_array().all(dim='variable')
            valid_target = (~np.isnan(ds[self.target])).to_array().any(dim='variable')

            # Create valid data indices for this basin
            masks = [('pre-train', is_train & valid_sequence & valid_irregular & valid_target),
                     ('train', is_train & valid_sequence & valid_target),
                     ('test', ~is_train & valid_sequence & valid_target),
                     ('predict', ~is_train & valid_sequence)]
            for key, mask in masks:
                self.indices[key][basin] = ds['date'][mask].values
    
            ds = ds.assign_coords({'basin': basin})
            ds_list.append(ds)

        ds = xr.concat(ds_list, dim="basin")
        ds = ds.drop_duplicates('basin')
        x_d, scale = self._normalize_data(ds)
        
        return x_d, scale

    def _load_attributes(self):
        """
        Loads the basin attributes from a CSV file.

        Returns:
            dict: A basin-keyed dictionary of static attributes.
        """
        file_path = f"{self.cfg['data_dir']}/attributes/attributes.csv"
        attributes_df = pd.read_csv(file_path, index_col="index")
        attributes_df.index = attributes_df.index.astype(str)

        if self.static_features is not None:
            attributes_df = attributes_df[self.static_features]

        # Remove columns with zero variance
        zero_var_cols = list(attributes_df.columns[attributes_df.std(ddof=0) == 0])
        if zero_var_cols:
            print(f"Dropping static attributes with 0 variance: {zero_var_cols}")
            attributes_df.drop(columns=zero_var_cols, inplace=True)

        self.static_features = list(attributes_df.columns)

        mu = attributes_df.mean()
        std = attributes_df.std(ddof=0)
        scaled_attributes = (attributes_df-mu)/std

        # Create dict of the scaling factors
        scale = {var: {'offset': mu[var], 'scale': std[var]} for var in attributes_df.columns}
                
        x_s = {}
        for basin in self.basins:
            if basin not in attributes_df.index:
                raise KeyError(f"Basin '{basin}' not found in attributes DataFrame.")
            x_s[basin] = scaled_attributes.loc[basin].values
            
        return x_s, scale
    
    def _precompute_date_ranges(self):
        unique_dates = self.x_d['date'].values
        date_ranges = {date: pd.date_range(end=date, periods=self.cfg['sequence_length'], freq='D').values for date in unique_dates}
        return date_ranges

    
    def __getitems__(self, ids):
        """Generate one batch of data."""
        # Prepare to collect all basin and date information for the indices
        basins = [self.basin_index_pairs[idx][0] for idx in ids]
        dates = [self.basin_index_pairs[idx][1] for idx in ids]
        sequenced_dates = [self.date_ranges[date] for date in dates]

        # Convert to xarray-friendly formats
        basins_da = xr.DataArray(basins, dims="sample")
        sequenced_dates_da = xr.DataArray(sequenced_dates, dims=["sample", "time"])

        ds = self.x_d.sel(basin=basins_da, date=sequenced_dates_da)
        x_dd = np.moveaxis(ds[self.daily_features].to_array().values,0,2)
        x_di = np.moveaxis(ds[self.irregular_features].to_array().values,0,2)
        x_s = np.array([self.x_s[b] for b in basins])
        y = np.moveaxis(ds[self.target].to_array().values,0,2)

        batch = { 'x_dd': jnp.array(x_dd), 
                 'x_di': jnp.array(x_di), 
                 'x_s': jnp.array(x_s), 
                 'y': jnp.array(y)}
        
        return basins, dates, batch

    def _normalize_data(self, ds):
        """
        Normalize the input data using log normalization for specified variables and standard normalization for others.
        
        Returns:
            ds: the input xarray dataset after normalization
            scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
        """
        # Initialize
        scale = {k: {'offset': 0, 'scale': 1, 'log_norm': False} for k in ds.data_vars}
    
        # Subset the dataset to the training time period
        training_ds = ds.sel(date=slice(None, self.cfg['split_time']))
    
        # Iterate over each variable in the dataset
        for var in ds.data_vars:
            if var in self.cfg.get('log_norm_cols',[]):
                # Perform log normalization
                scale[var]['log_norm'] = True
                x = ds[var] + self.log_pad
                training_x = x.sel(date=slice(None, self.cfg['split_time']))
                scale[var]['offset'] = np.nanmean(np.log(training_x))    
                # Apply normalization and offset
                ds[var] = np.log(x) - scale[var]['offset']
            elif var in self.cfg.get('range_norm_cols',[]):
                # Perform min-max scaling
                scale[var]['scale'] = training_ds[var].max().values.item()
                ds[var] = ds[var] / scale[var]['scale']
            else:
                # Perform standard normalization
                scale[var]['offset'] = training_ds[var].mean().values.item()
                scale[var]['scale'] = training_ds[var].std().values.item()
                ds[var] = (ds[var] - scale[var]['offset']) / scale[var]['scale']
                
        return ds, scale
 
    def denormalize_target(self, y_normalized):
        """
        Denormalizes the target variable.

        Args:
            y_normalized (np.ndarray or jnp.ndarray): Normalized target data.

        Returns:
            np.ndarray or jnp.ndarray: Denormalized target data.
        """
        # Retrieve the normalization parameters for the target variable
        offset = self.scale[self.target]['offset']
        scale = self.scale[self.target]['scale']
        log_norm = self.scale[self.target]['log_norm']

        # Reverse the normalization process
        if log_norm:
            return np.exp(y_normalized + offset) - self.log_pad
        else:
            return y_normalized * scale + offset
      
    def update_indices(self, *,
                       data_subset:str,
                       basin_subset:list = []):
        # Validate the data_subset choice
        if data_subset not in self.indices.keys():
             raise ValueError(f"data_subset ({data_subset}) must be in data index dict keys ({self.indices.keys()}) ")
        ids = self.indices[data_subset]
            
        # If a non-empty subset list exists, we will only generate batches for those basins.
        if basin_subset:
            if not isinstance(basin_subset,list):
                basin_subset = [basin_subset]
            ids = {basin: ids[basin] for basin in basin_subset}

        basin_index_pairs = [(basin, date) for basin, dates in ids.items() for date in dates] 

        self.data_subset = data_subset
        self.basin_index_pairs = basin_index_pairs
        
    def add_smoothed_features(self, ds, window_sizes):
        new_ds = ds.copy()
        data_vars = ds.data_vars
        for window_size in window_sizes:
            # Apply rolling mean and rename variables
            for var_name in data_vars:
                # Perform rolling operation
                smoothed_var = ds[var_name].rolling(date=window_size, min_periods=1, center=False).mean(skipna=True)
                # Assign to new dataset with a new variable name
                new_ds[f"{var_name}_smooth{window_size}"] = smoothed_var
        return new_ds

def _get_available_devices():
    """
    Returns a dict of number of available backend devices
    """
    devices = {}
    for backend in ['cpu', 'gpu', 'tpu']:
        try:
            n = jax.local_device_count(backend=backend)
            devices[backend] = n
        except RuntimeError:
            pass   
    return devices


