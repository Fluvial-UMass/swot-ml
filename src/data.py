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

from utils import smart_tqdm


class TAPDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):    
        num_workers = kwargs.get('num_workers', 1)
        persistent_workers = False if num_workers==0 else kwargs.get('persistent_workers',True)
        
        super().__init__(dataset,
                         collate_fn=self.collate_batch,
                         shuffle=kwargs.get('shuffle', True),
                         batch_size=kwargs.get('batch_size', 1),
                         num_workers=num_workers,
                         pin_memory=kwargs.get('pin_memory', True),
                         persistent_workers=persistent_workers)     
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        # Dataset index params
        self.data_subset = kwargs.get('data_subset','train')
        self.basin_subset = kwargs.get('basin_subset',None)
        self.set_mode()

        # Data sharding params
        backend = kwargs.get('backend', None)
        num_devices = kwargs.get('num_devices', None)
        self.set_jax_sharding(backend, num_devices)

    @staticmethod
    def collate_batch(sample):
        """Collates sample data into batched data."""
        basin = [item['basin'] for item in sample]
        date = [item['date'] for item in sample]
        x_dd = jnp.stack([item['x_dd'] for item in sample])
        x_di = jnp.stack([item['x_di'] for item in sample])
        x_s = jnp.stack([item['x_s'] for item in sample])
        y = jnp.stack([item['y'] for item in sample])
        
        batch = {'x_dd':x_dd, 'x_di':x_di, 'x_s':x_s, 'y': y}
        return basin, date, batch

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

    Attributes:
        data_dir (Path): Path to the directory containing the data.
        basins (list): List of basin identifiers.
        features_dict (dict): Dictionary specifying the daily and irregular features.
        target (str): Name of the target variable.
        time_slice (slice): Time slice for selecting the data.
        split_time (np.datetime64): Date to split the data into training and testing sets.
        sequence_length (int): Length of the input sequences.
        train (bool): Whether to load training data. If False, loads testing data.
        log_norm_cols (list): List of columns to apply log normalization.
        range_norm_cols (list): List of columns to apply range normalization.
    """
    def __init__(self, *,
                 data_dir: Path, 
                 basin_file: Path,
                 features: dict,
                 time_slice: slice,
                 split_time: np.datetime64,
                 sequence_length: int,
                 sequence: bool = True,
                 log_norm_cols: list = [],
                 range_norm_cols: list = [],
                 clip_target_to_zero: bool = False,
                 quiet: bool = False,
                 **kwargs):

        # Validate the feature dict
        if not isinstance(features.get('daily'), list):
            raise ValueError("features_dict must contain a list of daily features at a minimum.")

        self.data_dir = data_dir
        self.basin_file = basin_file
        self.time_slice = time_slice
        self.split_time = split_time
        self.sequence_length = sequence_length
        self.log_norm_cols = log_norm_cols
        self.range_norm_cols = range_norm_cols
        self.clip_target_to_zero = clip_target_to_zero
        self.quiet = quiet
        
        with open(basin_file, 'r') as file:
            basin_list = file.readlines()
            basin_list = [basin.strip() for basin in basin_list]
        self.basins = basin_list
        
        self.daily_features = features['daily']
        self.irregular_features = features.get('irregular') # is None if key doesn't exist. 
        self.static_features = features.get('static')
        self.target = features['target']
        
        self.all_features =  self.daily_features + self.irregular_features

        self.log_pad = 0.001

        self.x_s, self.attributes_scale = self._load_attributes()
        self.x_d = self._load_basin_data()
        self.scale = self._normalize_data()
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
        min_train_date = (np.datetime64(self.time_slice.start) +
                          np.timedelta64(self.sequence_length, 'D'))  

        self.indices = defaultdict(dict)
        ds_list = []
        for basin in smart_tqdm(self.basins, self.quiet, desc="Loading Basins"):
            file_path = f"{self.data_dir}/time_series/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.time_slice)
            ds['date'] = ds['date'].astype('datetime64[ns]')
    
            # Filter to keep only the necessary features and the target variable
            ds = ds[[*self.daily_features, *self.irregular_features, self.target]]

            # Replace negative values with NaN in specific columns without explicit loop
            for col in self.log_norm_cols:
                ds[col] = ds[col].where(ds[col] >= 0, np.nan)
            # Repetitive if target is in log_norm_cols, but not a problem. 
            if self.clip_target_to_zero:
                ds[self.target] = ds[self.target].where(ds[self.target] >= 0, np.nan)

            # Testing if this is causing an issue
            for col in self.daily_features:
                ds[col] = ds[col].where(~np.isnan(ds[col]), 0)
                
            # Component masks for creating data indices
            is_train = ds['date'] < self.split_time
            valid_sequence = ds['date'] >= min_train_date
            valid_irregular = (~np.isnan(ds[self.irregular_features])).to_array().all(dim='variable')
            valid_target = ~np.isnan(ds[self.target])

            # Create valid data indices for this basin
            masks = [('pre-train', is_train & valid_sequence & valid_irregular & valid_target),
                     ('train', is_train & valid_sequence & valid_target),
                     ('test', ~is_train & valid_sequence)]
            for key, mask in masks:
                self.indices[key][basin] = ds['date'][mask].values
    
            ds = ds.assign_coords({'basin': basin})
            ds_list.append(ds)

        ds = xr.concat(ds_list, dim="basin")
        ds = ds.drop_duplicates('basin')
        
        return ds

    def _load_attributes(self):
        """
        Loads the basin attributes from a CSV file.

        Returns:
            dict: A basin-keyed dictionary of static attributes.
        """
        file_path = f"{self.data_dir}/attributes/attributes.csv"
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
        date_ranges = {date: pd.date_range(end=date, periods=self.sequence_length, freq='D').values for date in unique_dates}
        return date_ranges

    def __getitem__(self, idx):
        """Generate one batch of data."""
        basin, date = self.basin_index_pairs[idx]
        sequence_dates = self.date_ranges[date]
        x_dd = self.x_d.sel(basin=basin, date=sequence_dates)[self.daily_features].to_array().values.T
        x_di = self.x_d.sel(basin=basin, date=sequence_dates)[self.irregular_features].to_array().values.T
        x_s = self.x_s[basin]
        y = self.x_d.sel(basin=basin, date=sequence_dates)[self.target].values

        sample = {'basin': basin, 'date': date, 'x_dd': x_dd, 'x_di': x_di, 'x_s': x_s, 'y': y}
        return sample

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
                x = self.x_d[var] + self.log_pad
                training_x = x.sel(date=slice(None, self.split_time))
                scale[var]['offset'] = np.nanmean(np.log(training_x))    
                # Apply normalization and offset
                self.x_d[var] = np.log(x) - scale[var]['offset']
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


