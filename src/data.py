import os
from pathlib import Path
from tqdm.notebook import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jutil
import jax.sharding as jshard
from torch.utils.data import Dataset, DataLoader

class TAPDataLoader(DataLoader):
    def __init__(self, **kwargs):
        dataset = kwargs.get('dataset', TAPDataset(**kwargs))
        super().__init__(dataset,
                         collate_fn=self.collate_batch,
                         shuffle=kwargs.get('suffle', True),
                         batch_size=kwargs.get('batch_size', 1),
                         num_workers=kwargs.get('num_workers', 1),
                         pin_memory=kwargs.get('pin_memory', True),
                         persistent_workers=kwargs.get('peristent_workers',True))
        
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

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

    def set_mode(self, *, train, shuffle=None, basin_subset=None):
        self.train = train
        self.shuffle = train if shuffle is None else shuffle
        self.dataset.update_indices(train, basin_subset)
    
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
        batch_size (int): Size of batches.
        sequence_length (int): Length of the input sequences.
        train (bool): Whether to load training data. If False, loads testing data.
        discharge_col (str): Name of the discharge column, if any.
        log_norm_cols (list): List of columns to apply log normalization.
        range_norm_cols (list): List of columns to apply range normalization.
    """
    def __init__(self, *,
                 data_dir: Path, 
                 basin_list: list,
                 features_dict: dict,
                 target: str,
                 time_slice: slice,
                 split_time: np.datetime64,
                 batch_size: int,
                 sequence_length: int,
                 train: bool = True,
                 discharge_col: str = None,
                 log_norm_cols: list = [],
                 range_norm_cols: list = [],
                 num_devices: int = 0,
                 backend: str = None,
                 **kwargs):

        # Validate the feature dict
        if not isinstance(features_dict.get('daily'), list):
            raise ValueError("features_dict must contain a list of daily features at a minimum.")

        self.data_dir = data_dir
        self.basins = basin_list
        self.target = target
        self.time_slice = time_slice
        self.split_time = split_time
        self.sequence_length = sequence_length
        self.train = train
        self.log_norm_cols = log_norm_cols
        self.range_norm_cols = range_norm_cols

        # self.update_sharding(backend, num_devices)
        
        self.all_features =  [value for sublist in features_dict.values() for value in sublist] # Unpack all features
        self.daily_features = features_dict['daily']
        self.discharge_idx = features_dict['daily'].index(discharge_col)
        self.irregular_features = features_dict.get('irregular') # is None if key doesn't exist. 

        self.log_pad = 0.001
        
        self.train_ids = {}
        self.test_ids = {}

        self.x_s = self._load_attributes()
        self.x_d = self._load_basin_data()
        self.scale = self._normalize_data()
        self.date_ranges = self._precompute_date_ranges()

        self.update_indices(train, basin_list)

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
        ds_list = []
        for basin in tqdm(self.basins, desc="Loading Basins"):
            file_path = f"{self.data_dir}/time_series/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.time_slice)
            ds['date'] = ds['date'].astype('datetime64[ns]')
    
            # Filter to keep only the necessary features and the target variable
            ds = ds[[*self.all_features, self.target]]
    
            # Create dict of train and test dates
            min_train_date = (np.datetime64(self.time_slice.start) +
                              np.timedelta64(self.sequence_length, 'D'))  # Minimum date for training data
            train_mask = (ds['date'] >= min_train_date) & (ds['date'] < self.split_time) & (~np.isnan(ds[self.target]))
            train_dates = ds['date'][train_mask].values
            self.train_ids[basin] = train_dates
    
            test_mask = ds['date'] >= self.split_time
            test_dates = ds['date'][test_mask].values
            self.test_ids[basin] = test_dates
    
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
        x_s = {}
        for basin in self.basins:
            if basin not in attributes_df.index:
                raise KeyError(f"Basin '{basin}' not found in attributes DataFrame.")
            x_s[basin] = attributes_df.loc[basin].values
        return x_s
    
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

        sample = {'basin':basin, 'date':date, 'x_dd':x_dd, 'x_di':x_di, 'x_s':x_s, 'y':y}
        # sample = jutil.tree_map(lambda x: jax.device_put(x, self.named_sharding), sample)
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
                # self.x_d[var] = self.x_d[var].where(self.x_d[var]<0, 0)
                self.x_d[var] = np.log(self.x_d[var] + self.log_pad)
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
            return np.exp(y_normalized) - self.log_pad
        else:
            return y_normalized * scale + offset

    def update_indices(self, train, basin_subset=[]):
        if train:
            basin_index_pairs = [(basin, date) for basin, dates in self.train_ids.items() for date in dates]
        else:
            # If a non-empty subset list exists, we will only generate batches for those basins.
            if basin_subset:
                test_ids = {basin: self.test_ids[basin] for basin in basin_subset}
            else:
                test_ids = self.test_ids
            basin_index_pairs = [(basin, date) for basin, dates in test_ids.items() for date in dates] 
            
        self.train = train
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


