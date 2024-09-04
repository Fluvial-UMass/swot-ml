import os
import hashlib
import yaml
import pickle
import copy
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import itertools
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jutil
import jax.sharding as jshard
from torch.utils.data import Dataset, DataLoader


class HydroDataLoader(DataLoader):
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
                         timeout=cfg.get('timeout',900),
                         persistent_workers=persistent_workers)     
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        # Batch sharding params
        backend = cfg.get('backend', None)
        num_devices = cfg.get('num_devices', None)
        self.set_jax_sharding(backend, num_devices)
        
    @staticmethod
    def collate_fn(sample):
        # I can't figure out how to just not collate. Can't even use lambdas because of multiprocessing.
        return sample
    
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
            raise ValueError(f"requested devices ({backend}: {num_devices}) cannot be greater than available backend devices ({available_devices}).")
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


class HydroDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """
    def __init__(self, cfg, *, 
                 inference_mode=False, 
                 dynamic_encoding=None, 
                 dynamic_scale=None,
                 static_encoding=None, 
                 static_scale=None):
        
        self.cfg = copy.deepcopy(cfg)
        self.log_pad = 0.001
        self.inference_mode = inference_mode

        self.features = cfg['features']
        _validate_feature_dict(self.features)  
        self.target = self.features['target']

        self._read_basin_files()
        self.x_s, self.s_encoding, self.s_scale = self._load_attributes(static_encoding, static_scale)
        self.x_d, self.d_encoding, self.d_scale = self._load_or_read_basin_data(dynamic_encoding, dynamic_scale)
        self.date_ranges = self._precompute_date_ranges()

        self.update_indices(data_subset=cfg.get('data_subset','train'),
                            exclude_target=cfg.get('exclude_target_from_index'),
                            basin_subset=cfg.get('basin_subset'))

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.basin_index_pairs)
    
    def _read_basin_files(self):   
        # for convenience and readability
        data_dir = self.cfg.get('data_dir')
        basin_file = self.cfg.get('basin_file')
        train_basin_file = self.cfg.get('train_basin_file')
        test_basin_file = self.cfg.get('test_basin_file')

        def read_file(fp):
            with open(fp, 'r') as file:
                basin_list = file.readlines()
                basin_list = [basin.strip() for basin in basin_list]
                return basin_list 
        
        # Same basins are used for train and test.
        if basin_file:
            self.all_basins = read_file(data_dir / basin_file)
            self.train_basins = self.test_basins = self.all_basins
        # Seperate files for train and test
        elif train_basin_file and test_basin_file:
            self.train_basins = read_file(data_dir / train_basin_file)
            self.test_basins = read_file(data_dir / test_basin_file)
            self.all_basins = list(set(self.train_basins + self.test_basins))
        else:
            raise ValueError('Must set either "basin_file" or "train_basin_file" AND "test_basin_file"')
        # make sure order of basins doesn't affect hasing. 
        self.all_basins.sort()


    def _load_or_read_basin_data(self, encoding, scale):
        data_hash = self.get_data_hash()
        print(f"Data Hash: {data_hash}")

        data_file = self.cfg.get('data_dir') / "cache" / f"{data_hash}.pkl"
        # If data from this cfg hash exists, read it in.
        if data_file.is_file():
            print("Using cached basin dataset file.")
            with open(data_file, 'rb') as file:
                x_d, scale, encoding, self.features['dynamic'] = pickle.load(file)
        # Else load the dataset from basin files and save it.
        else:
            x_d, encoding, scale = self._load_basin_data(encoding, scale)
            # Save our new loaded data
            with open(data_file, 'wb') as file:
                pickle.dump((x_d, scale, encoding, self.features['dynamic']), file)
            
        return x_d, encoding, scale
    
    def _load_basin_data(self, encoding=None, scale=None):
        """
        Loads the basin data from NetCDF files and applies the time slice.

        Returns:
            xr.Dataset: An xarray dataset of time series data with time and basin coordinates.
        """
        ds_list = []
        for basin in tqdm(self.all_basins, disable=self.cfg['quiet'], desc="Loading Basins"):
            file_path = f"{self.cfg['data_dir']}/time_series/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.cfg['time_slice'])
            ds['date'] = ds['date'].astype('datetime64[ns]')
    
            # Filter to keep only the necessary features and the target variable if not in inference mode
            features_to_keep = list(itertools.chain(*self.features['dynamic'].values()))
            if not self.inference_mode:
                features_to_keep.extend(self.target)
            
            missing_columns = set(features_to_keep) - set(ds.data_vars)
            if missing_columns:
                raise ValueError(
                    f"The following columns are missing from the dataset: {missing_columns}"
                    f"The following variables are available in the dataset: {ds.data_vars}"
                )
            ds = ds[features_to_keep]

            # Clip selected columns to the specified range. This range is preprocessed in config.py.
            if self.cfg['clip_feature_range'] and not self.inference_mode:
                for col, [lower, upper] in self.cfg['clip_feature_range'].items():
                    if col not in ds:
                        warnings.warn(f"Column '{col}' not found in dataset. Skipping clipping for this column.", UserWarning)
                        continue
                    inside_range = (ds[col] >= lower) & (ds[col] <= upper)
                    ds[col] = ds[col].where(inside_range, np.nan)

            # Replace negative values with NaN in specific columns without explicit loop
            for col in self.cfg['log_norm_cols']:
                ds[col] = ds[col].where(ds[col] >= 0, np.nan)
                
            # Apply rolling means at 1 or more intervals.    
            window_sizes = self.cfg.get('add_rolling_means')
            if window_sizes is not None:
                ds = self.add_smoothed_features(ds, window_sizes)
    
            ds = ds.assign_coords({'basin': basin})
            ds_list.append(ds)

        ds = xr.concat(ds_list, dim="basin")
        ds = ds.drop_duplicates('basin')

        ds, encoding = self._encode_data(ds, 'dynamic', encoding)
        x_d, scale = self._normalize_data(ds, 'dynamic', encoding, scale)
        
        return x_d, encoding, scale

    def _load_attributes(self, encoding=None, scale=None):
        """
        Loads the basin attributes from a CSV file.

        Returns:
            xr.Dataset: An xarray dataset of attribute data with basin coordinates.
        """
        feat = self.features['static']
        if isinstance(feat, list) and len(feat)==0:
            return None, None

        file_path = f"{self.cfg['data_dir']}/attributes/attributes.csv"
        df = pd.read_csv(file_path, index_col="index")
        df.index = df.index.astype(str)

        # Trim the dataset to the config'd list.
        if feat is not None:
            df = df[feat]

        #Remove columns with zero variance or NaN values
        zero_var_cols = list(df.columns[df.std(ddof=0) == 0])
        nan_cols = list(df.columns[df.isna().any()])
        cols_to_drop = list(set(zero_var_cols + nan_cols)) 
        if cols_to_drop:
            warnings.warn(f"Dropping numerical attributes with 0 variance or NaN values: {cols_to_drop}", UserWarning)
            df.drop(columns=cols_to_drop, inplace=True)

        # Update or set the static feature list.
        if feat is None:
            self.features['static'] = list(df.columns)

        # Convert the DataFrame to an xarray Dataset
        ds = df.to_xarray().rename({'index': 'basin'})
        ds, encoding = self._encode_data(ds, 'static', encoding)
        x_s, scale = self._normalize_data(ds, 'static', encoding, scale)

        return x_s, encoding, scale


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

        batch = {'dynamic':{}}
        for source, col_names in self.features['dynamic'].items():
            batch['dynamic'][source] = jnp.array(np.moveaxis(ds[col_names].to_array().values,0,2))

        # Handle static data as an xarray Dataset
        if self.x_s is not None:
            static_ds = self.x_s.sel(basin=basins_da)
            batch['static'] = jnp.array(np.moveaxis(static_ds.to_array().values, 0, 1)) 
        
        if not self.inference_mode:
            batch['y'] = jnp.array(np.moveaxis(ds[self.target].to_array().values,0,2))
        
        return basins, dates, batch
    
    def _encode_data(self, ds, feat_group, encoding:dict | None):
        columns_in = ds.data_vars
        one_hot_enc = encoding.get('one_hot') if encoding else None
        bitmask_enc = encoding.get('bitmask') if encoding else None

        ds, one_hot = self._one_hot_encoding(ds, feat_group, one_hot_enc)
        ds, bitmask = self._bitmask_expansion(ds, feat_group, bitmask_enc)

        new_columns = set(ds.data_vars) - set(columns_in)

        encoding = {'one_hot': one_hot,
                    'bitmask': bitmask,
                    'encoded_columns': list(new_columns)}
        
        return ds, encoding

    def _one_hot_encoding(self, ds, feat_group, onehot_enc:dict | None):
        # Apply one-hot encoding to categorical columns
        if not onehot_enc:
            categorical_cols = self.cfg.get('categorical_cols',{}).get(feat_group, [])
            if not categorical_cols:
                return ds, None
            onehot_enc = {col: None for col in categorical_cols}
        
        for col, prescribed_cols in onehot_enc.items():
            if col not in ds.data_vars:
                continue

            df = ds[col].to_dataframe()
            encoded = pd.get_dummies(df.astype(str), prefix=col)

            if prescribed_cols:
                # Add missing categories as columns filled with zeros
                for c in prescribed_cols:
                    if c not in encoded.columns:
                        encoded[c] = 0
                # Filter out columns not in the prescribed encoding
                encoded = encoded[prescribed_cols]
            else:
                onehot_enc[col] = encoded.columns
            
            # Add encoded data and remove the original categorical column
            ds = xr.merge([ds, encoded.to_xarray()])
            ds = ds.drop_vars(col)

            # Locate the col inside the features dict, remove and replace.
            # This is kind of ugly but deals with the 2 level feature dict.
            if feat_group == 'dynamic':
                for source, source_features in self.features[feat_group].items():
                    if col in source_features:
                        self.features[feat_group][source].remove(col)
                        self.features[feat_group][source].extend(encoded.columns)
            else:
                self.features[feat_group].remove(col)
                self.features[feat_group].extend(encoded.columns)


        return ds, onehot_enc

    def _bitmask_expansion(self, ds, feat_group, bitmask_enc:dict | None):
        if not bitmask_enc:
            bitmask_cols = self.cfg.get('bitmask_cols',{}).get(feat_group,[])
            if not bitmask_cols:
                return ds, None
            bitmask_enc = {k: None for k in bitmask_cols}
        
        for col, num_bits in bitmask_enc.items():
            if col not in ds.data_vars:
                continue
            # Get the bitmask integers
            x = ds[col].values
            x[np.isnan(x)] = 0
            x = x.astype(int)

            if not num_bits:
                num_bits = int(np.ceil(np.log2(x.max())))
                bitmask_enc[col] = num_bits

            # Expand into bits
            new_vars = {}
            for n in range(num_bits):
                bit_arr = (x // 2**n) % 2

                new_vars[f"{col}_bit_{n}"] = xr.DataArray(
                    data=bit_arr,
                    dims=['basin', 'date'],
                    coords={'basin': ds.basin, 'date': ds.date}
                )

            ds = xr.merge([ds, xr.Dataset(new_vars)])
            # Remove the original categorical column
            ds = ds.drop_vars(col)

            # Locate the col inside the dynamic features dict, remove and replace.
            # This is kind of ugly but deals with the 2 level feature dict.
            if feat_group == 'dynamic':
                for source, source_features in self.features['dynamic'].items():
                    if col in source_features:
                        self.features['dynamic'][source].remove(col)
                        self.features['dynamic'][source].extend(new_vars.keys())
            else:
                self.features[feat_group].remove(col)
                self.features[feat_group].extend(new_vars.keys())
        
        return ds, bitmask_enc

    def _normalize_data(self, ds, feat_group, encoding, scale=None):
        """
        Normalize the input data using log normalization for specified variables and standard normalization for others.
        
        Returns:
            ds: the input xarray dataset after normalization
            scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
        """
        if scale is None:
            # Subset the dataset to the training time period
            if feat_group == 'dynamic':
                training_ds = ds.sel(date=slice(None, self.cfg['split_time']))
            else:
                training_ds = ds

            # Initialize
            scale = {k: {'encoded': False,
                         'log_norm': False,
                         'offset': 0,
                         'scale': 1} for k in ds.data_vars}
        
            # Iterate over each variable in the dataset and calculate scaler
            for var in ds.data_vars:
                if var in encoding['encoded_columns']:
                    # One-hot encoded columns don't need normalization
                    scale[var]['encoded'] = True
                
                elif var in self.cfg.get('log_norm_cols',[]):
                    # Log normalization
                    scale[var]['log_norm'] = True
                    x = training_ds[var] + self.log_pad
                    scale[var]['offset'] = np.nanmean(np.log(x))
                    
                elif var in self.cfg.get('range_norm_cols',[]):
                    # Min-max scaling
                    min_val = training_ds[var].min().values.item()
                    max_val = training_ds[var].max().values.item()
                    scale[var]['offset'] = min_val
                    scale[var]['scale'] = max_val - min_val
                    
                else:
                    # Standard normalization
                    scale[var]['offset'] = training_ds[var].mean().values.item()
                    scale[var]['scale'] = training_ds[var].std().values.item()

        for var in ds.data_vars:
            scl = scale[var]
            if scl['encoded']:
                continue
            elif scl['log_norm']:
                ds[var] = np.log(ds[var] + self.log_pad) - scl['offset']
            else:
                ds[var] = (ds[var] - scl['offset']) / scl['scale']

        return ds, scale
 
    def denormalize_target(self, y_normalized):
        """
        Denormalizes the target variable.

        Args:
            y_normalized (np.ndarray or jnp.ndarray): Normalized target data.

        Returns:
            np.ndarray or jnp.ndarray: Denormalized target data.
        """
        y = jnp.empty_like(y_normalized)
    
        for i in range(len(self.target)):
            # Retrieve the normalization parameters for the target variable
            target = self.target[i]
            offset = self.d_scale[target]['offset']
            scale = self.d_scale[target]['scale']
            log_norm = self.d_scale[target]['log_norm']

            # Reverse the normalization process using .at and .set
            if log_norm:
                y = y.at[:, i].set(jnp.exp(y_normalized[:, i] + offset) - self.log_pad)
            else:
                y = y.at[:, i].set(y_normalized[:, i] * scale + offset)
        return y
    
    def date_batching(self, data_subset='predict_all', date_range=None):
        self.data_subset = data_subset

        if self.data_subset not in ['predict','predict_all']:
            raise ValueError(f"Invalid data_subset: {self.data_subset}")

        # Minimum date for sequenced training data
        min_train_date = (np.datetime64(self.cfg['time_slice'].start) +
                        np.timedelta64(self.cfg['sequence_length'], 'D'))  
        valid_sequence = self.x_d['date'] >= min_train_date

        all_dates = self.x_d['date'].values
        if date_range:
            start_date = np.datetime64(date_range[0])
            end_date = np.datetime64(date_range[1])
            in_date_range = (start_date <= all_dates) & (all_dates < end_date)
            valid_dates = all_dates[in_date_range & valid_sequence]
        elif data_subset == 'predict':
            is_train = self.x_d['date'] < self.cfg['split_time']
            valid_dates = all_dates[~is_train & valid_sequence]
        else:
            valid_dates = all_dates[valid_sequence]
        
        if self.data_subset == 'predict':
            basins = self.test_basins
        else:
            basins = self.all_basins
            
        self.basin_index_pairs = []
        for date in valid_dates:
            self.basin_index_pairs.extend([(basin, date) for basin in basins])

        n_basins = len(basins)
        dataloader_kwargs = {'shuffle': False, 'batch_size': n_basins}

        return dataloader_kwargs
    
    def update_data_masks(self):
        # Validate the data_subset choice
        subset_choices = ['pre_train','train','test','predict','predict_all']
        if self.data_subset not in subset_choices:
             raise ValueError(f"data_subset ({self.data_subset}) must be in ({subset_choices}) ")
        
        # Minimum date for sequenced training data
        min_train_date = (np.datetime64(self.cfg['time_slice'].start) +
                          np.timedelta64(self.cfg['sequence_length'], 'D'))  
        indices = {}

        if self.basin_subset is not None:
            basins = self.basin_subset
        elif self.data_subset in ['pre_train','train']:
            basins = self.train_basins
        elif self.data_subset in ['test','predict']:
            basins = self.test_basins
        elif self.data_subset == 'predict_all':
            basins = self.all_basins

        for basin in tqdm(basins, disable=self.cfg['quiet'], desc="Updating Indices"):
            ds = self.x_d.sel(basin=basin)

            # Component masks for creating data indices
            is_train = ds['date'] < self.cfg['split_time']
            valid_sequence = ds['date'] >= min_train_date

            def valid_target():
                return (~np.isnan(ds[self.targets_to_index])).to_array().any(dim='variable')

            def valid_irregular():
                if self.irregular_features:
                    valid_irregular = (~np.isnan(ds[self.irregular_features])).to_array().all(dim='variable')
                else:
                    valid_irregular = True
                return valid_irregular

            # Create valid data indices for this basin
            if self.data_subset == 'pre_train':
                mask = is_train & valid_sequence & valid_irregular() & valid_target()
            elif self.data_subset == 'train':
                mask = is_train & valid_sequence & valid_target()
            elif self.data_subset == 'test':
                mask =  ~is_train & valid_sequence & valid_target()
            elif self.data_subset == 'predict':
                mask =  ~is_train & valid_sequence
            elif self.data_subset == 'predict_all':
                mask = valid_sequence
            indices[basin] = ds['date'][mask].values

        return indices

    def update_indices(self, data_subset:str, exclude_target:list=None, basin_subset:list = None):
        self.data_subset = data_subset
          
        if basin_subset is not None and not isinstance(basin_subset,list):
            basin_subset = [basin_subset]
        self.basin_subset = basin_subset

        if exclude_target:
            self.targets_to_index = [item for item in self.target if item not in exclude_target]
        else:
            self.targets_to_index = self.target
            
        ids = self.update_data_masks()
            
        # If a non-empty subset list exists, we will only generate batches for those basins.
        if self.basin_subset:
            ids = {basin: ids[basin] for basin in self.basin_subset}

        # These are the indices that will be used for selecting sequences of data.
        basin_index_pairs = [(basin, date) for basin, dates in ids.items() for date in dates] 
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
    
    def get_data_hash(self):
        cfg_keys = ['data_dir', 'features', 'exclude_target_from_index', 
                    'time_slice', 'split_time', 'add_rolling_means', 
                    'log_norm_cols', 'categorical_cols', 'bitmask_cols', 'range_norm_cols',
                    'clip_feature_range']
        data_config = {k: self.cfg.get(k) for k in cfg_keys}
        data_config['basins'] = self.all_basins

        """Generate a SHA256 hash for the contents of the dict."""
        hasher = hashlib.sha256()
        # Convert the dictionary to a sorted, consistent string representation
        dict_str = yaml.dump(data_config, sort_keys=True)
        hasher.update(dict_str.encode('utf-8'))
        return hasher.hexdigest()

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

def _validate_feature_dict(d):
    if not isinstance(d, dict):
        raise ValueError("features in config must be a dict. See examples.")
    
    invalid_entries = []
    for key, value in d.items():
        if key == 'dynamic':
            if not isinstance(value, dict):
                raise ValueError(f"The features dict key 'daily' must be a dict.")
            else:
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, list):
                        invalid_entries.append(str(sub_key))
        elif not isinstance(value, list) and value is not None:
            invalid_entries.append(str(key))
    
    if len(invalid_entries)>0:
        raise ValueError(f"The features dict in config file must contains lists. {invalid_entries} is not a list.")

