import hashlib
import yaml
import pickle
import copy
from tqdm.auto import tqdm
import itertools
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset


class HydroDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(self, cfg: dict, *, train_ds=None, use_cache=True):
        self.cfg = copy.deepcopy(cfg)
        self.log_pad = 0.001
        self.dataloader_kwargs = {}

        self.inference_mode = isinstance(train_ds, HydroDataset)
        self.s_encoding = train_ds.s_encoding if self.inference_mode else None
        self.s_scale = train_ds.s_scale if self.inference_mode else None
        self.d_encoding = train_ds.d_encoding if self.inference_mode else None
        self.d_scale = train_ds.d_scale if self.inference_mode else None

        self.features = self.cfg['features']
        self.target = self.features['target']

        self._read_basin_files()
        self.x_s = self._load_attributes()
        self.x_d = self._load_or_read_basin_data(use_cache)
        self.date_ranges = self._precompute_date_ranges()

        self.update_indices(data_subset=self.cfg.get('data_subset', 'train'), basin_subset=self.cfg.get('basin_subset'))

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sequence_indices)

    def _read_basin_files(self):
        # for convenience and readability
        data_dir = self.cfg.get('data_dir')
        basin_file = self.cfg.get('basin_file')
        train_basin_file = self.cfg.get('train_basin_file')
        test_basin_file = self.cfg.get('test_basin_file')
        graph_network_file = self.cfg.get('graph_network_file')

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

        if graph_network_file:
            self.graph_mode = True
            if self.train_basins == self.test_basins:
                self.graph_matrix = np.loadtxt(data_dir / graph_network_file)
                if self.graph_matrix.shape[0] != len(self.train_basins):
                    raise ValueError('Graph network matrix shape be square of number of training basins.\n' +
                                     f'Graph network matrix shape: {self.graph_matrix.shape}\n' +
                                     f'Number of training basins: {len(self.train_basins)}.')
                if self.graph_matrix.shape[0] != self.graph_matrix.shape[1]:
                    raise ValueError('Graph network matrix must be square.\n' +
                                     f'Graph network matrix shape: {self.graph_matrix.shape}.')
            else:
                raise ValueError(
                    'Graph network modeling does not currently support different training and testing networks.')
        else:
            self.graph_matrix = None
            self.graph_mode = False

    def _load_or_read_basin_data(self, use_cache):
        print('Loading dynamic data')
        if use_cache:
            data_hash = self.get_data_hash()
            print(f"Data Hash: {data_hash}")

            cache_dir = self.cfg.get('data_dir') / "cache"
            cache_dir.mkdir(exist_ok=True)
            data_file = cache_dir / f"{data_hash}.pkl"

            # If data from this cfg hash exists, read it in.
            if data_file.is_file():
                print("Using cached basin dataset.")
                with open(data_file, 'rb') as file:
                    x_d, self.d_scale, self.d_encoding, self.features['dynamic'], self.time_gaps = pickle.load(file)
            # Else load the dataset from basin files and save it.
            else:
                print("No matching cached dataset.")
                x_d = self._load_basin_data()
                # Save our new loaded data
                with open(data_file, 'wb') as file:
                    pickle.dump((x_d, self.d_scale, self.d_encoding, self.features['dynamic'], self.time_gaps), file)
        else:
            x_d = self._load_basin_data()

        return x_d

    def _load_basin_data(self):
        """
        Loads the basin data from NetCDF files and applies the time slice.

        Returns:
            xr.Dataset: An xarray dataset of time series data with time and basin coordinates.
        """
        ts_dir = self.cfg.get('time_series_dir')
        ts_dir = 'time_series' if ts_dir is None else ts_dir

        ds_list = []
        for basin in tqdm(self.all_basins, disable=self.cfg['quiet'], desc="Loading Basins"):
            file_path = f"{self.cfg['data_dir']}/{ts_dir}/{basin}.nc"
            ds = xr.open_dataset(file_path).sel(date=self.cfg['time_slice'])
            ds['date'] = ds['date'].astype('datetime64[ns]')

            # Filter to keep only the necessary features and the target variable if not in inference mode
            features_to_keep = list(itertools.chain(*self.features['dynamic'].values()))
            if not self.inference_mode:
                features_to_keep.extend(self.target)

            missing_columns = set(features_to_keep) - set(ds.data_vars)
            if missing_columns:
                raise ValueError(f"The following columns are missing from the dataset: {missing_columns}"
                                 f"The following variables are available in the dataset: {ds.data_vars}")
            ds = ds[features_to_keep]

            # Clip selected columns to the specified range. This range is preprocessed in config.py.
            for col in ds.data_vars:
                if col not in self.cfg.get('clip_feature_range', {}).keys():
                    continue
                [lower, upper] = self.cfg['clip_feature_range'][col]
                inside_range = (ds[col] >= lower) & (ds[col] <= upper)
                ds[col] = ds[col].where(inside_range, np.nan)

            # Replace negative values with NaN in specific columns without explicit loop
            for col in ds.data_vars:
                if col not in self.cfg.get('log_norm_cols', []):
                    continue
                ds[col] = ds[col].where(ds[col] >= 0, np.nan)

            # Apply rolling means at 1 or more intervals.
            window_sizes = self.cfg.get('add_rolling_means')
            if window_sizes is not None:
                ds = self.add_smoothed_features(ds, window_sizes)

            ds = ds.assign_coords({'basin': basin})
            ds_list.append(ds)

        ds = xr.concat(ds_list, dim="basin")
        ds = ds.drop_duplicates('basin')

        # Check for missing values in each feature group
        self.time_gaps = {}
        for group, variables in self.features['dynamic'].items():
            self.time_gaps[group] = any(ds[variables].isnull().any().to_array())

        ds, self.d_encoding = self._encode_data(ds, 'dynamic', self.d_encoding)
        x_d, self.d_scale = self._normalize_data(ds, 'dynamic', self.d_encoding, self.d_scale)

        return x_d

    def _load_attributes(self):
        """
        Loads the basin attributes from a CSV file.

        Returns:
            xr.Dataset: An xarray dataset of attribute data with basin coordinates.
        """
        print('Loading static attributes')
        file_stem = self.cfg.get('attributes_file')
        file_stem = 'attributes.csv' if file_stem is None else file_stem
        file_path = f"{self.cfg['data_dir']}/attributes/{file_stem}"
        df = pd.read_csv(file_path, index_col="index")
        df.index = df.index.astype(str)

        if self.inference_mode:
            unencoded_cols = [k for k, v in self.s_scale.items() if not v['encoded']]
            one_hot_cols = list((self.s_encoding['one_hot'] or {}).keys())
            bitmask_cols = list((self.s_encoding['bitmask'] or {}).keys())
            feat = unencoded_cols + one_hot_cols + bitmask_cols
            df = df[feat]

        else:
            # Trim the dataset to the config'd list.
            feat = self.features['static']
            if isinstance(feat, list) and len(feat) == 0:
                self.s_scale = None
                return None
            df = df[feat] if feat else df

            #Remove columns with zero variance or NaN values
            nan_cols = list(df.columns[df.isna().any()])
            zero_var_cols = list(df.columns[df.std(ddof=0) == 0])
            cols_to_drop = list(set(zero_var_cols + nan_cols))
            if cols_to_drop:
                print(f"Dropping numerical attributes with 0 variance or NaN values: {cols_to_drop}")
                df.drop(columns=cols_to_drop, inplace=True)

        # Update or set the static feature list.
        self.features['static'] = list(df.columns)

        # Convert the DataFrame to an xarray Dataset
        ds = df.to_xarray().rename({'index': 'basin'})
        ds, self.s_encoding = self._encode_data(ds, 'static', self.s_encoding)
        x_s, self.s_scale = self._normalize_data(ds, 'static', self.s_encoding, self.s_scale)

        return x_s

    def _precompute_date_ranges(self):
        unique_dates = self.x_d['date'].values
        date_ranges = {
            date: pd.date_range(end=date, periods=self.cfg['sequence_length'], freq='D').values for date in unique_dates
        }
        return date_ranges

    def _calc_var_dt(self, x):
        valid_mask = np.all(~np.isnan(x), axis=2)
        indices = np.arange(valid_mask.shape[1])

        valid_indices = np.where(valid_mask, indices, -1)
        last_valid_index = np.maximum.accumulate(valid_indices, axis=1)

        first_values = valid_mask[:, 0].astype(int)[:, None]
        dt = np.concat([first_values, np.diff(last_valid_index, axis=1)], axis=1)
        return dt

    def __getitems__(self, ids):
        """Generate one batch of data."""
        # Collect all basin and date information for the indices
        if self.graph_mode:
            basins = np.tile(self.basin_subset, (len(ids), 1))
            basins_da = xr.DataArray(basins, dims=["sample", "basins"])
            dates = [self.sequence_indices[idx] for idx in ids]
        else:
            basins = [self.sequence_indices[idx][0] for idx in ids]
            basins_da = xr.DataArray(basins, dims="sample")
            dates = [self.sequence_indices[idx][1] for idx in ids]
        sequenced_dates = [self.date_ranges[date] for date in dates]

        # Convert to xarray-friendly formats
        sequenced_dates_da = xr.DataArray(sequenced_dates, dims=["sample", "time"])
        ds = self.x_d.sel(basin=basins_da, date=sequenced_dates_da)

        if self.graph_mode:
            batch = {'dynamic': {}}
            # Dynamic data. Shape (batch, sequence, nodes, features)
            for source, col_names in self.features['dynamic'].items():
                batch['dynamic'][source] = np.moveaxis(ds[col_names].to_array().values, [0, 1, 3], [-1, 0, 1])
                # dt calcs not yet implemented for new dimension.
                # batch['dynamic_dt'][source] = self._calc_var_dt(batch['dynamic'][source])

            # Static data. Shape (batch, nodes, features)
            if self.x_s is not None:
                static_ds = self.x_s.sel(basin=basins_da)
                batch['static'] = np.moveaxis(static_ds.to_array().values, 0, -1)

            # Target data. Shape (batch, sequence, features)
            if not self.inference_mode:
                batch['y'] = np.moveaxis(ds[self.target].to_array().values, [0, 1, 3], [-1, 0, 1])

        else:
            batch = {'dynamic': {}, 'dynamic_dt': {}}
            # Dynamic data. Shape (batch, sequence, features)
            for source, col_names in self.features['dynamic'].items():
                batch['dynamic'][source] = np.moveaxis(ds[col_names].to_array().values, 0, -1)
                batch['dynamic_dt'][source] = self._calc_var_dt(batch['dynamic'][source])

            # Static data. Shape (batch, features)
            if self.x_s is not None:
                static_ds = self.x_s.sel(basin=basins_da)
                batch['static'] = np.moveaxis(static_ds.to_array().values, 0, 1)

            # Target data. Shape (batch, sequence, features)
            if not self.inference_mode:
                batch['y'] = np.moveaxis(ds[self.target].to_array().values, 0, 2)

        return basins, dates, batch

    def _encode_data(self, ds, feat_group, encoding):
        columns_in = ds.data_vars
        one_hot_enc = encoding.get('one_hot') if encoding else None
        bitmask_enc = encoding.get('bitmask') if encoding else None

        ds, one_hot = self._one_hot_encoding(ds, feat_group, one_hot_enc)
        ds, bitmask = self._bitmask_expansion(ds, feat_group, bitmask_enc)

        new_columns = set(ds.data_vars) - set(columns_in)

        encoding = {'one_hot': one_hot, 'bitmask': bitmask, 'encoded_columns': list(new_columns)}

        return ds, encoding

    def _one_hot_encoding(self, ds, feat_group, onehot_enc: dict | None):
        # Apply one-hot encoding to categorical columns
        if not onehot_enc:
            categorical_cols = self.cfg.get('categorical_cols', {}).get(feat_group, [])
            if not categorical_cols:
                return ds, None
            onehot_enc = {col: None for col in categorical_cols}

        for col, prescribed_cols in onehot_enc.items():
            if col in ds.data_vars:
                df = ds[col].to_dataframe()
                encoded = pd.get_dummies(df.astype(str), prefix=col)
                # Remove the original categorical column
                ds = ds.drop_vars(col)
            else:
                # Create an empty DataFrame with the same index as ds
                encoded = pd.DataFrame(index=ds.basin)

            if prescribed_cols is not None and len(prescribed_cols) > 0:
                # Add missing categories as columns filled with zeros
                for c in prescribed_cols:
                    if c not in encoded.columns:
                        encoded[c] = 0
                # Filter out columns not in the prescribed encoding
                encoded = encoded[prescribed_cols]
            else:
                onehot_enc[col] = encoded.columns

            # Add encoded data
            ds = xr.merge([ds, encoded.to_xarray()])

            # Locate the col inside the features dict, remove and replace.
            # This is kind of ugly but deals with the 2 level feature dict.
            if feat_group == 'dynamic':
                for source, source_features in self.features[feat_group].items():
                    if col in source_features:
                        self.features[feat_group][source].remove(col)
                        self.features[feat_group][source].extend(encoded.columns)
            else:
                self.features[feat_group].extend(encoded.columns)
                if col in self.features[feat_group]:
                    self.features[feat_group].remove(col)
                else:
                    print(f"{col} not found in {feat_group} features. Encoded as 0s.")

        return ds, onehot_enc

    def _bitmask_expansion(self, ds, feat_group, bitmask_enc: dict | None):
        if not bitmask_enc:
            bitmask_cols = self.cfg.get('bitmask_cols', {}).get(feat_group, [])
            if not bitmask_cols:
                return ds, None
            bitmask_enc = {k: None for k in bitmask_cols}

        for col, num_bits in bitmask_enc.items():
            if col in ds.data_vars:
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

                    new_vars[f"{col}_bit_{n}"] = xr.DataArray(data=bit_arr,
                                                              dims=['basin', 'date'],
                                                              coords={
                                                                  'basin': ds.basin,
                                                                  'date': ds.date
                                                              })
                # Remove the original categorical column
                ds = ds.drop_vars(col)
            else:
                # Create all-zero columns for missing bitmask columns
                if num_bits is None:
                    raise ValueError(f"Number of bits for {col} is not specified in the encoding.")
                for n in range(num_bits):
                    new_vars[f"{col}_bit_{n}"] = xr.DataArray(data=np.zeros((len(ds.basin), len(ds.date))),
                                                              dims=['basin', 'date'],
                                                              coords={
                                                                  'basin': ds.basin,
                                                                  'date': ds.date
                                                              })

            ds = xr.merge([ds, xr.Dataset(new_vars)])

            # Locate the col inside the dynamic features dict, remove and replace.
            # This is kind of ugly but deals with the 2 level feature dict.
            if feat_group == 'dynamic':
                for source, source_features in self.features['dynamic'].items():
                    if col in source_features:
                        self.features['dynamic'][source].remove(col)
                        self.features['dynamic'][source].extend(new_vars.keys())
            else:
                self.features[feat_group].extend(new_vars.keys())
                if col in self.features[feat_group]:
                    self.features[feat_group].remove(col)
                else:
                    print(f"{col} not found in {feat_group} features. Encoded as 0s.")

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
                training_ds = ds.sel(date=slice(None, self.cfg.get('split_time')), basin=self.train_basins)
            else:
                training_ds = ds

            # Initialize
            scale = {k: {'encoded': False, 'log_norm': False, 'offset': 0, 'scale': 1} for k in ds.data_vars}

            # Iterate over each variable in the dataset and calculate scaler
            for var in ds.data_vars:
                log_norm_cols = self.cfg.get('log_norm_cols', [])
                range_norm_cols = self.cfg.get('range_norm_cols', [])

                if var in encoding['encoded_columns']:
                    # One-hot encoded columns don't need normalization
                    scale[var]['encoded'] = True

                elif log_norm_cols is not None and var in log_norm_cols:
                    # Log normalization
                    scale[var]['log_norm'] = True
                    x = training_ds[var] + self.log_pad
                    scale[var]['offset'] = np.nanmean(np.log(x))

                elif range_norm_cols is not None and var in range_norm_cols:
                    # Min-max scaling
                    min_val = training_ds[var].min().values.item()
                    max_val = training_ds[var].max().values.item()
                    scale[var]['offset'] = min_val
                    scale[var]['scale'] = max_val - min_val

                else:
                    # Standard normalization
                    scale[var]['offset'] = training_ds[var].mean().values.item()
                    scale[var]['scale'] = training_ds[var].std().values.item()

        for var in set(ds.data_vars).intersection(scale.keys()):
            scl = scale[var]
            if scl['encoded']:
                continue
            elif scl['log_norm']:
                ds[var] = np.log(ds[var] + self.log_pad) - scl['offset']
            else:
                # Handle 0 variance here
                if scl['scale'] == 0:
                    ds[var] = (ds[var] - scl['offset'])
                else:
                    ds[var] = (ds[var] - scl['offset']) / scl['scale']

        return ds, scale

    def denormalize_target(self, y_normalized):
        """
        Denormalizes the target variable(s).
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
                y = y.at[..., i].set(jnp.exp(y_normalized[..., i] + offset) - self.log_pad)
            else:
                y = y.at[..., i].set(y_normalized[..., i] * scale + offset)
        return y

    def _date_batching(self, valid_date_mask):
        if self.data_subset in ['pre_train', 'train', 'test']:
            valid_target = (~np.isnan(self.x_d[self.targets_to_index])).to_array().any(dim=['variable', 'basin'])
        else:
            valid_target = True

        mask = valid_date_mask & valid_target
        valid_dates = self.x_d['date'][mask].values

        self.sequence_indices = valid_dates

    def _basin_date_batching(self, valid_date_mask):

        def valid_target(ds):
            return (~np.isnan(ds[self.targets_to_index])).to_array().any(dim='variable')

        def valid_obs(ds):
            all_features = list(itertools.chain(*self.features['dynamic'].values()))
            valid_mask_arr = (~np.isnan(ds[all_features])).to_array().values
            return valid_mask_arr.all(axis=0)

        indices = {}
        for basin in tqdm(self.basin_subset, disable=self.cfg['quiet'], desc="Updating Indices"):
            ds_basin = self.x_d.sel(basin=basin)

            # Create valid data indices for this basin
            if self.data_subset == 'pre_train':
                mask = valid_date_mask & valid_target(ds_basin) & valid_obs(ds_basin)
            elif self.data_subset == 'train':
                mask = valid_date_mask & valid_target(ds_basin)
            elif self.data_subset == 'test':
                mask = valid_date_mask & valid_target(ds_basin)
            else:
                mask = valid_date_mask
            indices[basin] = ds_basin['date'][mask].values

        # These are the indices that will be used for selecting sequences of data.
        basin_date_pairs = [(basin, date) for basin, dates in indices.items() for date in dates]
        self.sequence_indices = basin_date_pairs

    def _get_basin_date_split(self):
        # Get the list of basins we are going to use.
        # If specified, use those. Otherwise select it based on the data subset.
        if self.basin_subset is None:
            if self.data_subset in ['pre_train', 'train']:
                self.basin_subset = self.train_basins
            elif self.data_subset in ['test', 'predict']:
                self.basin_subset = self.test_basins
            elif self.data_subset == 'predict_all':
                self.basin_subset = self.all_basins

        # Get a boolean mask of dates that match our time splitting scheme
        # for the current data subset.
        if self.cfg.get('split_time'):
            # Select before or after split time based on data subset.
            if self.data_subset in ['pre_train', 'train']:
                date_mask = self.x_d['date'] <= self.cfg['split_time']
            elif self.data_subset in ['test', 'predict']:
                date_mask = self.x_d['date'] > self.cfg['split_time']
        else:
            # No time splitting between train and test.
            date_mask = True

        # Minimum date for sequenced data
        min_train_date = (np.datetime64(self.cfg['time_slice'].start) +
                          np.timedelta64(self.cfg['sequence_length'], 'D'))
        valid_sequence = self.x_d['date'] >= min_train_date

        # Return the combination of valid sequences and valid dates from subset.
        valid_dates = valid_sequence & date_mask

        return valid_dates

    def update_indices(self, data_subset: str, basin_subset: list = None):
        # Validate the data_subset choice
        data_subsets = ['pre_train', 'train', 'test', 'predict', 'predict_all']
        if data_subset not in data_subsets:
            raise ValueError(f"data_subset ({data_subset}) must be in ({data_subsets}) ")
        self.data_subset = data_subset

        # Set the basin subset. Cast as list if needed (sometimes we use a single basin).
        # If none, will default to the basins defined by the data subset.
        if basin_subset is not None:
            if isinstance(basin_subset, list):
                self.basin_subset = basin_subset
            else:
                self.basin_subset = [basin_subset]
        else:
            self.basin_subset = None

        # Get a list of target variables check when indexing data.
        exclude_target = self.cfg.get('exclude_target_from_index')
        if exclude_target is None:
            self.targets_to_index = self.target
        else:
            self.targets_to_index = [item for item in self.target if item not in exclude_target]

        valid_date_mask = self._get_basin_date_split()
        if self.graph_mode:
            self._date_batching(valid_date_mask)
        else:
            self._basin_date_batching(valid_date_mask)

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
        cfg_keys = [
            'data_dir', "time_series_dir", 'features', 'time_slice', 'split_time', 'add_rolling_means', 'log_norm_cols',
            'categorical_cols', 'bitmask_cols', 'range_norm_cols', 'clip_feature_range'
        ]
        data_config = {k: self.cfg.get(k) for k in cfg_keys}
        data_config['basins'] = sorted(self.all_basins)
        data_config['graph_matrix'] = self.graph_matrix
        """Generate a SHA256 hash for the contents of the dict."""
        hasher = hashlib.sha256()
        # Convert the dictionary to a sorted, consistent string representation
        dict_str = yaml.dump(data_config, sort_keys=True)
        hasher.update(dict_str.encode('utf-8'))
        return hasher.hexdigest()
