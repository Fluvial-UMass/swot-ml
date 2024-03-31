import pandas as pd
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt

def get_dataloaders(train_df, test_df, **kwargs):
    train = DataLoader(train_df, train=True, shuffle=True, **kwargs)
    test = DataLoader(test_df, train=False, scale=train.scale, **kwargs)
    return train, test

class DataLoader:
    def __init__(self,
                 df: pd.DataFrame,
                 features: list,
                 target: str,
                 batch_size: int,
                 sequence_length: int,
                 train: bool,
                 discharge_col: str = None,
                 scale: pd.Series = None,
                 shuffle: bool = False,
                 fill_and_weight: bool = False,
                 split_time: bool = False):
        """
        Initializes the DataLoader for batching and optionally shuffling data.
        """
        if train:
            df_norm, scale = _normalize_data(df, None, [target])
        elif not train and scale:
            df_norm, _ = _normalize_data(df, scale, [target])
        else:
            raise ValueError("Scale must be provided when not training.")
            
        self.xd = df_norm[features].copy().values
        self.y = df_norm[target].copy().values
        self.scale = scale

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.shuffle = shuffle
        self.fill_and_weight = fill_and_weight
        self.split_time = split_time
        
        if self.fill_and_weight and self.split_time:
            raise ValueError("Both fill_and_weight and split_time cannot be True at the same time.")

        self.w = None
        if self.fill_and_weight:
            intervals = _distance_to_closest_obs(self.xd)
            self.w = 1 / intervals**2 # Weighting function could be tuned
            self.xd = _fill_nan_obs(self.xd)

        self.dt = None
        if self.split_time:
            self.dt = _distance_to_closest_obs(self.xd)
            self.xd = _fill_nan_obs(self.xd)

        # properties for loss funcs
        if train:
            self.unscaled_q = df[discharge_col].copy().values
        
        if train:
            self.indices = np.where(~np.isnan(self.y[:-sequence_length+1]))[0]
            self.indices = self.indices[self.indices >= sequence_length]
        else:
            self.indices = np.arange(sequence_length, self.xd.shape[0])

        self.dataset_size = np.sum(self.indices != 0)  # Number of valid samples


    def __iter__(self) -> tuple[np.ndarray, ...]:
        """
        Iterates over the data, yielding batches

        Yields:
            Tuple[np.ndarray, ...]: A tuple containing batch arrays for ids_batch, xd_batch, y_batch,
                and optionally w_batch, and dt_batch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, self.dataset_size, self.batch_size):
            # Create a 2D index array for slicing
            end = min(start + self.batch_size, self.dataset_size)
            batch_indices = self.indices[start:end]
            batch_sequences = np.array([np.arange(i - self.sequence_length + 1, i + 1) for i in batch_indices])

            batch_dict = {
                'ids': batch_indices,
                'xd': self.xd[batch_sequences],
                'y': self.y[batch_sequences]
            }
    
            if self.w is not None:
                batch_dict['w'] = self.w[batch_sequences]
            if self.dt is not None:
                batch_dict['dt'] = self.dt[batch_sequences]
            if self.train:
                batch_dict['unscaled_q'] = self.unscaled_q[batch_sequences]
    
            yield batch_dict


    def __len__(self):
        """
        Returns the number of batches the DataLoader will generate.
        """
        if self.shuffle:
            return np.inf
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

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