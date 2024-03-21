import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt

class DataLoader:
    def __init__(self, 
                 arrays: tuple[np.ndarray, ...], 
                 batch_size: int, 
                 sequence_length: int,
                 train: bool,
                 fill_and_weight: bool = False,
                 shuffle: bool = True):
        """
        Initializes the DataLoader for batching and optionally shuffling data.

        Args:
            arrays (Tuple[np.ndarray, ...]): A tuple of NumPy arrays to be batched and sequenced. Each array should have
                the same number of rows (samples).
            batch_size (int): The size of each batch.
            sequence_length (int): The length of each sequence.
            train (bool): If true, only returns batches with a non-nan target value. 
            fill_and_weight (bool, optional): If true, interpolates feature data and calculates observation weights. Defaults to True. 
            shuffle (bool, optional): Whether to shuffle the data before batching. Defaults to True.
        """
        self.arrays = arrays
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.fill_and_weight = fill_and_weight
        self.shuffle = shuffle

        if train: 
            self.indices =  np.where(~np.isnan(arrays[1][:-sequence_length+1]))[0]
            self.indices = self.indices[self.indices >= sequence_length]
        else: 
            self.indices = np.arange(sequence_length, arrays[0].shape[0])
        self.dataset_size = np.sum(self.indices != 0) # Number of valid samples
        
        assert all(array.shape[0] == self.arrays[0].shape[0] for array in self.arrays), "All arrays must have the same number of rows."

    def __iter__(self) -> tuple[np.ndarray, ...]:
        """
        Iterates over the data, yielding batches of sequences and corresponding weights.

        Yields:
            Tuple[np.ndarray, ...]: A tuple of batches, one for each array in 'arrays', and an additional batch of
            SIA weights if 'sia_weights' is True. Each batch is a NumPy array with dimensions [batch_size, sequence_length, feature_size],
            where feature_size is the number of features in the corresponding input array.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, self.dataset_size, self.batch_size):
            end = min(start + self.batch_size, self.dataset_size)
            batch_indices = self.indices[start:end]
            
            # batches = []
            # for array in self.arrays:
            #     batch_array = []
            #     for i in batch_indices:
            #         batch_array.append(array[i-self.sequence_length+1:i+1])
            #     batches.append(np.array(batch_array))
            # batches = tuple(batches)
            batches = tuple(np.array([array[i-self.sequence_length+1:i+1] for i in batch_indices]) for array in self.arrays)
            
            if self.fill_and_weight:
                intervals = distance_to_closest_obs(batches[0])
                weights = 1 / (1+intervals)**2 #Quick weight stand in. Probably needs tuning?
                filled = fill_nan_obs(batches[0])
                yield (batch_indices, filled, weights, batches[1])
            else:
                yield (batch_indices,*batches)

    def __len__(self):
        """
        Returns the number of batches the DataLoader will generate.
        """
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

def distance_to_closest_obs(arr):
    nan_mask = np.isnan(arr)
    distances = np.full(arr.shape, np.inf) # All nan columns will be inf. Weights -> 0
    
    for batch_idx in range(arr.shape[0]):
        for feature_idx in range(arr.shape[2]):
            # Calculate the distance transform for the column
            distances[batch_idx,:,feature_idx] = distance_transform_edt(nan_mask[batch_idx,:,feature_idx])
    return distances
    
def fill_nan_obs(arr):
    for batch_idx in range(arr.shape[0]):
        for feature_idx in range(arr.shape[2]):
            is_nan = jnp.isnan(arr[batch_idx,:,feature_idx])
            if np.sum(~is_nan)==0:
                # Can't interpolate without observations, but distances will be inf and weights 0.
                arr[batch_idx,:,feature_idx][is_nan] = 0
                continue
            #Interpolate based on closest measurement. 
            arr[batch_idx,is_nan,feature_idx] = np.interp(np.flatnonzero(is_nan), 
                                                        np.flatnonzero(~is_nan), 
                                                        arr[batch_idx,~is_nan,feature_idx])
    return arr


def normalize_data(data: np.array, scale: dict[str, float] = None):
    """
    Normalize the input data using the provided scale or calculate the scale if not provided.

    Args:
        data (np.array): The input data to be normalized.
        scale (Dict[str, float], optional): A dictionary containing the mean ('mean') and standard deviation ('std')
            to use for normalization. If not provided, the mean and standard deviation will be calculated from the data.

    Returns:
        np.array: The normalized data.
        Dict[str, float]: A dictionary containing the mean ('mean') and standard deviation ('std') used for normalization.
    """
    if scale is None:
        scale = {}
        scale['mean'] = np.mean(data, axis=0)
        scale['std'] = np.std(data, axis=0)
    normalized_data = (data - scale['mean']) / scale['std']
    return normalized_data, scale       