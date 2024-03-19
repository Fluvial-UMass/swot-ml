import jax.numpy as jnp
import numpy as np


class DataLoader:
    def __init__(self, 
                 arrays: tuple[np.ndarray, ...], 
                 batch_size: int, 
                 sequence_length: int, 
                 shuffle: bool = True):
        """
        Initializes the DataLoader for batching and optionally shuffling data.

        Args:
            arrays (Tuple[np.ndarray, ...]): A tuple of NumPy arrays to be batched and sequenced. Each array should have
                the same number of rows (samples).
            batch_size (int): The size of each batch.
            sequence_length (int): The length of each sequence.
            shuffle (bool, optional): Whether to shuffle the data before batching. Defaults to True.
        """
        self.arrays = arrays
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.dataset_size = arrays[0].shape[0] - sequence_length + 1
        assert all(array.shape[0] == self.arrays[0].shape[0] for array in self.arrays), "All arrays must have the same number of rows."

    def __iter__(self) -> tuple[np.ndarray, ...]:
        """
        Iterates over the data, yielding batches of sequences.

        Yields:
            Tuple[np.ndarray, ...]: A tuple of batches, one for each array in 'arrays'. Each batch is a NumPy array
            with dimensions [batch_size, sequence_length, feature_size], where feature_size is the number of features in the
            corresponding input array.
        """
        indices = np.arange(self.dataset_size)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.dataset_size, self.batch_size):
            end = min(start + self.batch_size, self.dataset_size)
            batch_indices = indices[start:end]
            yield tuple(np.array([array[i:i + self.sequence_length] for i in batch_indices]) for array in self.arrays)

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
    

def sequencer(array, sequence_length):
    """
    Creates sequences from an array.

    Args:
        array (jnp.array): The data array to be converted into sequences.
        sequence_length (int): The length of each sequence.

    Returns:
        jnp.array: An array of sequences.
    """
    # dataset_size = array.shape[0] - sequence_length + 1
    # return jnp.array([array[i:i + sequence_length] for i in range(dataset_size)])
    dataset_size = array.shape[0] - sequence_length + 1
    return jnp.array([array[i:i + sequence_length] for i in range(dataset_size - 1, -1, -1)])