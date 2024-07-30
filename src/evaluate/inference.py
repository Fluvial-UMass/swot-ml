import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

@eqx.filter_jit
def _model_map(model, batch, keys):
    return jax.vmap(model)(batch, keys)

# Calculates all dts through batch.
def _calc_dt_array(batch):
    # Simplifying to the first feature, since all features align
    valid_mask = ~np.isnan(batch['x_di'][:, :, 0])  
    # Initialize with -1 (indicating no valid data yet)
    indices = np.arange(valid_mask.shape[1])
    valid_indices_arr = np.full_like(valid_mask, -1, dtype=float)  

    for b in range(valid_mask.shape[0]):
        valid_indices = np.where(valid_mask[b])[0]
        valid_indices_arr[b, valid_indices] = indices[valid_indices]

    last_valid_index = np.maximum.accumulate(valid_indices_arr, axis=1) 
    last_valid_index[last_valid_index == -1] = np.nan
    batch_dt_arr = last_valid_index - indices
    
    return batch_dt_arr

# Only calculates dt for the final element in sequence (prediciton)
def _calc_last_dt(batch):
    # Simplifying to the first feature, since all features align
    valid_mask = ~np.isnan(batch['x_di'][:, :, 0])  
    # Find the last valid index for each sequence in the batch
    batch_dt_last = np.argmax(valid_mask[:, ::-1], axis=1)

    # Logic fails if there are no obs, so we handle those cases here.
    seq_length = valid_mask.shape[1]
    all_nan_mask = ~np.any(valid_mask, axis=1)
    batch_dt_last[all_nan_mask] = seq_length+1

    return batch_dt_last

def model_iterate(model, dataloader, return_dt=False, quiet=False, denormalize=True):
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    # Dummy batch keys (only used for dropout, which is off).
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, dataloader.batch_size)

    inference_mode = dataloader.dataset.inference_mode
    for basin, date, batch in tqdm(dataloader, disable=quiet):
        y_pred = _model_map(model, batch, keys)
        if not inference_mode: 
            y = batch['y'][:,-1,:]

        if denormalize:
            if not inference_mode: 
                y = dataloader.dataset.denormalize_target(y)
            y_pred = dataloader.dataset.denormalize_target(y_pred)
        
        out_tuple = (basin, date, y_pred)
        if not inference_mode:
            out_tuple += (y,)
        if return_dt:
            dt = _calc_last_dt(batch)
            out_tuple += (dt,)
        yield out_tuple

def predict(model, dataloader, *, return_dt=False, quiet=False, denormalize=True):
    inference_mode = dataloader.dataset.inference_mode
    
    basins = []
    dates = []
    y_hat_list = []
    if not inference_mode: y_list = []
    if return_dt: dt_list = []

    for iter_out in model_iterate(model, dataloader, return_dt, quiet, denormalize):
        basins.extend(iter_out[0])
        dates.extend(iter_out[1])
        y_hat_list.append(iter_out[2])
        if not inference_mode: y_list.append(iter_out[3])
        if return_dt: dt_list.append(iter_out[4])
    
    y_hat_arr = np.concatenate(y_hat_list)
    if not inference_mode:
        y_arr = np.concatenate(y_list)
        data = np.hstack((y_arr, y_hat_arr))
        cols = ['obs', 'pred']
    else:
        data = y_hat_arr
        cols = ['pred']

     # Place the data into a dataframe with multilevel indices.
    datetime_index = pd.MultiIndex.from_arrays([basins,dates],names=['basin','date'])
    column_index = pd.MultiIndex.from_product([cols, dataloader.dataset.target],
                                              names=['Type', 'Feature'])

    results = pd.DataFrame(data, index=datetime_index, columns=column_index)

    if return_dt:
        dt_arr = np.concatenate(dt_list)
        dt_index = pd.MultiIndex.from_product([['metadata'], ['dt']], names=['Type', 'Feature'])
        results[dt_index] = dt_arr

    return results
