import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

@eqx.filter_jit
def _model_map(model, batch, keys):
    return jax.vmap(model)(batch, keys)

def predict(model, dataloader, *, seed=0, quiet=False, denormalize=True, return_dt=False):
    key = jax.random.PRNGKey(seed)
    
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    
    basins = []
    dates = []
    y = []
    y_hat = []
    dt = []
    for basin, date, batch in tqdm(dataloader, disable=quiet):
        keys = jax.random.split(key, len(basin)+1)
        key = keys[0]
        batch_keys = keys[1:] 
        
        pred = _model_map(model, batch, batch_keys)
        
        basins.extend(basin)
        dates.extend(date)
        y.append(batch['y'][:,-1,:])
        y_hat.append(pred)
            
        if return_dt:
            valid_mask = ~np.isnan(batch['x_di'][:, :, 0])  # Simplifying to the first feature, since all features align

            indices = np.arange(valid_mask.shape[1])
            valid_indices_arr = np.full_like(valid_mask, -1, dtype=float)  # Start with -1 (indicating no valid data yet)

            for b in range(valid_mask.shape[0]):
                valid_indices = np.where(valid_mask[b])[0]
                valid_indices_arr[b, valid_indices] = indices[valid_indices]

            last_valid_index = np.maximum.accumulate(valid_indices_arr, axis=1) 
            last_valid_index[last_valid_index == -1] = np.nan
            batch_dt = last_valid_index - indices
            dt.extend(batch_dt[:,-1])  
    
    
    y_arr = np.concatenate(y)
    y_hat_arr = np.concatenate(y_hat)
    if denormalize:
        y_arr = dataloader.dataset.denormalize_target(y_arr)
        y_hat_arr = dataloader.dataset.denormalize_target(y_hat_arr)
    
    # Place the data into a dataframe with multilevel indices.
    datetime_index = pd.MultiIndex.from_arrays([basins,dates],names=['basin','date'])
    column_index = pd.MultiIndex.from_product([['obs', 'pred'], dataloader.dataset.target],
                                              names=['Type', 'Feature'])
    data = np.hstack((y_arr, y_hat_arr))
    results = pd.DataFrame(data, index=datetime_index, columns=column_index)

    return results
