import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

@eqx.filter_jit
def _predict_map(model, batch, keys):
    return jax.vmap(model)(batch, keys)

def predict(model, dataloader, *, seed=0, denormalize=True, return_dt=False):
    key = jax.random.PRNGKey(seed)
    
    # Set model to inference mode (no dropout)
    # model = eqx.nn.inference_mode(model)
    
    basins = []
    dates = []
    y = []
    y_hat = []
    dt = []
    for basin, date, batch in tqdm(dataloader):
        keys = jax.random.split(key, len(basin)+1)
        key = keys[0]
        batch_keys = keys[1:] 
        pred = _predict_map(model, batch, batch_keys)
        
        basins.extend(basin)
        dates.extend(date)
        y.extend(batch['y'][:,-1])
        y_hat.extend(pred)
            
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

        
    # Create a dataframe with multi-index
    multi_index = pd.MultiIndex.from_arrays([basins,dates],names=['basin','date'])
    y = np.array(y).flatten()
    y_hat = np.array(y_hat).flatten()
    if denormalize:
        y = dataloader.dataset.denormalize_target(y)
        y_hat = dataloader.dataset.denormalize_target(y_hat)
    results = pd.DataFrame({'obs':  y, 'pred': y_hat}, index=multi_index)
    
    if return_dt:
        results['dt'] = dt
        
    metrics = get_all_metrics(y, y_hat)
    for key, value in metrics.items():
        print(f"{key}: {value:0.4f}")

    return results, metrics


def get_all_metrics(y, y_hat):
    metrics = {
        'nBias': calc_nbias(y, y_hat),
        'rRMSE': calc_rrmse(y, y_hat),
        'KGE': calc_kge(y, y_hat),
        'NSE': calc_nse(y, y_hat),
        'lNSE': calc_lnse(y, y_hat),
        'Agreement': calc_agreement(y, y_hat)}
    return metrics

def mask_nan(func):
    def wrapper(y, y_hat, *args, **kwargs):
        mask = (~np.isnan(y)) & (~np.isnan(y_hat))
        y_masked = y[mask]
        y_hat_masked = y_hat[mask]
        return func(y_masked, y_hat_masked, *args, **kwargs)
    return wrapper

@mask_nan
def calc_nbias(y, y_hat):
    norm_err = (y - y_hat) / np.mean(y)
    return np.nanmean(norm_err)

@mask_nan
def calc_rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))
    
@mask_nan
def calc_rrmse(y, y_hat):
    rmse = calc_rmse(y, y_hat)
    return rmse/np.mean(y_hat)*100
 
@mask_nan
def calc_kge(y, y_hat):
    correlation = np.corrcoef(y, y_hat)[0, 1]
    mean_y = np.mean(y)
    mean_y_hat = np.mean(y_hat)
    std_y = np.std(y)
    std_y_hat = np.std(y_hat)
    kge = 1 - np.sqrt((correlation - 1)**2 + (std_y_hat/std_y - 1)**2 + (mean_y_hat/mean_y - 1)**2) 
    return kge

@mask_nan
def calc_nse(y, y_hat):
    denominator = ((y_hat - y_hat.mean())**2).sum()
    numerator = ((y - y_hat)**2).sum()
    nse = 1 - (numerator / denominator)
    return nse

@mask_nan
def calc_lnse(y, y_hat):
    log_y = np.log(y)
    log_yhat = np.log(y_hat)
    return calc_nse(log_y, log_yhat)

@mask_nan
def calc_agreement(y, y_hat):
    """ https://www.nature.com/articles/srep19401 """
    corr = np.corrcoef(y, y_hat)[0,1]
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * np.abs(np.mean((y-np.mean(y))*(y_hat-np.mean(y_hat))))

    numerator = np.mean((y - y_hat)**2)
    denominator = np.var(y) + np.var(y_hat) + (np.mean(y) - np.mean(y_hat))**2 + kappa
    return 1 - (numerator / denominator)
            
    

