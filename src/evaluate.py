import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

@eqx.filter_jit
def _predict_map(model, batch, keys):
    return jax.vmap(model)(batch,keys)

def predict(model, dataloader, *, seed=0, denormalize=True):
    key = jax.random.PRNGKey(seed)
    
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    
    basins = []
    dates = []
    y = []
    y_hat = []
    for basin, date, batch in tqdm(dataloader):
        keys = jax.random.split(key, len(basin)+1)
        key = keys[0]
        batch_keys = keys[1:] 
        pred = _predict_map(model, batch, batch_keys)
        
        basins.extend(basin)
        dates.extend(date)
        y.extend(batch['y'][:,-1])
        y_hat.extend(pred)
        
    # Create a dataframe with multi-index
    multi_index = pd.MultiIndex.from_arrays([basins,dates],names=['basin','date'])
    y = np.array(y).flatten()
    y_hat = np.array(y_hat).flatten()
    if denormalize:
        y = dataloader.dataset.denormalize_target(y)
        y_hat = dataloader.dataset.denormalize_target(y_hat)
    results = pd.DataFrame({'obs':  y, 'pred': y_hat}, index=multi_index)

    return results


def get_all_metrics(y, y_hat):
    metrics = {
        'mae': calc_mae(y, y_hat),
        'mse': calc_mse(y, y_hat),
        'rmse': calc_rmse(y, y_hat),
        'kge': calc_kge(y, y_hat),
        'nse': calc_nse(y, y_hat)}
    return metrics

def _mask_nan(y, y_hat):
    mask = (~np.isnan(y)) & (~np.isnan(y_hat))
    return y[mask], y_hat[mask]

def calc_mae(y, y_hat):
    return np.nanmean(np.abs(y - y_hat))

def calc_mse(y, y_hat):
    return np.nanmean((y - y_hat)**2)

def calc_rmse(y, y_hat):
    return np.sqrt(calc_mse(y, y_hat))

def calc_kge(y, y_hat):
    y, y_hat = _mask_nan(y, y_hat)
    
    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(y, y_hat)[0, 1]
    
    # Calculate mean and standard deviation
    mean_y = np.mean(y)
    mean_y_hat = np.mean(y_hat)
    std_y = np.std(y)
    std_y_hat = np.std(y_hat)
    
    # Calculate KGE
    kge_value = 1 - np.sqrt((correlation - 1)**2 + (std_y_hat/std_y - 1)**2 + (mean_y_hat/mean_y - 1)**2)
    
    return kge_value

def calc_nse(y, y_hat):
    y, y_hat = _mask_nan(y,y_hat)

    denominator = ((y_hat - y_hat.mean())**2).sum()
    numerator = ((y - y_hat)**2).sum()

    value = 1 - numerator / denominator

    return float(value)