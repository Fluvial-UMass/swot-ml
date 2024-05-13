import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

@eqx.filter_jit
def _predict_map(model, batch):
    return jax.vmap(model)(batch)

def predict(model, dataloader, denormalize=True):

    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    
    basins = []
    dates = []
    y = []
    y_hat = []
    for basin, date, batch in tqdm(dataloader):
        basins.extend(basin)
        dates.extend(date)
        y.extend(batch['y'][:,-1])
        y_hat.extend(_predict_map(model,batch))
        
    # Create a dataframe with multi-index
    multi_index = pd.MultiIndex.from_arrays([basins,dates],names=['basin','date'])
    y = np.array(y).flatten()
    y_hat = np.array(y_hat).flatten()
    if denormalize:
        y = dataloader.dataset.denormalize_target(y)
        y_hat = dataloader.dataset.denormalize_target(y_hat)
    results = pd.DataFrame({'obs':  y, 'pred': y_hat}, index=multi_index)

    return results