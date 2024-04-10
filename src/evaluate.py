import equinox as eqx
import jax
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

@eqx.filter_jit
def _predict_map(model, batch):
    return jax.vmap(model)(batch)

def predict(model, dataloader, basin_subset=[]):
    if not isinstance(basin_subset,list):
        basin_subset = [basin_subset]
    
    model = eqx.tree_at(lambda m: m.dropout.inference, model, True)
    dataloader.train = False
    dataloader.basin_subset = basin_subset
    
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
    y = dataloader.denormalize_target(np.array(y).flatten())
    y_hat = dataloader.denormalize_target(np.array(y_hat).flatten())
    results = pd.DataFrame({'obs':  y, 'pred': y_hat}, index=multi_index)

    return results