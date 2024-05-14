# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import numpy as np
import optax
from pathlib import Path
from tqdm import trange, tqdm

from .config import read_config
from .data import TAPDataset, TAPDataLoader
from .models import TAPLSTM, EALSTM, TEALSTM
from .train import Trainer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to the configuration file as an argument")
        
    cfg_path = Path(sys.argv[1]).resolve()
    cfg, cfg_str = read_config(cfg_path)
    dataset = TAPDataset(quiet=quiet, **cfg['data_args']) 
    dataloader = TAPDataLoader(dataset, **cfg['loader_args'])
    
    # Model config is a bit more dynamic
    model_name = cfg['model'].lower()
    if model_name == 'taplstm':
        model_fn = TAPLSTM
        cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
        cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    elif model_name == 'ealstm':
        model_fn = EALSTM
        cfg['model_args']['dynamic_in_size'] = len(dataset.daily_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    elif model_name == 'tealstm':
        model_fn = TEALSTM
        cfg['model_args']['dynamic_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    else:
        raise ValueError("Please provide a valid model name (taplstm, ealstm, tealstm)")

    
    trainer = Trainer(cfg, model_fn, dataloader, log_parent=cfg_path.parent)
    trainer.start_training()
    
    
    
    
    
