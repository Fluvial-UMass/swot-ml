# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import pandas as pd
import optax
from pathlib import Path
from tqdm import trange, tqdm

from config import read_config
from data import TAPDataset, TAPDataLoader
from train import Trainer
from evaluate import predict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to the configuration file as an argument")
        
    cfg_path = Path(sys.argv[1]).resolve()
    cfg, cfg_str = read_config(cfg_path)
    
    dataset = TAPDataset(cfg) 
    dataloader = TAPDataLoader(cfg, dataset)
    
    # Model config is a bit more dynamic
    model_name = cfg['model'].lower()
    if model_name == "eatransformer":
        cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
        cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
        cfg['model_args']['seq_length'] = cfg['sequence_length']
    elif model_name == 'taplstm':
        cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
        cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    elif model_name == 'ealstm':
        cfg['model_args']['dynamic_in_size'] = len(dataset.daily_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    elif model_name == 'tealstm':
        cfg['model_args']['dynamic_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
    else:
        raise ValueError("Please provide a valid model name (eatransformer, taplstm, ealstm, tealstm)")
        
    trainer = Trainer(cfg, dataloader, log_parent=cfg_path.parent)
    trainer.start_training()
    
    cfg['data_subset'] = 'test'
    dataloader = TAPDataLoader(cfg, dataset)
    results = predict(trainer.model, dataloader, seed=0, denormalize=True)
    results.to_pickle(trainer.log_dir / "test.pkl")
