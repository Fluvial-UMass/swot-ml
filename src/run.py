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

from utils import read_config
from data import TAPDataset, TAPDataLoader
from models import TAPLSTM, EALSTM, TEALSTM, HybridModel
from train import Trainer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to the configuration file as an argument")
        
    config_path = Path(sys.argv[1]).resolve()
    config, config_str = read_config(config_path)
    
    quiet = config.get('quiet',True)

    dataset = TAPDataset(quiet=quiet, **config['data_args']) 
    dataloader = TAPDataLoader(dataset, **config['loader_args'])
    
    model_name = config['model'].lower()
    if model_name == 'taplstm':
        model = TAPLSTM
    elif model_name == 'ealstm':
        model = EALSTM
    elif model_name == 'tealstm':
        model = TEALSTM
    elif model_name == 'hybrid':
        model = HybridModel
    else:
        raise ValueError("Please provide a valid model name (taplstm, ealstm, tealstm, hybrid)")
        
    config['trainer_args']['model_func'] = model
    config['trainer_args']['dataloader'] = dataloader
    
    trainer = Trainer(quiet=quiet, 
                      config_dir=config_path.parent,
                      config_str=config_str,
                      **config['trainer_args'])
    trainer.start_training()