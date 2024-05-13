import sys
import yaml
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import optax

from models import TAPLSTM, EALSTM, LSTM

def read_config(yml_path):
    with open(yml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg_str = yaml.dump(cfg, default_flow_style=False)

    data_dir = Path(cfg['data_args']['data_dir'])
    cfg['data_args']['data_dir'] = data_dir
    cfg['data_args']['basin_file'] = data_dir / cfg['data_args']['basin_file']
    cfg['data_args']['time_slice'] = slice(*cfg['data_args']['time_slice'])
    cfg['data_args']['split_time'] = np.datetime64(cfg['data_args']['split_time'])

    cfg['model_args']['daily_in_size'] = len(cfg['data_args']['features']['daily'])
    cfg['model_args']['irregular_in_size'] = len(cfg['data_args']['features']['irregular'])
    cfg['model_args']['static_in_size'] = len(cfg['data_args']['features']['static'])
    cfg['model_args']['out_size'] = 1

    cfg['trainer_args'] = {}
    lr_schedule = optax.exponential_decay(cfg['learning_rate'], 
                                          cfg['num_epochs'], 
                                          cfg['decay_rate'])
    cfg['trainer_args']['lr_schedule'] = lr_schedule
    cfg['trainer_args']['model_args'] = cfg['model_args']
    cfg['trainer_args']['num_epochs'] = cfg['num_epochs']
    cfg['trainer_args']['max_grad_norm'] = cfg.get('max_grad_norm', None)
    cfg['trainer_args']['l2_weight'] = cfg.get('l2_weight', None)
    
    return cfg, cfg_str

class NoOpTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __next__(self):
        raise StopIteration()

    def __getattr__(self, item):
        # Any method call (like set_description or set_postfix_str) gets a no-op lambda
        return lambda *args, **kwargs: None

def smart_tqdm(iterable, quiet:bool, *args, **kwargs):
    if quiet:
        # Return an iterable that ignores all method calls
        return NoOpTqdm(iterable, *args, **kwargs)  
    else:
        return tqdm(iterable, *args, **kwargs)