import sys
import yaml
import numpy as np
from pathlib import Path
import optax

def read_config(yml_path):
    with open(yml_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    cfg = format_config(raw_cfg)
    cfg_str = yaml.dump(raw_cfg, default_flow_style=False)
    
    return cfg, cfg_str

def format_config(cfg):
    # Ensures some data types and that some keys exist
    data_dir = Path(cfg['data_args']['data_dir'])
    cfg['data_args']['data_dir'] = data_dir
    cfg['data_args']['basin_file'] = data_dir / cfg['data_args']['basin_file']
    cfg['data_args']['time_slice'] = slice(*cfg['data_args']['time_slice'])
    cfg['data_args']['split_time'] = np.datetime64(cfg['data_args']['split_time'])
    
    cfg['model_args']['out_size'] = 1

    cfg['trainer_args'] = {}
    lr_schedule = optax.exponential_decay(cfg['learning_rate'], 
                                          cfg['num_epochs'], 
                                          cfg['decay_rate'])
    cfg['trainer_args']['lr_schedule'] = lr_schedule
    cfg['trainer_args']['num_epochs'] = cfg['num_epochs']
    cfg['trainer_args']['max_grad_norm'] = cfg.get('max_grad_norm', None)
    cfg['trainer_args']['l2_weight'] = cfg.get('l2_weight', None)
    
    return cfg