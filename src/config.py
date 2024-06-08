import sys
import yaml
import numpy as np
from pathlib import Path

def read_config(yml_path):
    with open(yml_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    raw_cfg['cfg_path'] = yml_path
    cfg = format_config(raw_cfg)
    
    cfg_str = yaml.dump(raw_cfg, default_flow_style=False)
    
    return cfg, cfg_str

def format_config(cfg):
    # Ensures some data types and that some keys exist
    data_dir = Path(cfg['data_dir']).resolve()
    cfg['data_dir'] = data_dir
    cfg['basin_file'] = data_dir / cfg['basin_file']
    cfg['time_slice'] = slice(*cfg['time_slice'])
    cfg['split_time'] = np.datetime64(cfg['split_time'])

    cfg['log_norm_cols'] = cfg.get('log_norm_cols',[])
    
    cfg['model'] = cfg['model'].lower()
    
    return cfg


def set_model_data_args(cfg, dataset):
    cfg['model_args']['out_size'] = len(dataset.target)
    
    model_name = cfg['model'].lower()
    if model_name in ["eatransformer", "hybrid"]:
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
        raise ValueError("Invalid model name")
        
    return cfg