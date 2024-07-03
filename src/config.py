import yaml
import numpy as np
from pathlib import Path

def read_yml(yml_path):
    if not isinstance(yml_path, Path):
        yml_path = Path(yml_path)

    with open(yml_path, 'r') as f:
        yml = yaml.safe_load(f)
    return yml

def read_config(yml_path):
    raw_cfg = read_yml(yml_path)
    cfg = format_config(raw_cfg)
    cfg['cfg_path'] = yml_path
    cfg_str = yaml.dump(raw_cfg, default_flow_style=False)
    
    return cfg, cfg_str

def format_config(cfg):
    # Ensures some data types and that some keys exist
    data_dir = Path(cfg['data_dir']).resolve()
    cfg['data_dir'] = data_dir

    cfg['time_slice'] = slice(*cfg['time_slice'])
    cfg['split_time'] = np.datetime64(cfg['split_time'])

    cfg['log_norm_cols'] = cfg.get('log_norm_cols',[])
    
    cfg['model'] = cfg['model'].lower()
    
    return cfg


def set_model_data_args(cfg, dataset):
    cfg['model_args']['out_size'] = len(dataset.target)
    
    model_name = cfg['model'].lower()
    if model_name in ["eatransformer", "hybrid", "fusion"]:
        cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
        cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
        cfg['model_args']['static_in_size'] = len(dataset.static_features)
        cfg['model_args']['seq_length'] = cfg['sequence_length']
    elif model_name == 'tft':
        dynamic_sizes = {'x_dd': len(dataset.daily_features), 
                         'x_di': len(dataset.irregular_features)}
        
        cfg['model_args']['dynamic_sizes'] = dynamic_sizes
        cfg['model_args']['static_size'] = len(dataset.static_features)
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
        raise ValueError(f"{model_name}: Invalid model name")
    
    return cfg