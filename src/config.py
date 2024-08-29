import yaml
import numpy as np
import itertools
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
    cfg['clip_feature_range'] = {
        key: process_clip_range(value)
        for key, value in cfg['clip_feature_range'].items()
    }

    # Create the target_weights list from the dict
    cfg['step_kwargs']['target_weights'] = [
        cfg['step_kwargs']['target_weights'].get(target, 1) 
        for target in cfg['features']['target']
    ]

    if cfg['step_kwargs'].get('agreement_weight', 0) != 0:
        required_targets = {'ssc','flux','usgs_q'}
        if not required_targets.issubset(set(cfg['features']['target'])):
            raise ValueError("Must predict at least ssc, flux, and usgs_q when using flux agreement regularization.")

    cfg['model'] = cfg['model'].lower()
    
    return cfg

def process_clip_range(range_list):
    if len(range_list) != 2:
        raise ValueError("Each range must have exactly 2 elements")
    
    lower = -np.inf if range_list[0] is None else range_list[0]
    upper = np.inf if range_list[1] is None else range_list[1]
    
    return [lower, upper]

def get_grid_update_tuples(cfg):
    param_dict = cfg['param_search_dict']
    key_list = []
    value_list = []
    for k1, v1 in param_dict.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                key_list.append((k1, k2))
                value_list.append(v2)
        elif isinstance(v1, list):
            key_list.append(k1)
            value_list.append(v1)
        else:
            raise ValueError('param_search_dict must be a dict that contains only lists and dicts of lists') 

    # Shuffle the hyperparam grid and select one based on idx.
    # This allows for both random search or grid search based on range of ids passed.  
    param_grid_list = list(itertools.product(*value_list))
    rng = np.random.default_rng(42) #Do not change!
    rng.shuffle(param_grid_list)

    return key_list, param_grid_list

def update_cfg_from_grid(cfg:dict, idx:int):
    key_list, param_grid_list = get_grid_update_tuples(cfg)
    updates = param_grid_list[idx]
    
    # Insert the updates into the config
    for k, v in zip(key_list, updates):
        if isinstance(k, tuple):
            if len(k) != 2:
                raise RuntimeError("tuple keys in 'param_search_dict' must have length 2")
            cfg[k[0]][k[1]] = v
        else:
            cfg[k] = v

    return cfg


def set_model_data_args(cfg, dataset):
    target = dataset.target
    target = target if isinstance(target, list) else list(target)
    cfg['model_args']['target'] = target
    
    model_name = cfg['model'].lower()
    if model_name in ['flexible_hybrid', 'hybrid']:
        cfg['model_args']['dynamic_sizes'] = {k: len(v) for k, v in dataset.dynamic_features.items()}
        cfg['model_args']['static_size'] = len(dataset.static_features)

    # if model_name in ["eatransformer", "fusion"]:
    #     cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
    #     cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
    #     cfg['model_args']['static_in_size'] = len(dataset.static_features)
    #     cfg['model_args']['seq_length'] = cfg['sequence_length']
        
    # elif model_name in ['tft', 'tft_mha']:
    #     dynamic_sizes = {'x_dd': len(dataset.daily_features)}
    #     if dataset.irregular_features:
    #         dynamic_sizes['x_di'] = len(dataset.irregular_features)
            
    #     cfg['model_args']['dynamic_sizes'] = dynamic_sizes
    #     cfg['model_args']['static_size'] = len(dataset.static_features)

    # elif model_name in ['taplstm','hybrid']:
    #     cfg['model_args']['daily_in_size'] = len(dataset.daily_features)
    #     cfg['model_args']['irregular_in_size'] = len(dataset.irregular_features)
    #     cfg['model_args']['static_in_size'] = len(dataset.static_features)

    # elif model_name == 'ealstm':
    #     cfg['model_args']['dynamic_in_size'] = len(dataset.daily_features)
    #     cfg['model_args']['static_in_size'] = len(dataset.static_features)

    # elif model_name == 'tealstm':
    #     cfg['model_args']['dynamic_in_size'] = len(dataset.irregular_features)
    #     cfg['model_args']['static_in_size'] = len(dataset.static_features)

    else:
        raise ValueError(f"{model_name}: Invalid model name")
    
    return cfg