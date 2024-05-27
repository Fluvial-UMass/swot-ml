import sys
import yaml
import numpy as np
from pathlib import Path

def read_config(yml_path):
    with open(yml_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
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
    
    cfg['model_args']['out_size'] = 1

    cfg['model'] = cfg['model'].lower()
    
    return cfg