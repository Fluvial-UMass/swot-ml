# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import argparse
from pathlib import Path
import pickle

from config import read_config, set_model_data_args
from data import TAPDataset, TAPDataLoader
from train import Trainer, load_last_state
from evaluate import *

def start_training(yml_file):
    cfg, _ = read_config(yml_file)
    dataset = TAPDataset(cfg) 
    cfg = set_model_data_args(cfg, dataset)

    cfg['data_subset'] = 'train'
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, log_parent=yml_file.parent)
    trainer.start_training()

    return cfg, trainer.model, trainer.log_dir, dataset

def continue_training(run_dir):
    state = load_last_state(run_dir)
    cfg = state[0]
    dataset = TAPDataset(cfg) 

    cfg['data_subset'] = 'train'
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer.model, dataset


def load_model(run_dir):
    state = load_last_state(run_dir)
    cfg = state[0]
    model = state[1]
    dataset = TAPDataset(cfg) 

    return cfg, model, dataset


def test_model(cfg, model, dataset, log_dir):
    cfg['data_subset'] = 'test'
    dataloader = TAPDataLoader(cfg, dataset)

    results = predict(model, dataloader, seed=0, denormalize=True)
    bulk_metrics = get_all_metrics(results)
    basin_metrics = get_basin_metrics(results)
    
    with open(log_dir / "test_data.pkl", 'wb') as f:
        pickle.dump((results, bulk_metrics, basin_metrics), f)

    return results, bulk_metrics, basin_metrics

def make_plots(cfg, results, bulk_metrics, basin_metrics, log_dir):
    fig_dir = log_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig = mosaic_scatter(cfg, results, bulk_metrics, str(log_dir))
    fig.savefig(fig_dir / f"density_scatter.png",  dpi=300)

    metric_args = {
    'nBias':{'range':[-1,1]},
    'rRMSE':{'range':[0,500]},
    'KGE':{'range':[-2,1]},
    'NSE':{'range':[-5,1]},
    'Agreement':{'range':[0,1]},
    'num_obs':{'log':True}}
    figs = basin_metric_histograms(basin_metrics, metric_args)
    for target, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_metrics_hist_.png",  dpi=300)

def main(args):
    if args.train:
        yml_file = Path(args.train).resolve()
        cfg, model, run_dir, dataset = start_training(yml_file)
    elif args.continue_training:
        run_dir = Path(args.continue_training).resolve()
        cfg, model, dataset = continue_training(run_dir)
    elif args.test:
        run_dir = Path(args.test).resolve()
        cfg, model, dataset = load_model(run_dir)

    results, bulk_metrics, basin_metrics = test_model(cfg, model, dataset, run_dir)
    make_plots(cfg, results, bulk_metrics, basin_metrics, run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training or continue training based on the command line arguments.")

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', 
                       type=str, 
                       help='Path to the training configuration file.')
    group.add_argument('--continue', 
                       dest='continue_training', 
                       type=str, 
                       help='Directory path to continue training.')
    group.add_argument('--test',  
                       type=str, 
                       help='Path to directory with model to test.')
    args = parser.parse_args()
    main(args)