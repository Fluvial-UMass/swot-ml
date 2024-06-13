# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import argparse
from pathlib import Path

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

    return cfg, trainer

def continue_training(run_dir):
    state = load_last_state(run_dir)
    cfg = state[0]
    dataset = TAPDataset(cfg) 

    cfg['data_subset'] = 'train'
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer

def test_model(cfg, trainer):
    cfg['data_subset'] = 'test'
    dataloader = TAPDataLoader(cfg, trainer.dataloader.dataset)

    results = predict(trainer.model, dataloader, seed=0, denormalize=True)
    bulk_metrics = get_all_metrics(results)
    basin_metrics = get_basin_metrics(results)

    return results, (bulk_metrics, basin_metrics)

def make_plots(cfg, results, metrics, trainer):
    bulk_metrics, basin_metrics = metrics

    fig = mosaic_scatter(cfg, results, bulk_metrics, str(trainer.log_dir))
    fig.savefig(trainer.log_dir / f"epoch{trainer.epoch:03d}_density_scatter.png",  dpi=300)

    metric_args = {
    'nBias':{'range':[-1,1]},
    'rRMSE':{'range':[0,500]},
    'KGE':{'range':[-2,1]},
    'NSE':{'range':[-5,1]},
    'Agreement':{'range':[0,1]},
    'num_obs':{'log':True}}
    figs = basin_metric_histograms(basin_metrics, metric_args)
    for target, fig in figs.items():
        fig.savefig(trainer.log_dir / f"epoch{trainer.epoch:03d}_{target}_metrics_hist_.png",  dpi=300)

def main(args):
    if args.train:
        yml_file = Path(args.train).resolve()
        cfg, trainer = start_training(yml_file)
    elif args.continue_training:
        run_dir = Path(args.continue_training).resolve()
        cfg, trainer = continue_training(run_dir)

    results, metrics = test_model(cfg, trainer)
    make_plots(cfg, results, metrics, trainer)

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
    args = parser.parse_args()
    main(args)