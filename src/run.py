# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import argparse
import os
import signal
from pathlib import Path
import pickle

from config import *
from data import TAPDataset, TAPDataLoader
from train import Trainer, load_last_state
from evaluate import *

def load_model(run_dir):
    state = load_last_state(run_dir)
    cfg = state[0]
    model = state[1]
    trainer_state = state[2]
    return cfg, model, trainer_state

def start_training(config_yml):
    cfg, _ = read_config(config_yml)
    dataset = TAPDataset(cfg) 
    cfg = set_model_data_args(cfg, dataset)
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, log_parent=config_yml.parent)
    trainer.start_training()

    return cfg, trainer.model, trainer.log_dir, dataset

def continue_training(run_dir):
    cfg, _, _ = load_model(run_dir)
    dataset = TAPDataset(cfg) 
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer.model, dataset

def finetune(finetune_yml:Path):
    finetune = read_yml(finetune_yml)
    run_dir = Path(finetune_yml).parent

    # Load the config and manipulate it a bit
    cfg, _, trainer_state = load_model(run_dir)
    stop_epoch = trainer_state['epoch']
    cfg['num_epochs'] = stop_epoch + finetune.get('additional_epochs',0)
    cfg['transition_begin'] = stop_epoch if finetune.get('reset_lr') else 0
    cfg['cfg_path'] = finetune_yml
    # Insert these params directly.
    cfg.update(finetune.get('config_update',{}))

    dataset = TAPDataset(cfg) 
    dataloader = TAPDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, log_parent=finetune_yml.parent, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer.model, trainer.log_dir, dataset

def hyperparam_grid_search(config_yml:Path, idx):
    cfg, _ = read_config(config_yml)
    cfg = update_cfg_from_grid(cfg, idx)

    for i in range(4):
        cfg['test_basin_file'] = f"metadata/site_lists/k_folds/test_{i}.txt"
        cfg['train_basin_file'] = f"metadata/site_lists/k_folds/train_{i}.txt"
        dataset = TAPDataset(cfg) 

        cfg = set_model_data_args(cfg, dataset)
        dataloader = TAPDataLoader(cfg, dataset)
        
        log_dir = config_yml.parent / f"index_{idx}" / f"fold_{i}"
        if log_dir.is_dir():
            trainer = Trainer(cfg, dataloader, continue_from=log_dir)
        else:
            trainer = Trainer(cfg, dataloader, log_dir=log_dir)

        start_epoch = trainer.epoch
        while trainer.epoch < trainer.num_epochs:
            start_epoch = trainer.epoch
            trainer.start_training()
            if trainer.epoch == start_epoch:
                break

        # Check if we actually did some training this run.
        out_file = trainer.log_dir / "test_data.pkl"
        if (start_epoch != trainer.epoch) or (not out_file.is_file()):
            eval_model(cfg, trainer.model, dataset, trainer.log_dir, False)


def make_plots(cfg, results, bulk_metrics, basin_metrics, data_subset, log_dir):
    fig_dir = log_dir / "figures" / data_subset
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = mosaic_scatter(cfg, results, bulk_metrics, str(log_dir))
    fig.savefig(fig_dir / f"density_scatter.png",  dpi=300)

    metric_args = {
        'R2': {'range':[-1,1]},
        'nBias':{'range':[-1,1]},
        'rRMSE':{'range':[0,500]},
        'KGE':{'range':[-2,1]},
        'NSE':{'range':[-5,1]},
        'Agreement':{'range':[0,1]}
    }
    figs = basin_metric_histograms(basin_metrics, metric_args)
    for target, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_metrics_hist_.png",  dpi=300)


def eval_model(cfg, model, dataset, log_dir, plots=True):
    def eval_subset(data_subset):
        dataset.update_indices(data_subset)
        dataloader = TAPDataLoader(cfg, dataset)

        results = predict(model, dataloader, quiet=cfg.get('quiet',True), denormalize=True)
        bulk_metrics = get_all_metrics(results)
        basin_metrics = get_basin_metrics(results)
    
        with open(log_dir / f"{data_subset}_data.pkl", 'wb') as f:
            pickle.dump((results, bulk_metrics, basin_metrics), f)

        if plots:
            make_plots(cfg, results, bulk_metrics, basin_metrics, data_subset, log_dir) 

    eval_subset('test')
    # eval_subset('train')
     

def main(args):
    if args.train:
        config_yml = Path(args.train).resolve()
        cfg, model, run_dir, dataset = start_training(config_yml)
    elif args.continue_training:
        run_dir = Path(args.continue_training).resolve()
        cfg, model, dataset = continue_training(run_dir)
    elif args.finetune:
        finetune_yml = Path(args.finetune).resolve()
        cfg, model, run_dir, dataset = finetune(finetune_yml)
    elif args.grid_search:
        config_yml = Path(args.grid_search).resolve()
        hyperparam_grid_search(config_yml, args.grid_index)
        return
    elif args.test:
        run_dir = Path(args.test).resolve()
        cfg, model, _ = load_model(run_dir)
        dataset = TAPDataset(cfg) 
    eval_model(cfg, model, dataset, run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model based on the command line arguments.")

    # Create a mutually exclusive arg group for train/continue/test. 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', 
                    type=str, 
                    help='Path to the training configuration file.')
    group.add_argument('--continue', 
                    dest='continue_training', 
                    type=str, 
                    help='Directory path to continue training.')
    group.add_argument('--finetune',
                        type=str,
                        help='Path to the finetune configuration yml file.')
    group.add_argument('--grid_search',
                    type=str,
                    help='Path to the grid search configuration file.')
    group.add_argument('--test',  
                    type=str, 
                    help='Path to directory with model to test.')
    
    # Add a new argument for grid search index
    parser.add_argument('--grid_index', 
                        type=int, 
                        help='Index in the hyperparameter grid to evaluate (required if --grid_search is used)',
                        required='--grid_search' in sys.argv)
    
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup all processes (helps with pytorch dataloader workers)
        os.killpg(os.getpgid(0), signal.SIGKILL)
    