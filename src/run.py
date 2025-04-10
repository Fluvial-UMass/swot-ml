#!/usr/bin/env python3

# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import traceback
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import pickle
import copy

from smac.utils.configspace import get_config_hash

from config import *
from data import HydroDataset, HydroDataLoader
from train import Trainer, update_smac_config, manual_smac_optimize
from evaluate import *


def cleanup_dl(dl: HydroDataLoader):
    if dl is None:
        return
    if dl._iterator:
        dl._iterator._shutdown_workers()
    del dl


def train_from_config(cfg: dict, log_dir: Path | None = None):
    """Trains a model from a given configuration file.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    trainer_kwargs : dict, optional
        Additional keyword arguments to pass to the Trainer constructor. This is useful
        to continue training from a save point or to set a custom logging directory. 

    Returns
    -------
    cfg : dict
        The updated configuration dictionary.
    trainer : Trainer
        The Trainer object after training. Contains the model and other 
        useful attributes.
    dataset : HydroDataset
        The HydroDataset loaded for training.
    """
    trainer = None
    dataset = HydroDataset(cfg)
    cfg = set_model_data_args(cfg, dataset)
    dataloader = HydroDataLoader(cfg, dataset)

    if log_dir and log_dir.is_dir():
        trainer = Trainer.load_last_checkpoint(log_dir)
        # Could fail to load if nothing was saved.
        if trainer is not None:
            trainer.dataloader = dataloader
    if trainer is None:
        trainer = Trainer(cfg, dataloader, log_dir=log_dir)

    # trainer = Trainer(cfg, dataloader, **trainer_kwargs)

    # if model_finetune_kwargs:
    #     trainer.model.finetune_update(**model_finetune_kwargs)

    trainer.start_training()
    cleanup_dl(dataloader)

    return cfg, trainer, dataset


def train_ensemble(config_yml: Path, ensemble_seed: int):
    """Trains a model for an ensemble by modifying the random seed.

    Parameters
    ----------
    config_yml : Path
        Path to the YAML configuration file.
    ensemble_seed : int
        Seed used to differentiate ensemble members. This seed is added to the random
        seed in the config that is used in model creation and dataloader shuffling.

    Returns
    -------
    cfg : dict
        The updated configuration dictionary.
    model : eqx.Module
        The trained model.
    log_dir : Path
        The directory where training logs and checkpoints were saved.
    dataset : HydroDataset
        The HydroDataset loaded for training.
    """
    cfg, _ = read_config(config_yml)
    cfg['model_args']['seed'] += ensemble_seed

    log_dir = config_yml.parent / "base_models" / f"seed_{ensemble_seed:02d}"
    cfg, trainer, dataset = train_from_config(cfg, log_dir)

    return cfg, trainer.model, trainer.log_dir, dataset


# def finetune(finetune_yml: Path):
#     """Fine-tunes a pre-trained model using a separate configuration file.

#     Parameters
#     ----------
#     finetune_yml : Path
#         Path to the YAML file containing fine-tuning parameters. This file should be in
#         the same directory as the original model run. It contains minimal parameters,
#         only those that are updated.

#     Returns
#     -------
#     cfg : dict
#         The updated configuration dictionary.
#     model : eqx.Module
#         The fine-tuned model.
#     log_dir : Path
#         The directory where training logs and checkpoints were saved.
#     dataset : HydroDataset
#         The HydroDataset loaded for training.
#     """
#     # Load the config and manipulate it a bit
#     run_dir = finetune_yml.parent
#     trainer = Trainer.load_last_checkpoint(run_dir)
#     cfg = trainer.cfg.copy()

#     # Read in the finetuning parameters
#     finetune = read_yml(finetune_yml)
#     cfg['num_epochs'] = trainer.epoch + finetune.get('additional_epochs', 0)
#     cfg['transition_begin'] = trainer.epoch if finetune.get('reset_lr') else 0
#     cfg['cfg_path'] = finetune_yml

#     # Insert these params directly.
#     cfg.update(finetune.get('config_update', {}))

#     trainer_kwargs = {'continue_from': run_dir}

#     model_finetune_kwargs = finetune.get('model_update', None)
#     cfg, trainer, dataset = train_from_config(cfg, trainer_kwargs,
#                                               model_finetune_kwargs)

#     return cfg, trainer.model, trainer.log_dir, dataset


def hyperparam_grid_search(config_yml: Path, idx: int):
    """Performs a single hyperparameter grid search trial using k-fold 
    cross-validation.

    Parameters
    ----------
    config_yml : Path
        Path to the YAML configuration file.
    idx : int
        Index of the current hyperparameter combination in the grid.

    Returns
    -------
    None. Results are saved to files within the specified directories.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """

    cfg, _ = read_config(config_yml)
    cfg = update_cfg_from_grid(cfg, idx)

    k = 4
    for i in range(k):
        log_dir = config_yml.parent / "trials" / f"index_{idx}" / f"fold_{i}"
        out_file = log_dir / "test_data.pkl"
        if out_file.is_file():
            print(f"Fold {i+1} of {k} was already completed.")
            continue

        cfg['test_basin_file'] = f"metadata/site_lists/k_folds/test_{i}_{k}.txt"
        cfg['train_basin_file'] = f"metadata/site_lists/k_folds/train_{i}_{k}.txt"
        dataset = HydroDataset(cfg)
        cfg = set_model_data_args(cfg, dataset)
        dataloader = HydroDataLoader(cfg, dataset)

        if log_dir.is_dir():
            trainer = Trainer.load_last_checkpoint(log_dir)
            trainer.dataloader = dataloader
        else:
            trainer = Trainer(cfg, dataloader, log_dir=log_dir)

        start_epoch = trainer.epoch
        while trainer.epoch < trainer.num_epochs:
            start_epoch = trainer.epoch
            trainer.start_training()
            if trainer.epoch == start_epoch:
                break

        # Check if we actually did some training this run.
        if (start_epoch != trainer.epoch) or (not out_file.is_file()):
            eval_model(cfg,
                       trainer.model,
                       dataset,
                       trainer.log_dir,
                       run_predict=False,
                       run_train=False,
                       make_plots=False)
        print(f"Fold {i+1} of {k} done!")

        if dataloader:
            cleanup_dl(dataloader)


def hyperparam_smac_optimize(config_yml: Path, n_workers: int, n_runs: int):
    cfg, _ = read_config(config_yml)

    def target_fun(updates, seed):
        local_cfg, _ = read_config(updates['cfg_path'])
        local_cfg = update_smac_config(local_cfg, updates, seed)

        trial_name = get_config_hash(updates)
        path = Path(local_cfg['cfg_path'])
        log_dir = path.parent / 'trials' / f'{path.stem}_{trial_name}_{seed}'

        local_cfg, trainer, dataset = train_from_config(local_cfg, log_dir)

        eval_model(local_cfg,
                   trainer.model,
                   dataset,
                   trainer.log_dir,
                   run_test=True,
                   run_predict=False,
                   run_train=False,
                   make_plots=False)

        # Weird to load instead of returning directly but this is a bandaid.
        results_file = trainer.log_dir / 'test_data.pkl'
        with open(results_file, 'rb') as f:
            results, bulk_metrics, basin_metrics = pickle.load(f)

        # return basin_metrics['flux']['RE'].median()
        return basin_metrics[:]['RE'].median().mean()

    manual_smac_optimize(cfg, n_workers, n_runs, target_fun)


def calc_attributions(run_dir: Path):
    trainer = Trainer.load_last_checkpoint(run_dir)
    cfg = trainer.cfg
    cfg['batch_size'] = cfg['batch_size'] // 10
    cfg['data_subset'] = 'test'

    dataset = HydroDataset(cfg)
    cfg = set_model_data_args(cfg, dataset)
    dataloader = HydroDataLoader(cfg, dataset)

    save_dir = run_dir / 'figures' / 'attribution'
    save_dir.mkdir(parents=True, exist_ok=True)

    save_all_intgrads(cfg, trainer.model, dataloader, save_dir, m_steps=10)
    plot_average_attribution(save_dir, dataset)


def load_prediction_model(run_dir: Path, chunk_idx: int | None = None):
    """Loads a pre-trained model and chunk of the dataset into memory.

    Parameters
    ----------
    run_dir : Path
        Path to the model training directory.
    chunk_idx : int
        Index of the basin chunking to select and predict on.

    Returns
    -------
    cfg : dict
        The model training configuration dictionary.
    model : eqx.Module
        The pre-trained model weights and structure.
    predict_dataset : HydroDataset
        A dataset containing a subset of the prediction domain.
    eval_dir : Path
        A directory for saving the results of this subset.
    """
    # cfg, model, _ = load_model(run_dir)
    trainer = Trainer.load_last_checkpoint(run_dir)
    cfg = trainer.cfg

    train_dataset = HydroDataset(cfg)
    cfg['data_subset'] = 'predict'
    cfg['shuffle'] = False  # No need to shuffle for inference
    if chunk_idx:
        cfg['basin_file'] = f'metadata/site_lists/predictions/chunk_{chunk_idx:02}.txt'

    predict_dataset = HydroDataset(cfg, train_ds=train_dataset, use_cache=False)

    eval_dir = run_dir / 'inference'
    eval_dir.mkdir(parents=True, exist_ok=True)

    return cfg, trainer.model, predict_dataset, eval_dir


def make_all_plots(cfg: dict, results: pd.DataFrame, bulk_metrics: dict,
                   basin_metrics: pd.DataFrame, data_subset: str, log_dir: Path):
    """Generates and saves some accuracy plots for a given data subset.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    results : pd.DataFrame
        The model predictions and observations.
    bulk_metrics : dict
        The bulk evaluation metrics.
    basin_metrics : pd.DataFrame
        DataFrame of basin-specific evaluation metrics.
    data_subset : str
        Name of the data subset (e.g., 'train', 'test').
    log_dir : Path
        The directory where logs and figures are saved.

    Returns
    -------
    None. Figures are saved to the specified directory.
    """

    fig_dir = log_dir / "figures" / data_subset
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = mosaic_scatter(cfg, results, bulk_metrics, str(log_dir))
    fig.savefig(fig_dir / f"density_scatter.png", dpi=300)

    figs = basin_metric_histograms(basin_metrics)
    for target, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_metrics_hist.png", dpi=300)


def eval_model(cfg: dict,
               model,
               dataset: HydroDataset,
               log_dir: Path,
               run_test: bool | str = True,
               run_predict: bool | str = True,
               run_train: bool | str = True,
               make_plots=True):
    """Evaluates a model on specified data subsets, calculates metrics, and 
    generates plots.

    Parameters
    ----------
    cfg : dict
        The model training configuration dictionary.
    model : eqx.Module
        The pre-trained model weights and structure.
    dataset : HydroDataset
        A dataset for estimating and evaluating.
    log_dir : Path
        The directory to save results in.
    run_test, run_predict, run_train : bool | str, optional
        Whether to run the respective phase. Defaults to True. If a string is provided,
        this will be used as the stem of the filepath for saving results.
    make_plots : bool, optional
        Whether to save generic accuracy plots. Defaults to True.

    Notes
    -----
    The `predict` subset does not generate plots, regardless of the value of
    `make_plots`. These would be identical to `test`, as 'predict' covers the same
    basins and time period as 'test' but without validation data.

    Saves
    -----
    Pickled results, bulk metrics, and basin metrics for each data subset. Plots in the
    `log_dir` directory if `make_plots` is `True`.
    """

    def eval_data_subset(data_subset, out_stem=None):
        if out_stem is False:
            return
        elif isinstance(out_stem, str):
            if out_stem[-4:] != '.pkl':
                out_stem += '.pkl'
            else:
                print("Pass string")
        else:
            out_stem = f"{data_subset}_data.pkl"

        results_file = log_dir / out_stem
        print(f'Evaluating {data_subset} subset and saving to: {results_file}')

        dataset.cfg['exclude_target_from_index'] = None
        dataset.update_indices(data_subset)
        dataloader = HydroDataLoader(cfg, dataset)

        results = predict(model,
                          dataloader,
                          quiet=cfg.get('quiet', True),
                          denormalize=True)
        cleanup_dl(dataloader)

        if data_subset != "predict":
            bulk_metrics = get_all_metrics(results)
            basin_metrics = get_basin_metrics(results)
            out = (results, bulk_metrics, basin_metrics)
        else:
            out = results
        with open(results_file, 'wb') as f:
            pickle.dump(out, f)

        if make_plots and data_subset != 'predict':
            make_all_plots(cfg, results, bulk_metrics, basin_metrics, data_subset,
                           log_dir)

    eval_data_subset('test', run_test)
    eval_data_subset('predict', run_predict)
    eval_data_subset('train', run_train)


def main(args: ArgumentParser):
    """Command-line interface for training, testing, and evaluating deep learning models.

    Parameters
    ----------
    args: ArgumentParser
        Parsed command line arguments. The following arguments are supported:

        - **train**: Path to the training configuration file.
        - **train_ensemble**: Path to the training configuration file for ensemble training.
        - **finetune**: Path to the fine-tuning configuration file.
        - **grid_search**: Path to the grid search configuration file.
        - **test**: Path to the directory containing the model to test.
        - **prediction_model**: Path to the directory containing the model to use for predictions.
        - **ensemble_seed**: Integer seed to be added to the model seed (required for ensemble training).
        - **grid_index**: Index of the hyperparameter grid to evaluate (required for grid search).
        - **basin_chunk_index**: Index of the chunked basin list to predict on (required for prediction).

    Notes
    -----
    This script provides a command-line interface for training, testing, and evaluating a hydrological model.
    It supports various modes of operation, including training, fine-tuning, grid search, testing, and prediction.
    The specific actions performed by the script depend on the command-line arguments provided by the user.
    """

    # Default values
    run_test = run_predict = run_train = True

    if args.train:
        config_yml = Path(args.train).resolve()
        cfg, _ = read_config(config_yml)
        cfg, trainer, dataset = train_from_config(cfg)
        model = trainer.model
        eval_dir = trainer.log_dir
    elif args.train_ensemble:
        config_yml = Path(args.train_ensemble).resolve()
        cfg, model, eval_dir, dataset = train_ensemble(config_yml, args.ensemble_seed)
    # elif args.finetune:
    #     finetune_yml = Path(args.finetune).resolve()
    #     cfg, model, eval_dir, dataset = finetune(finetune_yml)
    elif args.grid_search:
        config_yml = Path(args.grid_search).resolve()
        hyperparam_grid_search(config_yml, args.grid_index)
        return
    elif args.smac_optimize:
        config_yml = Path(args.smac_optimize).resolve()
        hyperparam_smac_optimize(config_yml, args.smac_workers, args.smac_runs)
        return
    elif args.test:
        run_dir = args.test.resolve()
        trainer = Trainer.load_last_checkpoint(run_dir)
        cfg = trainer.cfg
        model = trainer.model
        dataset = HydroDataset(cfg)
        eval_dir = run_dir
    elif args.attribution:
        run_dir = args.attribution.resolve()
        calc_attributions(run_dir)
        return
    elif args.prediction_model:
        run_test = run_train = False
        run_predict = f'chunk_{args.basin_chunk_index:02}'
        run_dir = Path(args.prediction_model).resolve()
        cfg, model, dataset, eval_dir = load_prediction_model(
            run_dir, args.basin_chunk_index)

    eval_model(cfg, model, dataset, eval_dir, run_test, run_predict, run_train)


def positive_int(value):
    """Custom argparse type to check for positive integers."""
    if int(value) <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")
    return int(value)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Run model based on the command line arguments.")

    # Create a mutually exclusive arg group for train/continue/test.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train',
                       type=Path,
                       help='Path to the training configuration file.')
    group.add_argument('--train_ensemble',
                       type=Path,
                       help='Path to the training configuration file.')
    group.add_argument('--finetune',
                       type=Path,
                       help='Path to the finetune configuration yml file.')
    group.add_argument('--grid_search',
                       type=Path,
                       help='Path to the grid search configuration file.')
    group.add_argument('--smac_optimize',
                       type=Path,
                       help='Path to the smac optimization configuration file.')
    group.add_argument('--test',
                       type=Path,
                       help='Path to directory with model to test.')
    group.add_argument(
        '--attribution',
        type=Path,
        help='Path to directory with model to use for feature attributions.')
    group.add_argument('--prediction_model',
                       type=Path,
                       help='Path to directory with model to use for predictions.')

    parser.add_argument(
        '--ensemble_seed',
        type=int,
        help='Integer to be added to the model seed (required with --train_ensemble)',
        required='--train_ensemble' in sys.argv)

    parser.add_argument(
        '--grid_index',
        type=positive_int,
        help=
        'Index in the hyperparameter grid to evaluate (required with --grid_search)',
        required='--grid_search' in sys.argv)

    parser.add_argument(
        '--smac_runs',
        type=positive_int,
        help='Maximum number of hyperparameter tests (required with --smac_optimize)',
        required='--smac_optimize' in sys.argv)
    parser.add_argument(
        '--smac_workers',
        type=positive_int,
        help='Maximum number of SLURM jobs (required with --smac_optimize)',
        required='--smac_optimize' in sys.argv)

    parser.add_argument(
        '--basin_chunk_index',
        type=positive_int,
        help=
        'Index of the chunked basin list to predict on. Must have a matching basin file.',
        required='--prediction_model' in sys.argv)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    print('Done! Returning from python.')
    sys.exit(0)
