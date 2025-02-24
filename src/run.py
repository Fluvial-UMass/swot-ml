# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path
import pickle

from config import *
from data import HydroDataset, HydroDataLoader
from train import Trainer, load_last_state
from evaluate import *


def cleanup_dl(dl: HydroDataLoader):
    if dl is None:
        return
    if dl._iterator:
        dl._iterator._shutdown_workers()
    del dl


def load_model(run_dir: Path):
    """Loads a model and its associated state from a specified directory.

    Parameters
    ----------
    run_dir : Path
        Path to the directory containing the saved model and state.

    Returns
    -------
    cfg : dict
        The configuration dictionary.
    model : eqx.Module
        The loaded model.
    trainer_state : dict
        A dictionary of the current epoch and lost list.

    Raises
    ------
    RuntimeError
        If the model could not be loaded from the specified directory.
    """
    model_loaded, state = load_last_state(run_dir)
    if not model_loaded:
        raise RuntimeError(f"Model could not be loaded from {run_dir}")

    cfg = state[0]
    model = state[1]
    trainer_state = state[2]
    return cfg, model, trainer_state


def train_from_config(cfg: dict, trainer_kwargs: dict = {}):
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
    dataset = HydroDataset(cfg)
    cfg = set_model_data_args(cfg, dataset)
    dataloader = HydroDataLoader(cfg, dataset)

    trainer = Trainer(cfg, dataloader, **trainer_kwargs)
    trainer.start_training()
    cleanup_dl(dataloader)

    return cfg, trainer, dataset


def start_training(config_yml: Path):
    """Starts training a new model from a configuration file.

    Parameters
    ----------
    config_yml : Path
        Path to the YAML configuration file.

    Returns
    -------
    cfg : dict
        The configuration dictionary.
    model : eqx.Module
        The trained model.
    log_dir : Path
        The directory where training logs and checkpoints were saved.
    dataset : HydroDataset
        The HydroDataset loaded for training.
    """
    cfg, _ = read_config(config_yml)
    cfg, trainer, dataset = train_from_config(cfg)

    return cfg, trainer.model, trainer.log_dir, dataset


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
    trainer_kwargs = {'log_dir': log_dir}
    cfg, trainer, dataset = train_from_config(cfg, trainer_kwargs)

    return cfg, trainer.model, trainer.log_dir, dataset


def finetune(finetune_yml: Path):
    """Fine-tunes a pre-trained model using a separate configuration file.

    Parameters
    ----------
    finetune_yml : Path
        Path to the YAML file containing fine-tuning parameters. This file should be in
        the same directory as the original model run. It contains minimal parameters,
        only those that are updated.

    Returns
    -------
    cfg : dict
        The updated configuration dictionary.
    model : eqx.Module
        The fine-tuned model.
    log_dir : Path
        The directory where training logs and checkpoints were saved.
    dataset : HydroDataset
        The HydroDataset loaded for training.
    """
    # Load the config and manipulate it a bit
    run_dir = finetune_yml.parent
    cfg, _, trainer_state = load_model(run_dir)
    stop_epoch = trainer_state['epoch']

    # Read in the finetuning parameters
    finetune = read_yml(finetune_yml)
    cfg['num_epochs'] = stop_epoch + finetune.get('additional_epochs', 0)
    cfg['transition_begin'] = stop_epoch if finetune.get('reset_lr') else 0
    cfg['cfg_path'] = finetune_yml
    # Insert these params directly.
    cfg.update(finetune.get('config_update', {}))

    trainer_kwargs = {'continue_from': run_dir}
    cfg, trainer, dataset = train_from_config(cfg, trainer_kwargs)

    return cfg, trainer.model, trainer.log_dir, dataset


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


def load_prediction_model(run_dir: Path, chunk_idx: int):
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
    cfg, model, _ = load_model(run_dir)
    train_dataset = HydroDataset(cfg)

    cfg['data_subset'] = 'predict'
    cfg['basin_file'] = f'metadata/site_lists/predictions/chunk_{chunk_idx:02}.txt'
    cfg['shuffle'] = False  # No need to shuffle for inference

    predict_dataset = HydroDataset(cfg, train_ds=train_dataset, use_cache=False)

    eval_dir = run_dir / 'inference'
    eval_dir.mkdir(parents=True, exist_ok=True)

    return cfg, model, predict_dataset, eval_dir


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
            if make_plots:
                make_all_plots(cfg, results, bulk_metrics, basin_metrics, data_subset,
                               log_dir)
            out = (results, bulk_metrics, basin_metrics)
        else:
            out = results

        with open(results_file, 'wb') as f:
            pickle.dump(out, f)

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
        cfg, model, eval_dir, dataset = start_training(config_yml)
    elif args.train_ensemble:
        config_yml = Path(args.train_ensemble).resolve()
        cfg, model, eval_dir, dataset = train_ensemble(config_yml, args.ensemble_seed)
    elif args.finetune:
        finetune_yml = Path(args.finetune).resolve()
        cfg, model, eval_dir, dataset = finetune(finetune_yml)
    elif args.grid_search:
        config_yml = Path(args.grid_search).resolve()
        hyperparam_grid_search(config_yml, args.grid_index)
        return
    elif args.test:
        run_dir = args.test.resolve()
        cfg, model, _ = load_model(run_dir)
        dataset = HydroDataset(cfg)
        eval_dir = run_dir
    elif args.prediction_model:
        run_test = run_train = False
        run_predict = f'chunk_{args.basin_chunk_index:02}'
        run_dir = Path(args.prediction_model).resolve()
        cfg, model, dataset, eval_dir = load_prediction_model(
            run_dir, args.basin_chunk_index)

    eval_model(cfg, model, dataset, eval_dir, run_test, run_predict, run_train)


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
    group.add_argument('--test',
                       type=Path,
                       help='Path to directory with model to test.')
    group.add_argument('--prediction_model',
                       type=Path,
                       help='Path to directory with model to use for predictions.')

    # Add a new argument for grid search index
    parser.add_argument(
        '--ensemble_seed',
        type=int,
        help='Integer to be added to the model seed (required if --grid_search is used)',
        required='--train_ensemble' in sys.argv)
    # Add a new argument for grid search index
    parser.add_argument(
        '--grid_index',
        type=int,
        help=
        'Index in the hyperparameter grid to evaluate (required if --grid_search is used)',
        required='--grid_search' in sys.argv)

    # Add a new argument for prediction chunk index
    parser.add_argument('--basin_chunk_index',
                        type=int,
                        help='Index of the chunked basin list to predict on',
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
