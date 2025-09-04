from pathlib import Path
from dask_jobqueue import SLURMCluster
from distributed import as_completed


def _create_configspace(cfg):
    """Creates a ConfigSpace object from a parameter dictionary."""
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    cs.add(CSH.Constant("cfg_path", value=str(cfg["cfg_path"])))
    for param_name, param in cfg["param_search_dict"].items():
        for param_type, param_range in param.items():
            args = {"name": param_name, "lower": param_range[0], "upper": param_range[1]}
            if param_type == "int":
                hp = CSH.UniformIntegerHyperparameter(**args)
            elif param_type == "float":
                hp = CSH.UniformFloatHyperparameter(**args)
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
            cs.add(hp)
    return cs


def update_smac_config(base_cfg, updates, seed):
    for k, v in base_cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk in updates.keys():
                    base_cfg[k][kk] = updates[kk]
        elif isinstance(v, int) or isinstance(v, float):
            if k in updates.keys():
                base_cfg[k] = updates[k]
    base_cfg["model_args"]["seed"] = seed
    return base_cfg


def get_dask_client(cfg, n_workers):
    path = Path(cfg["cfg_path"])
    cluster = SLURMCluster(
        job_name="dask-" + path.stem,
        queue="gpu",
        cores=2,
        processes=1,
        memory="16GB",
        walltime="14-00:00:00",
        log_directory=path.parent / "_slurm_outputs" / "trials",
        job_extra_directives=[
            "-q long",
            "--gpus=1",
            "--constraint=sm_61&vram11",
        ],
        job_script_prologue=[
            "module load conda/latest",
            "conda activate tss-ml",
            "module load cuda/12.6",
            "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8",
            'export PYTHONPATH="/work/pi_kandread_umass_edu/tss-ml/src:$PYTHONPATH"',
        ],
    )
    cluster.scale(n_workers)
    cluster.wait_for_workers(n_workers)
    client = cluster.get_client()
    client.wait_for_workers(n_workers)
    return client


def get_smac_facade(cfg, target_fun, n_runs):
    from smac import BlackBoxFacade, Scenario
    from smac.main.config_selector import ConfigSelector

    configspace = _create_configspace(cfg)
    path = Path(cfg["cfg_path"])
    scenario = Scenario(
        configspace=configspace,
        name="smac_opt",
        output_directory=path.parent,
        n_trials=n_runs,
    )
    configselector = ConfigSelector(scenario=scenario, retrain_after=1)
    facade = BlackBoxFacade(
        scenario,
        target_fun,
        overwrite=False,  # Continues if a file with consistent metadata exists.
        config_selector=configselector,
    )
    return facade

def manual_smac_optimize(cfg: dict, n_workers: int, n_runs: int, target_fun):
    from smac.runhistory.dataclasses import TrialValue

    client = get_dask_client(cfg, n_workers)
    facade = get_smac_facade(cfg, target_fun, n_runs)
    initial_params = [facade.ask() for _ in range(n_workers)]
    futures = {client.submit(target_fun, p.config, p.seed): p for p in initial_params}
    next_params = facade.ask()
    while next_params:
        for future in as_completed(futures):
            info = futures[future]
            value = TrialValue(future.result())
            facade.tell(info, value)
            futures.pop(future)
            # Submit a new job to replace the completed one
            if next_params:
                next_future = client.submit(target_fun, next_params.config, next_params.seed)
                futures[next_future] = next_params  # Track it
            try:
                next_params = facade.ask()
            except StopIteration:
                next_params = None
