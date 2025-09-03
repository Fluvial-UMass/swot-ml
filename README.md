<h1 align='center'>swot-ml</h1>

JAX implementation of deep learning time series modeling of river discharge. This code includes methods for:
- training (single and ensemble)
- hyperparameter search
- finetuning
- inference 
- evaluation

which can be fully automated using configuration files and or arguments to the main interface [run.py](./src/run.py).

## Setup
### Environment
To train and run these models on your machine, you can use [uv](https://docs.astral.sh/uv/getting-started/) to create a virtual environment and install dependencies from `pyproject.toml`. Open a terminal and navigate to the project directory.

Create a new virtual environment with default name
```sh
uv venv
```
Activate the environment (for mac/linux)
```sh
source .venv/bin/activate
```
Install the dependencies in editable mode with dev packages
```
uv pip install -e .[dev]
```

JAX with CUDA support will be installed automatically from `pyproject.toml` but your system will still need the correct CUDA drivers.
You can check that JAX can locate your GPU using this command (while the virtual environment is activated):
```sh
python -c "import jax; print(jax.devices())"
```
Which should show something like ```[CudaDevice(id=0)``` for a single GPU. If JAX can only find your a CPU, it will print ```[CpuDevice(id=0)]```.


### Configuration file
All options for dataset creation, model hyperparameters, training progress, logging, etc. can be configured in a yaml file. These details are not exhaustively documented, but the [example config files](./runs/_examples/) provide some example uses. The full listing of potential options is only documented in the [the Config validation class](./src/config/config.py). 

### Data files
You will need at a minimum, two types of data to train the model, with another two types depending on model configuration:
1. **time series data**: NetCDF (.nc) files representing time-varying but regular interval (even if the observation is NA/missing) with a seperate file for each site. Each dataset must contain a 'date' index. This does not specify any particular frequency. However, even if you are modeling hourly features, these datetimes needs to be named 'date' unless you edit the hard-coded index in [data/hydrodata.py](./src/data/hydrodata.py).
1. **site lists**: Text (.txt) files indicating which sites (basins, gauges, reaches, etc.) will be used for training and testing. This file contains a new site on each line with no other delimeters.
1. **attributes** (optional): a comma-separated value (.csv) file with static attributes for each site. Only required if you specify static attributes in your configuration.
1. **graph network** (optional): A networkx directed graph in json format. The only data used from this graph are the edge definitions, although the node (and thus edge end points), need to be identical to the sites. Must contain all sites. Only required for certain models, and should only be specified in the configuration file if needed by the model. 

The recommended structure of the data directory is as follows, with some allowance for differences defined in the [Config validator](./src/config/config.py). 
```
<data_dir>/
├── attributes/
│   └── attributes.csv
├── time_series/
│   ├── xxxx.nc
│   └── ... (one .nc file for each site)
├── metadata/
│   ├── site_lists/
│   │   ├── train.txt
│   │   └── test.txt
│   └── graph.json
```

### Train a model
Once the environment, configuration file, and data files are in place, simply call:
```sh
uv run python ./src/run.py train <path_to_configuration_file.yml>
```
which will train a new model according to your config and then produce files with final predictions, error metrics, and some very basic error distribution figures.

## Extending this code

### Models
Models are implemented in [Equinox](https://github.com/patrick-kidger/equinox) and have to be compatible with the [sharp bits of jit'd JAX code.](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html). Each model is a subclass of the [BaseModel] class which helps to standardize the models for integration with the training and inference code. The forward pass of each model assumes that the model has been called using `jax.vmap` on the batch dimension of the data (i.e. only runs on a single sequence). 

To integrate a new model with this package:
1. Write the intialization and forward pass logic as a subclass of the [BaseModel](./src/models/base_model.py). 
1. Add your model to the [model.make() function](./src/models/__init__.py#:~:text=def%20make)
1. Define the required arguments as a new class within [config/model_args.py](./src/config/model_args.py)
1. (Optionally) define arguments to be set based on the loaded dataset within the [set_model_data_args() function](./src/models/__init__.py#:~:text=def%20set_model_data_args)

## Dataset & Dataloader
The [HydroDataset](./src/data/hydrodata.py) and [HydroDataLoader](./src/data/hydroloader.py) classes are implementations of PyTorch datasets and dataloaders. The `HydroDataset` class does several jobs to prepare the data:
- reads in list of training and testing basins
- reads in static basin attributes
- reads in dynamic timeseries data (compiled from per-basin netcdf files)
- normalizes the data based on the training subset
- encodes categorical or bitmask features (as defined in the yml)
- (optionally) caches compiled dynamic data based on the data configuration
- batches data depending on model config, i.e. time series data are shaped [batch_size, sequence_length, n_features] for lumped modeling or [batch_size, sequence_length, n_locations, n_features] for distributed modeling.

The `HydroDataLoader` is mostly a thin wrapper on the [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders). The other side of 'mostly' is that the HydroDataLoader also sets the [sharding and device settings](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) of each batch to ensure JAX takes full advantage of GPU(s). 

## #Trainer
the [Trainer](./src/train/trainer.py) class steps through the dataloader, making predictions, calculating loss, and updating the model. There are also methods for logging training progress to a file, monitoring gradients, and saving and loading model states. 

### Evaluate 
This module contains methods for making predictions, calculating error statistics, and making plots. The prediction methods are similar to those used for training, excet there is no requirement of target data and it is significantly faster since there is no loss or gradient calculations. The outputs are also collected and saved. The error metrics are defined in [metrics.py](./src/evaluate/metrics.py) for both bulk sample statistics and per-basin statistics. The plots are, admittedly, poorly implemented as they rely on some hardcoded ranges and lists of metrics that were useful when written for a manuscript. 

###