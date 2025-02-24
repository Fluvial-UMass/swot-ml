<h1 align='center'>tss-ml</h1>

JAX implementation of deep learning time series modeling of river discharge, sediment flux, and water quality. This code includes options for:
- training (single and ensemble)
- evaluation
- inference 
- hyperparameter grid/random search
- finetuning

which can be fully automated using a combination of configuration files and arguments to `run.py`.

## Environment
To train and run these models on your machine, make a new environment using the `packages.txt` file. To do this with conda:
```sh
conda create -n tss-ml --file packages.txt
```
If you want to use NVIDIA GPU(s) for training, you will have to install the correct CUDA drivers and [JAX with CUDA support](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu). You can check that JAX can locate your GPU using this simple script. 
```sh
conda activate tss-ml
python -c "import jax; print(jax.devices())"
```

## Configuration file
All options for dataset creation, model hyperparameters, training progress, logging, etc. can be configured in a yaml file. These details are not exhaustively documented, but the [example config files](./runs/_examples/) provide an overview of what is possible. This structure was inspired by [NeuralHydrology](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html) although the implementation is fairly different. 

## Dataset & Dataloader
The [HydroDataset](./src/data/hydrodata.py) and [HydroDataLoader](./src/data/hydroloader.py) classes are implementations of PyTorch datasets and dataloaders. The `HydroDataset` class does several jobs to prepare the data:
- reads in list of training and testing basins
- reads in static basin attributes
- reads in dynamic timeseries data (compiled from per-basin netcdf files)
- normalizes the data based on the training subset
- (optionally) caches compiled dynamic data based on the hash of the data configuration
- (optionally) encodes categorical or bitmask features
- creates batches of data of shape (batch, sequence, features) or (batch, sequence, nodes, features) when in graph mode. 

The `HydroDataLoader` is mostly a thin wrapper on the [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders). The other side of 'mostly' is that the HydroDataLoader also sets the [sharding and device settings](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) of each batch to ensure JAX takes full advantage of GPU(s). 

## Models
Models are implemented in [Equinox](https://github.com/patrick-kidger/equinox) and have to abide by certain rules to be compatible with the [sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) of `jit`'d JAX code. The forward pass of each model assumes that the model has been called using `jax.vmap` on the batch dimension of the data (i.e. only runs on a single sequence). To remain compatible with the training code, these models must all have the same signature for call:

```python
def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
```

where data is the dict returned from the `HydroDataLoader` (after being `jax.vmap`'d over the batch dimension) containing all dynamic, static, and target values. New models must be added to the [model.make()](./src/models/__init__.py#:~:text=def%20make) and [config.set_model_data_args()](./src/config.py#:~:text=def%20set_model_data_args) functions to correctly create them based on the config file. 

This code was developed with the primary motivation of merging continuous modeled climate data with irregularly-timed optical satellite imagery and on-the-ground water quality samples. As such, models initialization is based on the feature dictionary which is structured as:
```python
features: dict = {
    "dynamic": {
        "<source1>": [str],
        "<source2>": [str], 
    },
    "static": [str],
    "target": [str]
}
```
where `source1` and `source2` will be used to create different components of the model (e.g. LSTM and MLP for the lstm_mlp_attn model). These get blended together (making use of static data however you see fit) before making predictions for the targets.


## Trainer
the [Trainer](./src/train/trainer.py) class steps through the dataloader, making predictions, calculating loss, and updating the model. There are also methods for logging training progress to a file, monitoring gradients, and saving and loading model states. 


## Basic training example
The simplified workflow for training a model is:
- set up dataset files
- create configuration file
- run `python run.py --train <path/to/config.yml>`
