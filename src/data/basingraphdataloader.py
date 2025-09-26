from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp
import jax.sharding as jshard
from jaxtyping import Array
import torch
from torch.utils.data import Sampler, DataLoader

from config.config import Config
from .basingraphdataset import GraphBatch, BasinGraphDataset

import random


class BasinBatchSampler(Sampler[list[int]]):
    def __init__(self, cfg: Config, basin_date_map: dict[str, list[int]]):
        self.basin_date_map = basin_date_map
        self.batch_size = cfg.batch_size
        self.shuffle = cfg.shuffle
        self.basins = list(self.basin_date_map.keys())

        # Pre-calculate the total number of batches we can create
        self.num_batches = 0
        for basin_indices in self.basin_date_map.values():
            # Integer division to drop the last, potentially smaller batch
            self.num_batches += len(basin_indices) // self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Create a master list to hold all possible batches
        all_batches = []

        # Iterate through each basin to generate its batches
        for basin in self.basins:
            indices = self.basin_date_map[basin]
            if self.shuffle:
                # Shuffle the time steps within this basin's indices
                random.shuffle(indices)

            # Create batches for this basin and add them to the master list
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                # Skip the remaining dates from a basin that don't fit in a batch
                if len(batch_indices) == self.batch_size:
                    all_batches.append(batch_indices)

        # The key change: shuffle the entire list of batches
        if self.shuffle:
            random.shuffle(all_batches)

        # Yield from the shuffled master list
        for batch in all_batches:
            yield batch


def collate_fn(batch_of_tuples: list):
    """
    Collates a list of (basin, date, GraphBatch) tuples into a single batched GraphBatch.

    We have to use np.stack and then cast as jnp.array because jnp.stack can't handle NaNs
    for some reason. There might be a good JAX reason for this but I'm not sure.
    """
    # Unzip the list of tuples into separate lists for basins, dates, and samples.
    basins, dates, samples = zip(*batch_of_tuples)

    # Collate the 'dynamic' dictionary.
    batched_dynamic = {
        key: jnp.array(np.stack([s.dynamic[key] for s in samples]))
        for key in samples[0].dynamic.keys()
    }

    # 3. Collate the simple array fields by stacking them.
    batched_graph_edges = jnp.stack([s.graph_edges for s in samples])

    # 4. Handle optional fields, stacking only if they are not None.
    if samples[0].static is None:
        batched_static = None
    else:
        batched_static = jnp.array(np.stack([s.static for s in samples]))

    if samples[0].y is None:
        batched_y = None
    else:
        batched_y = jnp.array(np.stack([s.y for s in samples]))

    # 5. Construct the final batched GraphBatch object.
    batched_graph_batch = GraphBatch(
        dynamic=batched_dynamic,
        graph_edges=batched_graph_edges,
        static=batched_static,
        y=batched_y,
    )

    # Return the collated data, converting tuples from zip to lists.
    return list(basins), list(dates), batched_graph_batch


class BasinGraphDataLoader(DataLoader):
    def __init__(self, cfg: Config, dataset: BasinGraphDataset):
        torch.manual_seed(cfg.model_args.seed)

        batch_sampler = BasinBatchSampler(cfg, dataset.basin_index_map)

        super().__init__(
            dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            timeout=cfg.timeout,
            persistent_workers=cfg.persistent_workers,
        )
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        # self.set_jax_sharding(cfg.backend, cfg.num_devices)

    def set_jax_sharding(self, backend: str | None = None, num_devices: int | None = None):
        """
        Updates the jax device sharding of data.

        Args:
        backend (str): XLA backend to use (cpu, gpu, or tpu). If None is passed, select GPU if available.
        num_devices (int): Number of devices to use. If None is passed, use all available devices for 'backend'.
        """
        available_devices = _get_available_devices()
        # Default use GPU if available
        if backend is None:
            backend = "gpu" if "gpu" in available_devices.keys() else "cpu"
        else:
            backend = backend.lower()

        # Validate the requested number of devices.
        if num_devices is None:
            self.num_devices = available_devices[backend]
        elif num_devices > available_devices[backend]:
            raise ValueError(
                f"requested devices ({backend}: {num_devices}) cannot be greater than available backend devices ({available_devices})."
            )
        elif num_devices <= 0:
            raise ValueError(f"num_devices {num_devices} cannot be <= 0.")
        else:
            self.num_devices = num_devices

        if self.batch_size % self.num_devices != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be evenly divisible by the num_devices ({self.num_devices})."
            )

        print(f"Batch sharding set to {self.num_devices} {backend}(s)")
        devices = jax.local_devices(backend=backend)[: self.num_devices]
        mesh = jshard.Mesh(devices, ("batch",))
        pspec = jshard.PartitionSpec("batch")
        self.sharding = jshard.NamedSharding(mesh, pspec)

    def shard_batch(self, batch: dict):
        def map_fn(path, leaf):
            # Extract names/keys/indexes from all PyTree path parts
            keys = [
                getattr(p, "name", getattr(p, "key", getattr(p, "index", str(p)))) for p in path
            ]
            # keys = [p.key for p in path]
            if "dynamic_dt" in keys:
                return leaf
            return jax.device_put(jnp.array(leaf), self.sharding)

        batch = jax.tree_util.tree_map_with_path(map_fn, batch)

        return batch

    # Expose these dataset methods for convenience.
    def denormalize(self, x: Array, name: str):
        return self.dataset.denormalize(x, name)

    def denormalize_target(self, y_normalized: Array):
        return self.dataset.denormalize_target(y_normalized)

    def update_indices(self, data_subset: str, basin_subset: list[str] = None):
        self.dataset.update_indices(data_subset, basin_subset)


def _get_available_devices():
    """
    Returns a dict of number of available backend devices
    """
    devices = {}
    for backend in ["cpu", "gpu", "tpu"]:
        try:
            n = jax.local_device_count(backend=backend)
            devices[backend] = n
        except RuntimeError:
            pass
    return devices
