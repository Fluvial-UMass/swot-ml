import jax
import jax.numpy as jnp
import jax.sharding as jshard
from jaxtyping import Array
import torch
from torch.utils.data import DataLoader

from config.config import Config
from .hydrodata import HydroDataset


class HydroDataLoader(DataLoader):
    def __init__(self, cfg: Config, dataset: HydroDataset):
        torch.manual_seed(cfg.model_args.seed)

        super().__init__(
            dataset,
            collate_fn=self.collate_fn,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            timeout=cfg.timeout,
            persistent_workers=cfg.persistent_workers,
        )
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        self.set_jax_sharding(cfg.backend, cfg.num_devices)

    @staticmethod
    def collate_fn(sample):
        # I can't figure out how to just not collate. Can't even use lambdas because of multiprocessing.
        return sample
    

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
    
    # Expose these dataset methods for easier use.
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
