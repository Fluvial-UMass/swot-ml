import jax
import jax.numpy as jnp
import jax.tree_util as jutil
import jax.sharding as jshard
import torch
from torch.utils.data import DataLoader


class HydroDataLoader(DataLoader):

    def __init__(self, cfg, dataset):
        torch.manual_seed(cfg['model_args']['seed'])

        num_workers = cfg.get('num_workers', 1)
        persistent_workers = False if num_workers == 0 else cfg.get('persistent_workers', True)

        super().__init__(dataset,
                         collate_fn=self.collate_fn,
                         shuffle=cfg.get('shuffle', True),
                         batch_size=cfg.get('batch_size', 1),
                         num_workers=num_workers,
                         pin_memory=cfg.get('pin_memory', True),
                         drop_last=cfg.get('drop_last', False),
                         timeout=cfg.get('timeout', 900),
                         persistent_workers=persistent_workers)
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

        # Batch sharding params
        backend = cfg.get('backend', None)
        num_devices = cfg.get('num_devices', None)
        self.set_jax_sharding(backend, num_devices)

    @staticmethod
    def collate_fn(sample):
        # I can't figure out how to just not collate. Can't even use lambdas because of multiprocessing.
        return sample

    def set_jax_sharding(self, backend=None, num_devices=None):
        """
        Updates the jax device sharding of data. 
    
        Args:
        backend (str): XLA backend to use (cpu, gpu, or tpu). If None is passed, select GPU if available.
        num_devices (int): Number of devices to use. If None is passed, use all available devices for 'backend'.
        """
        available_devices = _get_available_devices()
        # Default use GPU if available
        if backend is None:
            backend = 'gpu' if 'gpu' in available_devices.keys() else 'cpu'
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
                f"batch_size ({self.batch_size}) must be evenly divisible by the num_devices ({self.num_devices}).")

        print(f"Batch sharding set to {self.num_devices} {backend}(s)")
        devices = jax.local_devices(backend=backend)[:self.num_devices]
        mesh = jshard.Mesh(devices, ('batch',))
        pspec = jshard.PartitionSpec('batch',)
        self.sharding = jshard.NamedSharding(mesh, pspec)

    def shard_batch(self, batch):

        def map_fn(path, leaf):
            keys = [p.key for p in path]
            if 'dynamic_dt' in keys:
                return leaf
            return jax.device_put(jnp.array(leaf), self.sharding)

        batch = jutil.tree_map_with_path(map_fn, batch)

        return batch


def _get_available_devices():
    """
    Returns a dict of number of available backend devices
    """
    devices = {}
    for backend in ['cpu', 'gpu', 'tpu']:
        try:
            n = jax.local_device_count(backend=backend)
            devices[backend] = n
        except RuntimeError:
            pass
    return devices
