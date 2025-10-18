from collections import deque, defaultdict
from functools import partial
import random
from typing import Iterator

import numpy as np
import jax
from jaxtyping import Array
import torch
from torch.utils.data import Sampler, DataLoader

from config.config import Config
from .cached_basingraphdataset import GraphBatch, CachedBasinGraphDataset


class GraphPackingSampler(Sampler[list[int]]):
    """
    An infinite sampler that intelligently packs graphs to a target node count.

    This sampler continuously yields batches and re-shuffles the data
    when one pass is complete. It uses a "best-fit" heuristic to fill batches,
    reducing sampling bias against large graphs and minimizing wasted space.
    """

    def __init__(
        self,
        basin_index_map: dict[str, list[int]],
        basin_node_counts: dict[str, int],
        target_nodes_per_batch: int,
        shuffle: bool,
        candidate_pool_size: int,
        infinite_sampling: bool,
    ):
        self.basin_index_map = basin_index_map
        self.basin_node_counts = basin_node_counts
        self.target_nodes = target_nodes_per_batch
        self.shuffle = shuffle
        self.candidate_pool_size = candidate_pool_size
        self.infinite_sampling = infinite_sampling

        # Ensure there is room for at least one padding node per batch.
        # Otherwise, the padded edges will have to point at real nodes, which even if
        # mask the results out later can cause gradient problems.
        self.target_real_nodes = target_nodes_per_batch - 1

        self.samples = []
        for basin, indices in self.basin_index_map.items():
            node_count = self.basin_node_counts.get(basin)
            for index in indices:
                self.samples.append({"index": index, "nodes": node_count})

        # Check for any basin(s) that would be excluded from sampling
        oversized_basins = {
            basin: count
            for basin, count in self.basin_node_counts.items()
            if count > self.target_real_nodes
        }
        if oversized_basins:
            raise ValueError(
                "Arg target_nodes_per_batch cannot be less than the node count for any basin. "
                f"Found {len(oversized_basins)} basins with more than {self.target_real_nodes} nodes.\n"
                f"{oversized_basins=}"
            )

        total_nodes = sum(s["nodes"] for s in self.samples)
        self.estimated_len = (total_nodes // self.target_real_nodes) + 1

    def __len__(self) -> int:
        # This sampler is infinite, so __len__ is not well-defined.
        # Returning a very large number can help with some utilities, but it's not strictly necessary.
        if self.infinite_sampling:
            return int(1e12)
        else:
            return self.estimated_len

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yields batches indefinitely.
        """
        # --- CHANGE: Main loop is now wrapped in `while True` to make the iterator infinite ---
        while True:
            samples_for_epoch = self.samples[:]
            if self.shuffle:
                random.shuffle(samples_for_epoch)

            samples_deque = deque(samples_for_epoch)
            while samples_deque:
                # 1. Seed the batch with the next available sample
                seed_sample = samples_deque.popleft()
                current_batch_indices = [seed_sample["index"]]
                nodes_in_batch = seed_sample["nodes"]

                # 2. Intelligently fill the rest of the batch
                while nodes_in_batch < self.target_real_nodes:
                    remaining_space = self.target_real_nodes - nodes_in_batch
                    best_fit_candidate_idx = -1
                    best_fit_nodes = 0

                    # 3. Look ahead at a pool of candidates
                    pool_size = min(self.candidate_pool_size, len(samples_deque))
                    for i in range(pool_size):
                        candidate = samples_deque[i]
                        if best_fit_nodes < candidate["nodes"] <= remaining_space:
                            best_fit_nodes = candidate["nodes"]
                            best_fit_candidate_idx = i

                    # 4. If a suitable candidate was found, add it to the batch
                    if best_fit_candidate_idx != -1:
                        best_fit_sample = samples_deque[best_fit_candidate_idx]
                        current_batch_indices.append(best_fit_sample["index"])
                        nodes_in_batch += best_fit_sample["nodes"]
                        del samples_deque[best_fit_candidate_idx]
                    else:
                        break  # No candidate in the pool fits, finalize the batch

                yield current_batch_indices

            if not self.infinite_sampling:
                break


def padding_collate_fn(batch: list[tuple], target_nodes_per_batch: int) -> GraphBatch:
    """
    Collates (basin, time) samples and then pads them to a fixed size.
    """
    basins, dates, samples = zip(*batch)  # List of tuples -> lists

    # Step 1: Pre-calculate sizes from all samples in the batch
    num_nodes_per_graph = [s.static.shape[0] for s in samples]
    num_real_nodes = sum(num_nodes_per_graph)
    num_padding_nodes = target_nodes_per_batch - num_real_nodes
    assert num_padding_nodes >= 0, "batch was created with more nodes than target"

    # Step 2: Pack the "real" data by concatenating
    # Dynamic features
    packed_dynamic = defaultdict(list)
    for sample in samples:
        for source, data in sample.dynamic.items():
            packed_dynamic[source].append(data)

    unpadded_dynamic = {
        source: np.concatenate(data_list, axis=1) for source, data_list in packed_dynamic.items()
    }

    # Static features and Targets
    unpadded_static = np.concatenate([s.static for s in samples if s.static is not None], axis=0)
    unpadded_y = np.concatenate([s.y for s in samples if s.y is not None], axis=1)
    unpadded_graph_idx = np.repeat(
        np.arange(len(samples)), repeats=np.array(num_nodes_per_graph)
    )

    node_offset = 0
    offset_edges_list = []
    for i, sample in enumerate(samples):
        offset_edges_list.append(sample.graph_edges + node_offset)
        node_offset += num_nodes_per_graph[i]

    unpadded_edges = np.concatenate(offset_edges_list, axis=1)
    num_real_edges = unpadded_edges.shape[1]
    target_edges_per_batch = target_nodes_per_batch
    num_padding_edges = target_edges_per_batch - num_real_edges
    assert num_padding_edges >= 0, "Batch has more edges than target"

    # --- Step 3: Pad All Tensors to Target Size ---
    # Pad node features
    padded_static = np.pad(unpadded_static, ((0, num_padding_nodes), (0, 0)))
    padded_y = np.pad(unpadded_y, ((0, 0), (0, num_padding_nodes), (0, 0)))
    padded_graph_idx = np.pad(unpadded_graph_idx, (0, num_padding_nodes))
    # ... (Pad dynamic features similarly) ...
    # This example assumes you have the logic for dynamic features already
    packed_dynamic = defaultdict(list)
    for sample in samples:
        for source, data in sample.dynamic.items():
            packed_dynamic[source].append(data)
    unpadded_dynamic = {
        source: np.concatenate(data_list, axis=1) for source, data_list in packed_dynamic.items()
    }
    padded_dynamic = {
        s: np.pad(d, ((0, 0), (0, num_padding_nodes), (0, 0))) for s, d in unpadded_dynamic.items()
    }

    # Pad with the index of the FIRST PADDED NODE (num_real_nodes).
    # This creates harmless self-loops on a non-existent node.
    padded_edges = np.pad(
        unpadded_edges, ((0, 0), (0, num_padding_edges)), constant_values=num_real_nodes
    )

    # --- Step 4: Create Final Masks ---
    node_mask = np.concatenate(
        [np.ones(num_real_nodes, dtype=np.bool_), np.zeros(num_padding_nodes, dtype=np.bool_)]
    )

    edge_mask = np.concatenate(
        [np.ones(num_real_edges, dtype=np.bool_), np.zeros(num_padding_edges, dtype=np.bool_)]
    )
    # # Unmask the FIRST padding edge to prevent NaN during the softmax edge calculations
    # edge_mask = edge_mask.at[num_real_edges].set(True)

    # --- Step 5: Assemble Final Batch ---
    final_batch = GraphBatch(
        dynamic=padded_dynamic,
        static=padded_static,
        graph_edges=padded_edges,
        graph_idx=padded_graph_idx,
        node_mask=node_mask,
        edge_mask=edge_mask,
        y=padded_y,
    )

    return list(basins), list(dates), final_batch


class CachedBasinGraphDataLoader(DataLoader):
    def __init__(self, cfg: Config, dataset: CachedBasinGraphDataset):
        torch.manual_seed(cfg.model_args.seed)

        collate_fn = partial(padding_collate_fn, target_nodes_per_batch=cfg.target_nodes_per_batch)
        is_training = dataset.data_subset == "train"
        basin_subbasin_counts = {k: len(v) for k, v in dataset.basin_subbasin_map.items()}

        batch_sampler = GraphPackingSampler(
            dataset.basin_index_map,
            basin_subbasin_counts,
            cfg.target_nodes_per_batch,
            cfg.shuffle,
            cfg.candidate_pool_size,
            infinite_sampling=is_training,
        )

        threading = cfg.num_workers > 0
        timeout = 900 if threading else 0
        persistent_workers = False
        # persistent_workers = True if threading else False

        super().__init__(
            dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            timeout=timeout,
            persistent_workers=persistent_workers,
        )
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")


    # Expose these dataset methods for convenience.
    def denormalize(self, x: Array, name: str):
        return self.dataset.denormalize(x, name)

    def denormalize_target(self, y_normalized: Array):
        return self.dataset.denormalize_target(y_normalized)


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
