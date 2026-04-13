"""Large-dataset workflow using memory-mapped numpy arrays.

Demonstrates how to:
1. Create a fake large dataset written to disk as memmaps.
2. Load it back via GroupedObservationDataset backed by np.memmap.
3. Use GroupedMapDataset + DataLoader for batched access.

This avoids materialising the full dataset in RAM.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from irregpca.data.batching import GroupedBatchSampler
from irregpca.data.collate import grouped_collate_fn
from irregpca.data.dataset import GroupedMapDataset
from irregpca.data.memmap import load_memmap_dataset

# ------------------------------------------------------------------
# 1. Write fake data to disk
# ------------------------------------------------------------------
n_samples = 200
obs_per = 50
n_obs = n_samples * obs_per
input_dim = 1

rng = np.random.default_rng(0)
sample_ids_arr = np.repeat(np.arange(n_samples, dtype=np.int64), obs_per)
locations_arr = rng.random((n_obs, input_dim)).astype(np.float32)
values_arr = (
    np.sin(2 * np.pi * locations_arr[:, 0])
    + 0.1 * rng.standard_normal(n_obs)
).astype(np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    locs_path = tmpdir / "locations.bin"
    vals_path = tmpdir / "values.bin"
    ids_path  = tmpdir / "sample_ids.bin"

    np.memmap(str(locs_path), dtype="float32", mode="w+", shape=(n_obs, input_dim))[:] = (
        locations_arr
    )
    np.memmap(str(vals_path), dtype="float32", mode="w+", shape=(n_obs,))[:] = values_arr
    np.memmap(str(ids_path),  dtype="int64",   mode="w+", shape=(n_obs,))[:] = sample_ids_arr

    # ------------------------------------------------------------------
    # 2. Load as GroupedObservationDataset backed by memmap
    # ------------------------------------------------------------------
    dataset = load_memmap_dataset(
        locations_path=locs_path,
        values_path=vals_path,
        sample_ids_path=ids_path,
        input_dim=input_dim,
        n_obs=n_obs,
        n_samples=n_samples,
    )

    print(f"Dataset: {dataset.n_samples} samples, {dataset.n_obs} observations")

    # ------------------------------------------------------------------
    # 3. Batched access via DataLoader
    # ------------------------------------------------------------------
    map_ds = GroupedMapDataset(dataset)
    sampler = GroupedBatchSampler(
        n_samples=len(map_ds),
        batch_size=16,
        shuffle=True,
    )
    loader = DataLoader(map_ds, batch_sampler=sampler, collate_fn=grouped_collate_fn)

    n_batches = 0
    total_obs = 0
    for batch in loader:
        n_batches += 1
        total_obs += batch["locations"].shape[0]

    print(f"Batches: {n_batches}")
    print(f"Total obs seen: {total_obs} (expected {n_obs})")
    print("Large-dataset memmap example completed successfully.")
