from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .schemas import GroupedObservationDataset


def load_memmap_dataset(
    locations_path: str | Path,
    values_path: str | Path,
    sample_ids_path: str | Path,
    input_dim: int,
    n_obs: int,
    n_samples: int,
    dtype: str = "float32",
) -> GroupedObservationDataset:
    """Load a GroupedObservationDataset from memory-mapped numpy arrays.

    The flat arrays (locations, values) are opened as ``np.memmap`` objects
    so that only the accessed pages are loaded into RAM. Metadata tensors
    (sample_ids, offsets, lengths) are loaded fully into memory.

    Parameters
    ----------
    locations_path : str or Path
        Path to a flat ``.npy`` or raw binary file of shape ``(n_obs, input_dim)``.
    values_path : str or Path
        Path to a flat file of shape ``(n_obs,)``.
    sample_ids_path : str or Path
        Path to a file of shape ``(n_obs,)`` with integer sample IDs.
    input_dim : int
        Dimensionality of the location space.
    n_obs : int
        Total number of observations.
    n_samples : int
        Number of unique samples (used for pre-allocating metadata).
    dtype : str
        NumPy dtype string (default ``"float32"``).

    Returns
    -------
    GroupedObservationDataset
        Locations and values are ``np.memmap`` instances; metadata tensors
        are CPU tensors.

    Notes
    -----
    The caller is responsible for ensuring the files are sorted by sample ID
    for contiguous group access. Use :func:`build_memmap_dataset` to create
    sorted files from raw data.
    """
    locations_mm = np.memmap(str(locations_path), dtype=dtype, mode="r", shape=(n_obs, input_dim))
    values_mm = np.memmap(str(values_path), dtype=dtype, mode="r", shape=(n_obs,))
    sample_ids_arr = np.memmap(str(sample_ids_path), dtype="int64", mode="r", shape=(n_obs,))

    # build grouped metadata from the sample IDs
    ids_tensor = torch.from_numpy(np.array(sample_ids_arr))
    unique_ids, counts = torch.unique_consecutive(ids_tensor, return_counts=True)
    offsets = torch.zeros(len(unique_ids), dtype=torch.long)
    offsets[1:] = counts[:-1].cumsum(0)

    return GroupedObservationDataset(
        sample_ids=unique_ids,
        offsets=offsets,
        lengths=counts.long(),
        locations=locations_mm,
        values=values_mm,
        input_dim=input_dim,
    )
