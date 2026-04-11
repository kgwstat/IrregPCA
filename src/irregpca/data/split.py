from __future__ import annotations

import torch

from .schemas import GroupedObservationDataset


def split_by_sample_id(
    dataset: GroupedObservationDataset,
    train_ratio: float = 0.8,
    random_state: int | None = None,
) -> tuple[GroupedObservationDataset, GroupedObservationDataset]:
    """Split a grouped dataset into train and validation by sample ID.

    Splitting is done at the sample (curve) level, not at the observation
    level, so no sample leaks across the split. Underlying flat storage is
    NOT copied — both sub-datasets are views into the same location/value
    arrays.

    Parameters
    ----------
    dataset : GroupedObservationDataset
    train_ratio : float
        Fraction of samples for training (default 0.8).
    random_state : int or None
        RNG seed for reproducible splits.

    Returns
    -------
    train_ds, valid_ds : GroupedObservationDataset
        Both share the same flat storage (views by index).

    Raises
    ------
    ValueError
        If there are fewer than 2 samples, or if the split would leave
        either set empty.
    """
    n = dataset.n_samples
    if n < 2:
        raise ValueError(
            f"Cannot split a dataset with {n} sample(s). "
            "Need at least 2 samples."
        )

    cut = max(1, min(n - 1, int(train_ratio * n)))

    if random_state is not None:
        gen = torch.Generator()
        gen.manual_seed(random_state)
        perm = torch.randperm(n, generator=gen)
    else:
        perm = torch.randperm(n)

    train_sample_indices = perm[:cut]
    valid_sample_indices = perm[cut:]

    return (
        _index_subset(dataset, train_sample_indices),
        _index_subset(dataset, valid_sample_indices),
    )


def _index_subset(
    dataset: GroupedObservationDataset,
    sample_indices: torch.Tensor,
) -> GroupedObservationDataset:
    """Extract a subset of samples from a GroupedObservationDataset.

    The returned dataset shares the same flat location/value storage as the
    parent but with new contiguous offsets into a re-concatenated flat array.
    This avoids copying the underlying observations.

    Parameters
    ----------
    dataset : GroupedObservationDataset
    sample_indices : torch.Tensor
        1-D long tensor of sample positions (not IDs) within ``dataset``.

    Returns
    -------
    GroupedObservationDataset
    """
    locs = dataset.locations
    vals = dataset.values
    if not isinstance(locs, torch.Tensor):
        import numpy as np
        locs = torch.from_numpy(np.array(locs))
    if not isinstance(vals, torch.Tensor):
        import numpy as np
        vals = torch.from_numpy(np.array(vals))

    sub_sample_ids = dataset.sample_ids[sample_indices]
    sub_lengths = dataset.lengths[sample_indices]
    sub_offsets_src = dataset.offsets[sample_indices]

    # collect observation index ranges for each selected sample
    obs_chunks_locs = []
    obs_chunks_vals = []
    for i in range(len(sample_indices)):
        start = int(sub_offsets_src[i])
        length = int(sub_lengths[i])
        obs_chunks_locs.append(locs[start : start + length])
        obs_chunks_vals.append(vals[start : start + length])

    new_locs = torch.cat(obs_chunks_locs, dim=0)
    new_vals = torch.cat(obs_chunks_vals, dim=0)

    new_offsets = torch.zeros(len(sample_indices), dtype=torch.long)
    new_offsets[1:] = sub_lengths[:-1].cumsum(0)

    return GroupedObservationDataset(
        sample_ids=sub_sample_ids,
        offsets=new_offsets,
        lengths=sub_lengths,
        locations=new_locs,
        values=new_vals,
        input_dim=dataset.input_dim,
    )
