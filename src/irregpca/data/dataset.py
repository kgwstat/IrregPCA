from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .schemas import GroupedObservationDataset


class GroupedMapDataset(Dataset):
    """Map-style PyTorch Dataset backed by a :class:`GroupedObservationDataset`.

    Each ``__getitem__(i)`` returns all observations for sample ``i``.
    Use with :func:`grouped_collate_fn` to batch into flat tensors with
    group metadata.

    Parameters
    ----------
    grouped : GroupedObservationDataset
        The underlying grouped data.
    """

    def __init__(self, grouped: GroupedObservationDataset) -> None:
        self._grouped = grouped

    def __len__(self) -> int:
        return self._grouped.n_samples

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Return all observations for sample ``i``.

        Parameters
        ----------
        i : int
            Sample index (0-based).

        Returns
        -------
        dict with keys:
            ``sample_id`` (scalar), ``locations`` (Tensor shape (n_i, d)),
            ``values`` (Tensor shape (n_i,)).
        """
        locs = self._grouped.locations
        vals = self._grouped.values

        if not isinstance(locs, torch.Tensor):
            import numpy as np

            locs = torch.from_numpy(np.asarray(locs))
        if not isinstance(vals, torch.Tensor):
            import numpy as np

            vals = torch.from_numpy(np.asarray(vals))

        start = int(self._grouped.offsets[i])
        length = int(self._grouped.lengths[i])
        return {
            "sample_id": self._grouped.sample_ids[i],
            "locations": locs[start : start + length].float(),
            "values": vals[start : start + length].float(),
        }
