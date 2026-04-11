from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GroupedObservationDataset:
    """Canonical grouped representation of irregularly observed functional data.

    Observations are stored flat (concatenated across all samples) for memory
    efficiency. Per-sample access uses ``offsets`` and ``lengths``.

    Attributes
    ----------
    sample_ids : torch.Tensor
        Unique sample IDs, shape ``(n_samples,)``.
    offsets : torch.Tensor
        Start index of each sample's observations in the flat storage,
        shape ``(n_samples,)``. dtype: torch.long.
    lengths : torch.Tensor
        Number of observations per sample, shape ``(n_samples,)``.
        dtype: torch.long.
    locations : torch.Tensor or np.ndarray
        Flat observation locations, shape ``(N_obs, d)``.
    values : torch.Tensor or np.ndarray
        Flat observation values, shape ``(N_obs,)``.
    input_dim : int
        Dimensionality ``d`` of the location space.

    Examples
    --------
    Create from split tensors::

        ds = GroupedObservationDataset.from_split(sample_ids, locations, values)
    """

    sample_ids: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor
    locations: torch.Tensor | np.ndarray
    values: torch.Tensor | np.ndarray
    input_dim: int

    @property
    def n_samples(self) -> int:
        """Number of unique samples."""
        return int(self.sample_ids.shape[0])

    @property
    def n_obs(self) -> int:
        """Total number of observations."""
        locs = self.locations
        return int(locs.shape[0])

    @classmethod
    def from_split(
        cls,
        sample_ids: torch.Tensor,
        locations: torch.Tensor,
        values: torch.Tensor,
    ) -> "GroupedObservationDataset":
        """Build from flat split arrays sorted by sample_id.

        Parameters
        ----------
        sample_ids : torch.Tensor
            Shape ``(N,)``, integer sample identifiers (need not be sorted).
        locations : torch.Tensor
            Shape ``(N, d)``.
        values : torch.Tensor
            Shape ``(N,)``.

        Returns
        -------
        GroupedObservationDataset
        """
        # sort by sample_id for contiguous grouping
        order = torch.argsort(sample_ids.long())
        sorted_ids = sample_ids.long()[order]
        sorted_locs = locations.float()[order]
        sorted_vals = values.float()[order]

        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        offsets = torch.zeros(len(unique_ids), dtype=torch.long)
        offsets[1:] = counts[:-1].cumsum(0)

        input_dim = sorted_locs.shape[1]

        return cls(
            sample_ids=unique_ids,
            offsets=offsets,
            lengths=counts.long(),
            locations=sorted_locs,
            values=sorted_vals,
            input_dim=input_dim,
        )

    @classmethod
    def from_packed(cls, data: torch.Tensor) -> "GroupedObservationDataset":
        """Build from a packed tensor of shape ``(N, d+2)``.

        Columns: ``[sample_id, loc_1, ..., loc_d, value]``.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(N, d+2)``.

        Returns
        -------
        GroupedObservationDataset
        """
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"`data` must be 2-D with >= 3 columns, got shape {tuple(data.shape)}."
            )
        return cls.from_split(
            sample_ids=data[:, 0],
            locations=data[:, 1:-1],
            values=data[:, -1],
        )

    def to_device(self, device: torch.device) -> "GroupedObservationDataset":
        """Move tensor attributes to ``device`` (not memmap arrays).

        Parameters
        ----------
        device : torch.device

        Returns
        -------
        GroupedObservationDataset
            New instance with tensors on ``device``.
        """
        locs = self.locations
        vals = self.values
        if isinstance(locs, torch.Tensor):
            locs = locs.to(device)
        if isinstance(vals, torch.Tensor):
            vals = vals.to(device)
        return GroupedObservationDataset(
            sample_ids=self.sample_ids.to(device),
            offsets=self.offsets.to(device),
            lengths=self.lengths.to(device),
            locations=locs,
            values=vals,
            input_dim=self.input_dim,
        )

    def to_packed(self) -> torch.Tensor:
        """Return a packed tensor of shape ``(N_obs, d+2)``.

        Columns: ``[sample_id, loc_1, ..., loc_d, value]``.

        Returns
        -------
        torch.Tensor
        """
        locs = self.locations
        vals = self.values
        if isinstance(locs, np.ndarray):
            locs = torch.from_numpy(locs)
        if isinstance(vals, np.ndarray):
            vals = torch.from_numpy(vals)

        # expand sample_ids to per-observation level
        expanded_ids = torch.repeat_interleave(
            self.sample_ids.float(), self.lengths
        ).unsqueeze(-1)
        return torch.cat([expanded_ids, locs.float(), vals.float().unsqueeze(-1)], dim=1)
