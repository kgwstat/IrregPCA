from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler):
    """Sampler that yields batches of sample indices for grouped datasets.

    Produces batches of exactly ``batch_size`` sample indices (last batch
    may be smaller if ``drop_last=False``).

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Shuffle sample order each epoch (default True).
    drop_last : bool
        Drop the last incomplete batch (default False).
    generator : torch.Generator or None
        RNG for reproducible shuffling.
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples, generator=self.generator).tolist()
        else:
            indices = list(range(self.n_samples))

        batch: list[int] = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return math.ceil(self.n_samples / self.batch_size)
