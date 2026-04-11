from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from irregpca.data.schemas import GroupedObservationDataset
from irregpca.data.dataset import GroupedMapDataset
from irregpca.data.collate import grouped_collate_fn
from irregpca.data.batching import GroupedBatchSampler


def make_grouped(n_samples: int = 10, obs_per: int = 5) -> GroupedObservationDataset:
    torch.manual_seed(0)
    n_obs = n_samples * obs_per
    ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locs = torch.rand(n_obs, 1)
    vals = torch.rand(n_obs)
    return GroupedObservationDataset.from_split(ids, locs, vals)


class TestGroupedMapDataset:
    def test_len(self):
        ds = make_grouped(n_samples=8)
        map_ds = GroupedMapDataset(ds)
        assert len(map_ds) == 8

    def test_getitem_keys(self):
        ds = make_grouped(n_samples=5, obs_per=4)
        map_ds = GroupedMapDataset(ds)
        item = map_ds[0]
        assert "sample_id" in item
        assert "locations" in item
        assert "values" in item

    def test_getitem_shapes(self):
        ds = make_grouped(n_samples=5, obs_per=4)
        map_ds = GroupedMapDataset(ds)
        item = map_ds[0]
        assert item["locations"].shape == (4, 1)
        assert item["values"].shape == (4,)


class TestGroupedCollateFn:
    def test_batch_keys(self):
        ds = make_grouped(n_samples=5, obs_per=3)
        map_ds = GroupedMapDataset(ds)
        samples = [map_ds[i] for i in range(3)]
        batch = grouped_collate_fn(samples)
        assert "sample_ids" in batch
        assert "locations" in batch
        assert "values" in batch
        assert "group_offsets" in batch
        assert "group_lengths" in batch

    def test_batch_total_obs(self):
        ds = make_grouped(n_samples=5, obs_per=3)
        map_ds = GroupedMapDataset(ds)
        samples = [map_ds[i] for i in range(4)]
        batch = grouped_collate_fn(samples)
        assert batch["locations"].shape[0] == 12  # 4 samples * 3 obs

    def test_offsets_and_lengths_consistent(self):
        ds = make_grouped(n_samples=6, obs_per=5)
        map_ds = GroupedMapDataset(ds)
        samples = [map_ds[i] for i in range(3)]
        batch = grouped_collate_fn(samples)
        total = int(batch["group_lengths"].sum())
        assert total == batch["locations"].shape[0]


class TestGroupedBatchSampler:
    def test_total_samples_covered(self):
        sampler = GroupedBatchSampler(n_samples=20, batch_size=4, shuffle=False)
        seen = []
        for batch in sampler:
            seen.extend(batch)
        assert sorted(seen) == list(range(20))

    def test_batch_sizes(self):
        sampler = GroupedBatchSampler(n_samples=10, batch_size=3, shuffle=False, drop_last=True)
        batches = list(sampler)
        assert all(len(b) == 3 for b in batches)

    def test_dataloader_integration(self):
        ds = make_grouped(n_samples=12, obs_per=4)
        map_ds = GroupedMapDataset(ds)
        sampler = GroupedBatchSampler(n_samples=len(map_ds), batch_size=3, shuffle=False)
        loader = DataLoader(map_ds, batch_sampler=sampler, collate_fn=grouped_collate_fn)
        batches = list(loader)
        assert len(batches) == 4  # 12 / 3
        for batch in batches:
            assert "locations" in batch
