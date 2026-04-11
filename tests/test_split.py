from __future__ import annotations

import pytest
import torch

from irregpca.data.schemas import GroupedObservationDataset
from irregpca.data.split import split_by_sample_id


def make_grouped(n_samples: int = 10, obs_per: int = 5) -> GroupedObservationDataset:
    n_obs = n_samples * obs_per
    ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locs = torch.rand(n_obs, 1)
    vals = torch.rand(n_obs)
    return GroupedObservationDataset.from_split(ids, locs, vals)


class TestGroupedObservationDataset:
    def test_n_samples(self):
        ds = make_grouped(n_samples=8, obs_per=4)
        assert ds.n_samples == 8

    def test_n_obs(self):
        ds = make_grouped(n_samples=8, obs_per=4)
        assert ds.n_obs == 32

    def test_from_packed_roundtrip(self, packed_dataset):
        ds = GroupedObservationDataset.from_packed(packed_dataset)
        assert ds.n_samples == 20
        assert ds.n_obs == 200

    def test_input_dim(self):
        ds = make_grouped()
        assert ds.input_dim == 1

    def test_lengths_sum_to_n_obs(self):
        ds = make_grouped(n_samples=6, obs_per=3)
        assert int(ds.lengths.sum()) == ds.n_obs

    def test_offsets_consistency(self):
        ds = make_grouped(n_samples=5, obs_per=4)
        for i in range(ds.n_samples):
            start = int(ds.offsets[i])
            length = int(ds.lengths[i])
            assert start + length <= ds.n_obs


class TestSplitBySampleId:
    def test_split_sizes(self):
        ds = make_grouped(n_samples=10, obs_per=5)
        train, valid = split_by_sample_id(ds, train_ratio=0.8, random_state=0)
        assert train.n_samples + valid.n_samples == 10
        assert train.n_samples >= 1
        assert valid.n_samples >= 1

    def test_no_id_overlap(self):
        ds = make_grouped(n_samples=20, obs_per=3)
        train, valid = split_by_sample_id(ds, train_ratio=0.7, random_state=1)
        train_ids = set(train.sample_ids.tolist())
        valid_ids = set(valid.sample_ids.tolist())
        assert train_ids.isdisjoint(valid_ids)

    def test_reproducible_split(self):
        ds = make_grouped(n_samples=12, obs_per=4)
        t1, v1 = split_by_sample_id(ds, train_ratio=0.75, random_state=7)
        t2, v2 = split_by_sample_id(ds, train_ratio=0.75, random_state=7)
        assert t1.sample_ids.tolist() == t2.sample_ids.tolist()
        assert v1.sample_ids.tolist() == v2.sample_ids.tolist()

    def test_too_few_samples_raises(self):
        ids = torch.tensor([0.0, 0.0, 0.0])
        locs = torch.rand(3, 1)
        vals = torch.rand(3)
        ds = GroupedObservationDataset.from_split(ids, locs, vals)
        with pytest.raises(ValueError, match="2 sample"):
            split_by_sample_id(ds)
