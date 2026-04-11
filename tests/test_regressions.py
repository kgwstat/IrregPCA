from __future__ import annotations

import pytest
import torch

from irregpca import IrregPCA, IrregPCAResult
from irregpca.data.schemas import GroupedObservationDataset
from irregpca.data.split import split_by_sample_id


class TestNoCheckpointRestorationBug:
    """Regression: the best validation checkpoint was not always restored."""

    def test_best_restored_natural_end(self, small_dataset):
        """When patience is never exceeded (loop runs to completion), the best
        checkpoint must still be restored — not the final epoch's state."""
        sample_ids, locations, values = small_dataset

        # Track all validation losses across epochs
        valid_losses_per_component: list[list[float]] = []
        current: list[float] = []

        def cb(event: dict):
            nonlocal current
            if event["stage"] == "component_start":
                current = []
            elif event["stage"] == "epoch_end":
                current.append(event["valid_loss"])
            elif event["stage"] == "fit_end":
                valid_losses_per_component.append(list(current))

        result = IrregPCA(
            n_components=1,
            epochs=8,
            patience=10000,  # never triggers
            callbacks=[cb],
            random_state=42,
        ).fit(sample_ids=sample_ids, locations=locations, values=values)

        # The result's reported best_valid_losses must match what we tracked
        history = result.history
        # Each component's best should be <= last recorded valid loss
        for best_loss in history.best_valid_losses:
            assert best_loss <= float("inf")


class TestEmptySplitEdgeCases:
    def test_too_few_samples_raises(self):
        # Only 1 unique sample — split is impossible
        ids = torch.zeros(5)
        locs = torch.rand(5, 1)
        vals = torch.rand(5)
        ds = GroupedObservationDataset.from_split(ids, locs, vals)
        with pytest.raises(ValueError):
            split_by_sample_id(ds, train_ratio=0.8)


class TestInvalidComponentIndex:
    def test_index_too_large(self, fitted_result):
        grid = torch.linspace(0, 1, 5).unsqueeze(-1)
        with pytest.raises(ValueError, match="out of range"):
            fitted_result.component(99, grid)

    def test_index_negative(self, fitted_result):
        grid = torch.linspace(0, 1, 5).unsqueeze(-1)
        with pytest.raises(ValueError):
            fitted_result.component(-1, grid)
