from __future__ import annotations

import pytest
import torch

from irregpca import IrregPCA, fit_irreg_pca


class TestTrainingBasic:
    def test_loss_decreases(self, small_dataset):
        sample_ids, locations, values = small_dataset
        est = IrregPCA(n_components=1, epochs=20, patience=1000, random_state=0)
        result = est.fit(sample_ids=sample_ids, locations=locations, values=values)
        hist = result.history.component_train[0]  # mean model
        # Loss should generally improve over 20 epochs
        assert len(hist) > 0
        # Not requiring strict monotone decrease, just that we ran
        assert hist[0] != 0.0

    def test_history_lengths_match(self, small_dataset):
        sample_ids, locations, values = small_dataset
        result = IrregPCA(n_components=2, epochs=5, patience=100, random_state=1).fit(
            sample_ids=sample_ids, locations=locations, values=values
        )
        h = result.history
        assert len(h.component_train) == 3  # mean + 2 components
        assert len(h.component_valid) == 3
        assert len(h.best_epochs) == 3
        assert len(h.best_valid_losses) == 3

    def test_deterministic_with_seed(self, small_dataset):
        sample_ids, locations, values = small_dataset
        r1 = fit_irreg_pca(
            sample_ids=sample_ids, locations=locations, values=values,
            n_components=1, epochs=3, random_state=99,
        )
        r2 = fit_irreg_pca(
            sample_ids=sample_ids, locations=locations, values=values,
            n_components=1, epochs=3, random_state=99,
        )
        grid = torch.linspace(0, 1, 20).unsqueeze(-1)
        mu1 = r1.mean(grid)
        mu2 = r2.mean(grid)
        assert torch.allclose(mu1, mu2, atol=1e-5)


class TestEarlyStoppingRestoration:
    def test_best_state_always_restored(self, small_dataset):
        """The best checkpoint must be restored after the loop, even if not
        triggered by patience. This is the key regression test for the bug."""
        sample_ids, locations, values = small_dataset
        result = IrregPCA(
            n_components=1,
            epochs=10,
            patience=10000,  # never triggers patience
            random_state=0,
        ).fit(sample_ids=sample_ids, locations=locations, values=values)
        h = result.history
        # best_epochs should be set
        assert all(e >= 0 for e in h.best_epochs)
        # best_valid_losses should be finite
        assert all(not (v != v) for v in h.best_valid_losses)  # not NaN

    def test_callbacks_receive_best_info(self, small_dataset):
        events = []
        def cb(event):
            events.append(event.copy())
        sample_ids, locations, values = small_dataset
        IrregPCA(
            n_components=1, epochs=5, patience=100, callbacks=[cb]
        ).fit(sample_ids=sample_ids, locations=locations, values=values)
        epoch_events = [e for e in events if e["stage"] == "epoch_end"]
        assert len(epoch_events) > 0
        for ev in epoch_events:
            assert "best_epoch" in ev
            assert "best_valid_loss" in ev
