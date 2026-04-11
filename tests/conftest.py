from __future__ import annotations

import pytest
import torch


@pytest.fixture
def small_dataset():
    """Small synthetic dataset: 20 samples x 10 obs each, 1-D locations."""
    torch.manual_seed(0)
    n_samples = 20
    obs_per = 10
    n_obs = n_samples * obs_per
    sample_ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locations = torch.rand(n_obs, 1)
    values = torch.sin(2 * 3.14159 * locations.squeeze()) + 0.05 * torch.randn(n_obs)
    return sample_ids, locations, values


@pytest.fixture
def packed_dataset(small_dataset):
    """Packed tensor representation of small_dataset."""
    sample_ids, locations, values = small_dataset
    return torch.cat([sample_ids.unsqueeze(-1), locations, values.unsqueeze(-1)], dim=1)


@pytest.fixture
def fitted_result(small_dataset):
    """A pre-fitted IrregPCAResult on the small dataset."""
    from irregpca import fit_irreg_pca
    sample_ids, locations, values = small_dataset
    return fit_irreg_pca(
        sample_ids=sample_ids,
        locations=locations,
        values=values,
        n_components=2,
        epochs=5,
        patience=100,
        random_state=42,
        verbose=False,
    )
