"""Basic usage on a small in-memory dataset.

This example shows the minimal workflow:
1. Prepare split-format input tensors.
2. Fit IrregPCA.
3. Evaluate the mean and component functions on a grid.
"""
from __future__ import annotations

import torch
from irregpca import fit_irreg_pca

torch.manual_seed(0)

# ------------------------------------------------------------------
# 1. Synthetic irregular data
# ------------------------------------------------------------------
n_samples = 30
obs_per = 8

sample_ids = torch.repeat_interleave(
    torch.arange(n_samples, dtype=torch.float), obs_per
)
locations = torch.rand(n_samples * obs_per, 1)
values = (
    torch.sin(2 * torch.pi * locations.squeeze())
    + 0.1 * torch.randn(n_samples * obs_per)
)

# ------------------------------------------------------------------
# 2. Fit
# ------------------------------------------------------------------
result = fit_irreg_pca(
    sample_ids=sample_ids,
    locations=locations,
    values=values,
    n_components=2,
    epochs=200,
    lr=1e-3,
    patience=100,
    random_state=0,
    verbose=True,
)

# ------------------------------------------------------------------
# 3. Evaluate on a grid
# ------------------------------------------------------------------
grid = torch.linspace(0, 1, 200).unsqueeze(-1)

mu = result.mean(grid)
phi1 = result.component(0, grid)
phi2 = result.component(1, grid)

print(f"\nMean range:       [{mu.min():.3f}, {mu.max():.3f}]")
print(f"Component 1 norm: {result.component_norms()[0]:.4f}")
print(f"Component 2 norm: {result.component_norms()[1]:.4f}")
print(f"Explained variance: {result.explained_variance_proxy().tolist()}")
