"""Synthetic 1-D irregular functional data experiment.

Generates data from a known 2-component functional PCA model and checks
that the fitted functions recover the true structure.
"""
from __future__ import annotations

import math
import torch
from irregpca import IrregPCA, IrregPCAConfig

torch.manual_seed(42)

# ------------------------------------------------------------------
# 1. True model: mean + 2 orthogonal components on [0,1]
# ------------------------------------------------------------------
def true_mean(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * t)

def true_phi1(t: torch.Tensor) -> torch.Tensor:
    # Normalised to have L2-norm ≈ 1/sqrt(2)
    return torch.cos(2 * math.pi * t)

def true_phi2(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(4 * math.pi * t)

# ------------------------------------------------------------------
# 2. Sample irregular observations
# ------------------------------------------------------------------
n_samples = 50
obs_per = 15
sigma = 0.05

ids_list, locs_list, vals_list = [], [], []
for i in range(n_samples):
    # random scores
    s1 = torch.randn(1).item()
    s2 = torch.randn(1).item()
    t = torch.rand(obs_per, 1)
    y = (
        true_mean(t.squeeze())
        + s1 * true_phi1(t.squeeze())
        + s2 * true_phi2(t.squeeze())
        + sigma * torch.randn(obs_per)
    )
    ids_list.append(torch.full((obs_per,), float(i)))
    locs_list.append(t)
    vals_list.append(y)

sample_ids = torch.cat(ids_list)
locations  = torch.cat(locs_list, dim=0)
values     = torch.cat(vals_list)

# ------------------------------------------------------------------
# 3. Fit with IrregPCAConfig
# ------------------------------------------------------------------
cfg = IrregPCAConfig(
    n_components=2,
    epochs=400,
    lr=1e-3,
    patience=200,
    random_state=42,
    verbose=True,
)
est = IrregPCA(config=cfg)
result = est.fit(sample_ids=sample_ids, locations=locations, values=values)

# ------------------------------------------------------------------
# 4. Compare on a dense grid
# ------------------------------------------------------------------
grid = torch.linspace(0, 1, 500).unsqueeze(-1)
mu_hat = result.mean(grid)
mu_true = true_mean(grid.squeeze())

mse_mean = ((mu_hat - mu_true) ** 2).mean()
print(f"\nMean MSE on grid: {mse_mean:.6f}")
print(f"Best epochs per model: {result.history.best_epochs}")
print(f"Gram matrix of components:\n{result.orthogonality_matrix()}")
