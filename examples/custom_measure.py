"""Custom integration measure example.

Demonstrates:
- Using WeightedDiscreteMeasure with user-specified quadrature nodes.
- Using MonteCarloMeasure with a custom domain sampler.
- Passing a custom measure directly to IrregPCA.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from irregpca import IrregPCA, LiveLossPlotCallback
from irregpca.objectives.quadrature import (
    MonteCarloMeasure,
    WeightedDiscreteMeasure,
)

torch.manual_seed(0)

# ------------------------------------------------------------------
# Synthetic data
# ------------------------------------------------------------------
n_samples = 20
obs_per = 10
sample_ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
locations   = torch.rand(n_samples * obs_per, 1)
values = (
    torch.sin(2 * torch.pi * locations.squeeze()) + 0.05 * torch.randn(n_samples * obs_per)
)

# ------------------------------------------------------------------
# 1. Weighted discrete measure: 5-point Gauss-Legendre-like nodes on [0,1]
# ------------------------------------------------------------------
gl_nodes   = torch.tensor([[0.0469101], [0.2307653], [0.5], [0.7692347], [0.9530899]])
gl_weights = torch.tensor([0.1184634, 0.2393143, 0.2844444, 0.2393143, 0.1184634])

weighted_measure = WeightedDiscreteMeasure(points=gl_nodes, weights=gl_weights)

loss_cb_wt = LiveLossPlotCallback(save_path="custom_measure_weighted_loss.png")

result_wt = IrregPCA(
    n_components=1, epochs=20, patience=100, random_state=0,
    measure=weighted_measure, callbacks=[loss_cb_wt],
).fit(sample_ids=sample_ids, locations=locations, values=values)

grid = torch.linspace(0, 1, 100).unsqueeze(-1)
print("Weighted discrete measure — component norm:",
      result_wt.component_norms().item())

# ------------------------------------------------------------------
# 2. Monte Carlo measure: uniform on [0,1]
# ------------------------------------------------------------------
def uniform_sampler(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 1, device=device)

mc_measure = MonteCarloMeasure(sampler=uniform_sampler, num_draws=2048)

loss_cb_mc = LiveLossPlotCallback(save_path="custom_measure_mc_loss.png")

result_mc = IrregPCA(
    n_components=1, epochs=20, patience=100, random_state=0,
    measure=mc_measure, callbacks=[loss_cb_mc],
).fit(sample_ids=sample_ids, locations=locations, values=values)

print("Monte Carlo measure — component norm:",
      result_mc.component_norms().item())

# ------------------------------------------------------------------
# 3. Plot fitted component functions from both measures
# ------------------------------------------------------------------
x = grid.squeeze().cpu().numpy()
phi_wt = result_wt.component(0, grid).cpu().numpy()
phi_mc = result_mc.component(0, grid).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, phi, title in zip(
    axes,
    [phi_wt, phi_mc],
    ["Weighted discrete measure  φ₁(t)", "Monte Carlo measure  φ₁(t)"],
    strict=False,
):
    ax.plot(x, phi, lw=2)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("custom_measure_example.png", dpi=150)
plt.show()
