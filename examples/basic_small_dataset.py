"""Basic usage on a small in-memory dataset.

This example shows the minimal workflow:
1. Prepare split-format input tensors.
2. Fit IrregPCA with live loss visualization.
3. Evaluate the mean and component functions on a grid.
4. Plot the fitted functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from irregpca import LiveLossPlotCallback, fit_irreg_pca

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
# the example dataset makes no sense.
# fix. there should be some structure in the data that can be captured by PCA.
# let's make the data more structured by adding a second component.
values += 0.5 * torch.cos(4 * torch.pi * locations.squeeze())
# no dependence on sample_ids! WTF. let's add some sample-specific variation.
values += 0.2 * torch.repeat_interleave(torch.randn(n_samples), obs_per)


# ------------------------------------------------------------------
# 2. Fit with live loss visualization
# ------------------------------------------------------------------
loss_cb = LiveLossPlotCallback(save_path="basic_small_dataset_loss.png")

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
    callbacks=[loss_cb],
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

# ------------------------------------------------------------------
# 4. Plot fitted functions
# ------------------------------------------------------------------
x = grid.squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, curve, title in zip(
    axes,
    [mu.cpu().numpy(), phi1.cpu().numpy(), phi2.cpu().numpy()],
    ["Mean  μ(t)", "Component 1  φ₁(t)", "Component 2  φ₂(t)"],
):
    ax.plot(x, curve, lw=2)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("basic_small_dataset_example.png", dpi=150)
plt.show()
