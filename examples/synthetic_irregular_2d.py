"""
synthetic_irregular_2d.py
-------------------------
Fit IrregPCA to scalar functional data observed on an irregular subset of
[0,1]^2 (a 2-D domain).  Demonstrates the monte_carlo integration mode,
which is required for d > 1.
"""

import torch
import matplotlib.pyplot as plt
from irregpca import fit_irreg_pca

# ── Synthetic data ────────────────────────────────────────────────────────────
# N_CURVES curves, each observed at a random number of (s, t) locations in
# [0,1]^2.  The latent process is:
#   X_i(s, t) = a_i * cos(π s) * cos(π t)  +  b_i * sin(2π s) * cos(π t)
# so the first two PC functions are the two trigonometric basis functions above.

torch.manual_seed(0)

N_CURVES = 200
MAX_OBS_PER_CURVE = 40

sample_ids_list, locations_list, values_list = [], [], []

for i in range(N_CURVES):
    n_obs = torch.randint(10, MAX_OBS_PER_CURVE + 1, ()).item()
    locs  = torch.rand(n_obs, 2)                      # (n_obs, 2) in [0,1]^2
    a = torch.randn(1)
    b = torch.randn(1)
    s, t  = locs[:, 0], locs[:, 1]
    vals  = (a * torch.cos(torch.pi * s) * torch.cos(torch.pi * t)
           + b * torch.sin(2 * torch.pi * s) * torch.cos(torch.pi * t)
           + 0.05 * torch.randn(n_obs))
    sample_ids_list.append(torch.full((n_obs,), i, dtype=torch.long))
    locations_list.append(locs)
    values_list.append(vals)

sample_ids = torch.cat(sample_ids_list)   # (N,)
locations  = torch.cat(locations_list)    # (N, 2)
values     = torch.cat(values_list)       # (N,)

# ── Fit ───────────────────────────────────────────────────────────────────────
result = fit_irreg_pca(
    sample_ids=sample_ids,
    locations=locations,
    values=values,
    n_components=2,
    epochs=800,
    lr=1e-3,
    patience=400,
    valid_split=0.2,
    integration_mode="monte_carlo",   # required for d > 1
    random_state=42,
    verbose=True,
)

print("Best epochs         :", result.history.best_epochs)
print("Component norms     :", result.component_norms())
print("Orthogonality matrix:\n", result.orthogonality_matrix())
print("Explained variance  :", result.explained_variance_proxy())

# ── Visualise on a regular grid ───────────────────────────────────────────────
res = 40
s_vals = torch.linspace(0, 1, res)
t_vals = torch.linspace(0, 1, res)
ss, tt = torch.meshgrid(s_vals, t_vals, indexing="ij")
grid = torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1)  # (res*res, 2)

with torch.no_grad():
    mu   = result.mean(grid).reshape(res, res)
    phi1 = result.component(0, grid).reshape(res, res)
    phi2 = result.component(1, grid).reshape(res, res)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, surface, title in zip(
    axes,
    [mu, phi1, phi2],
    ["Mean function  μ(s,t)", "Component 1  φ₁(s,t)", "Component 2  φ₂(s,t)"],
):
    im = ax.imshow(
        surface.numpy(), origin="lower", extent=[0, 1, 0, 1],
        aspect="equal", cmap="RdBu_r",
    )
    ax.set_title(title)
    ax.set_xlabel("s")
    ax.set_ylabel("t")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("irregpca_2d_example.png", dpi=150)
plt.show()
