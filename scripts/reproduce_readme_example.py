"""Reproduce the README quick-start example exactly.

Running this script confirms that the canonical API shown in the README works
from end to end. It should complete in under 60 seconds on CPU.

Usage
-----
    python scripts/reproduce_readme_example.py
"""
from __future__ import annotations

import math
import sys

import torch

from irregpca import IrregPCAConfig, IrregPCA, fit_irreg_pca


def make_data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small synthetic dataset: 40 curves, 15 obs each, 2 latent components."""
    torch.manual_seed(seed)
    n_samples, obs_per, sigma = 40, 15, 0.05
    ids_list, locs_list, vals_list = [], [], []
    for i in range(n_samples):
        s1, s2 = torch.randn(2).tolist()
        t = torch.rand(obs_per, 1)
        y = (
            torch.sin(2 * math.pi * t.squeeze(-1))
            + s1 * torch.cos(2 * math.pi * t.squeeze(-1))
            + s2 * torch.sin(4 * math.pi * t.squeeze(-1))
            + sigma * torch.randn(obs_per)
        )
        ids_list.append(torch.full((obs_per,), float(i)))
        locs_list.append(t)
        vals_list.append(y)
    return (
        torch.cat(ids_list),
        torch.cat(locs_list, dim=0),
        torch.cat(vals_list),
    )


def main() -> None:
    print("=== Reproducing README quick-start example ===\n")

    sample_ids, locations, values = make_data(seed=42)

    # ------------------------------------------------------------------
    # Functional convenience wrapper (split-input format)
    # ------------------------------------------------------------------
    print("1. fit_irreg_pca (split-input format) ...")
    result = fit_irreg_pca(
        sample_ids=sample_ids,
        locations=locations,
        values=values,
        n_components=3,
        epochs=300,
        random_state=42,
    )

    grid = torch.linspace(0, 1, 500).unsqueeze(-1)
    mu = result.mean(grid)
    phi1 = result.component(0, grid)
    all_comps = result.components(grid)

    assert mu.shape == (500,), f"Expected (500,), got {mu.shape}"
    assert phi1.shape == (500,), f"Expected (500,), got {phi1.shape}"
    assert all_comps.shape == (500, 3), f"Expected (500, 3), got {all_comps.shape}"
    print(f"   mean shape:       {tuple(mu.shape)}  ✓")
    print(f"   component shape:  {tuple(phi1.shape)}  ✓")
    print(f"   components shape: {tuple(all_comps.shape)}  ✓")

    # ------------------------------------------------------------------
    # Packed-input format
    # ------------------------------------------------------------------
    print("\n2. fit_irreg_pca (packed-input format) ...")
    data = torch.cat([sample_ids.unsqueeze(-1), locations, values.unsqueeze(-1)], dim=1)
    result2 = fit_irreg_pca(data=data, n_components=2, epochs=300, random_state=42)
    assert result2.mean(grid).shape == (500,)
    print("   packed-input fit completed  ✓")

    # ------------------------------------------------------------------
    # Config object API
    # ------------------------------------------------------------------
    print("\n3. IrregPCA with IrregPCAConfig ...")
    cfg = IrregPCAConfig(
        n_components=2,
        epochs=300,
        lr=1e-3,
        patience=150,
        valid_split=0.2,
        random_state=0,
        verbose=False,
    )
    est = IrregPCA(config=cfg)
    result3 = est.fit(sample_ids=sample_ids, locations=locations, values=values)
    norms = result3.component_norms()
    assert norms.shape == (2,)
    print(f"   component norms: {norms.tolist()}  ✓")

    # ------------------------------------------------------------------
    # Result summaries
    # ------------------------------------------------------------------
    print("\n4. Result summaries ...")
    gram = result3.orthogonality_matrix()
    assert gram.shape == (2, 2)
    ev = result3.explained_variance_proxy()
    assert ev.shape == (2,)
    assert abs(ev.sum().item() - 1.0) < 1e-5
    print(f"   explained variance proxy: {ev.tolist()}  ✓")

    print("\nAll assertions passed. README example is consistent with the current code.")
    sys.exit(0)


if __name__ == "__main__":
    main()
