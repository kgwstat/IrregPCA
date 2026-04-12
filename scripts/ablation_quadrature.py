"""Ablation: sensitivity of orthogonality to quadrature resolution.

Trains a 2-component IrregPCA model on a tiny synthetic dataset under
several values of `quadrature_points` and records:

  - max off-diagonal entry in the Gram matrix (orthogonality error),
  - component norms (expected ≈ 1).

Usage
-----
    python scripts/ablation_quadrature.py [--seed S]

Output is printed as a CSV-style table.
"""
from __future__ import annotations

import argparse
import math

import torch

from irregpca import IrregPCA, IrregPCAConfig


def make_data(
    n_samples: int = 60,
    obs_per: int = 20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    ids_list, locs_list, vals_list = [], [], []
    for i in range(n_samples):
        s1, s2 = torch.randn(2).tolist()
        t = torch.rand(obs_per, 1)
        ts = t.squeeze(-1)
        y = (
            torch.sin(2 * math.pi * ts)
            + s1 * torch.cos(2 * math.pi * ts)
            + s2 * torch.sin(4 * math.pi * ts)
            + 0.05 * torch.randn(obs_per)
        )
        ids_list.append(torch.full((obs_per,), float(i)))
        locs_list.append(t)
        vals_list.append(y)
    return (
        torch.cat(ids_list),
        torch.cat(locs_list, dim=0),
        torch.cat(vals_list),
    )


def run(
    sample_ids: torch.Tensor,
    locations: torch.Tensor,
    values: torch.Tensor,
    quadrature_points: int,
    seed: int,
) -> dict:
    cfg = IrregPCAConfig(
        n_components=2,
        epochs=300,
        lr=1e-3,
        patience=150,
        valid_split=0.2,
        random_state=seed,
        integration_mode="grid",
        quadrature_points=quadrature_points,
        verbose=False,
    )
    result = IrregPCA(config=cfg).fit(
        sample_ids=sample_ids, locations=locations, values=values
    )
    gram = result.orthogonality_matrix()
    off_diag = gram.clone()
    off_diag.fill_diagonal_(0.0)
    return {
        "quadrature_points": quadrature_points,
        "max_offdiag_gram": off_diag.abs().max().item(),
        "component_norms": result.component_norms().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadrature ablation study")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_ids, locations, values = make_data(seed=args.seed)

    resolutions = [64, 256, 1024, 4096]

    print("quadrature_points,max_offdiag_gram,norm_comp0,norm_comp1")
    for q in resolutions:
        r = run(sample_ids, locations, values, q, args.seed)
        n0, n1 = r["component_norms"]
        print(f"{q},{r['max_offdiag_gram']:.6f},{n0:.4f},{n1:.4f}")


if __name__ == "__main__":
    main()
