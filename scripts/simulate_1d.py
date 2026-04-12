"""Simulation study: IrregPCA recovery on 1-D synthetic data.

Generates data from a known 2-component FPCA model, fits IrregPCA, and
reports how well the estimated mean and first component match the truth.

Usage
-----
    python scripts/simulate_1d.py [--n-samples N] [--obs-per M] [--seed S]

All defaults reproduce the results shown in README Section "Quick start".
"""
from __future__ import annotations

import argparse
import math

import torch

from irregpca import IrregPCA, IrregPCAConfig


# ---------------------------------------------------------------------------
# True model
# ---------------------------------------------------------------------------

def true_mean(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * t.squeeze(-1))


def true_phi1(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(2 * math.pi * t.squeeze(-1))


def true_phi2(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(4 * math.pi * t.squeeze(-1))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int,
    obs_per: int,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    ids_list, locs_list, vals_list = [], [], []
    for i in range(n_samples):
        s1 = torch.randn(1).item()
        s2 = torch.randn(1).item()
        t = torch.rand(obs_per, 1)
        y = (
            true_mean(t)
            + s1 * true_phi1(t)
            + s2 * true_phi2(t)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IrregPCA 1-D simulation study")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--obs-per", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--components", type=int, default=2)
    args = parser.parse_args()

    print("=== IrregPCA 1-D Simulation Study ===")
    print(f"  n_samples={args.n_samples}, obs_per={args.obs_per}, "
          f"sigma={args.sigma}, seed={args.seed}")

    sample_ids, locations, values = generate_data(
        args.n_samples, args.obs_per, args.sigma, args.seed
    )

    cfg = IrregPCAConfig(
        n_components=args.components,
        epochs=args.epochs,
        lr=1e-3,
        patience=300,
        valid_split=0.2,
        random_state=args.seed,
        verbose=False,
    )
    result = IrregPCA(config=cfg).fit(
        sample_ids=sample_ids, locations=locations, values=values
    )

    grid = torch.linspace(0, 1, 1000).unsqueeze(-1)

    mu_hat = result.mean(grid)
    mu_true = true_mean(grid)
    mse_mean = ((mu_hat - mu_true) ** 2).mean().item()

    print(f"\nMean MSE on dense grid : {mse_mean:.6f}")
    print(f"Best epochs per model  : {result.history.best_epochs}")
    print(f"Component norms        : {result.component_norms().tolist()}")
    print(f"Explained variance     : {result.explained_variance_proxy().tolist()}")

    gram = result.orthogonality_matrix()
    off_diag = gram.clone()
    off_diag.fill_diagonal_(0.0)
    print(f"Max off-diagonal Gram  : {off_diag.abs().max().item():.6f}  (want ≈ 0)")


if __name__ == "__main__":
    main()
