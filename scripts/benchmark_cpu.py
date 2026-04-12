"""CPU training benchmark: wall time vs dataset size.

Sweeps over different sample counts and observations-per-sample and records
the time to complete a short training run. Useful for understanding scaling
before committing to large experiments.

Usage
-----
    python scripts/benchmark_cpu.py [--epochs N] [--components K]

Output is printed as a CSV-style table so it can be piped to a file:

    python scripts/benchmark_cpu.py > results/benchmark_cpu.csv
"""
from __future__ import annotations

import argparse
import time

import torch

from irregpca import IrregPCA, IrregPCAConfig


def generate_data(
    n_samples: int,
    obs_per: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float32), obs_per)
    locs = torch.rand(n_samples * obs_per, 1)
    vals = torch.randn(n_samples * obs_per)
    return ids, locs, vals


def run_benchmark(
    n_samples: int,
    obs_per: int,
    epochs: int,
    components: int,
    seed: int = 42,
) -> float:
    sample_ids, locations, values = generate_data(n_samples, obs_per, seed)
    cfg = IrregPCAConfig(
        n_components=components,
        epochs=epochs,
        lr=1e-3,
        patience=epochs,  # no early stopping so timing is deterministic
        valid_split=0.2,
        random_state=seed,
        verbose=False,
    )
    t0 = time.perf_counter()
    IrregPCA(config=cfg).fit(sample_ids=sample_ids, locations=locations, values=values)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="IrregPCA CPU training benchmark")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per configuration (default: 50)")
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_samples_grid = [20, 50, 100, 200]
    obs_per_grid = [10, 20, 50]

    header = "n_samples,obs_per,n_obs_total,epochs,wall_time_s"
    print(header)

    for n_samples in n_samples_grid:
        for obs_per in obs_per_grid:
            elapsed = run_benchmark(
                n_samples=n_samples,
                obs_per=obs_per,
                epochs=args.epochs,
                components=args.components,
                seed=args.seed,
            )
            print(f"{n_samples},{obs_per},{n_samples * obs_per},{args.epochs},{elapsed:.3f}")


if __name__ == "__main__":
    main()
