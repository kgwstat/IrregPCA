# Contributing to IrregPCA

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/kgwaghmare/irregpca
cd irregpca
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/                    # all fast tests
pytest tests/ -m "not slow"      # skip slow tests
pytest tests/ --cov=irregpca     # with coverage
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check src/ tests/           # lint
ruff format src/ tests/          # format (optional)
```

Type annotations are checked with mypy:

```bash
mypy src/irregpca --ignore-missing-imports
```

## Project structure

```
src/irregpca/
  config.py          # IrregPCAConfig dataclass
  estimator.py       # Public IrregPCA estimator and fit_irreg_pca
  result.py          # IrregPCAResult and LossHistory
  data/              # Dataset abstractions (GroupedObservationDataset, etc.)
  models/            # Neural network models (DefaultModel, factory)
  objectives/        # Loss functions and integration measures
  training/          # Training engine, early stopping, callbacks
  utils/             # Seeding, validation helpers
tests/               # pytest test suite
examples/            # Runnable example scripts
```

## Design principles

- **Correctness first.** Do not change the mathematical objective without a
  documented rationale and new tests.
- **Explicit over implicit.** Stochastic approximations must be labelled as
  such; device transfers must be intentional.
- **No silent data copies.** Train/validation splitting must not duplicate the
  flat observation storage.
- **Small pure functions.** Keep objective code separate from training
  orchestration.

## Adding a new integration measure

1. Implement the `InnerProductMeasure` protocol in
   `src/irregpca/objectives/quadrature.py`.
2. Add tests in `tests/test_measure.py` verifying correctness against known
   integrals.
3. Update `make_measure` or document that it must be passed explicitly.

## Reporting bugs

Please open an issue at https://github.com/kgwaghmare/irregpca/issues with:
- a minimal reproducible example,
- the Python and PyTorch versions,
- the device being used (CPU / CUDA / MPS).
