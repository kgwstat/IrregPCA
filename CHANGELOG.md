# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `scripts/` directory with reproducible simulation and benchmark scripts.

### Changed
- Ruff lint errors resolved across `src/` and `tests/`.

---

## [0.2.0] — 2026-04-11

Major research-grade refactor. All changes relative to the 0.1.x prototype.

### Added
- `IrregPCAConfig` dataclass for structured, validated configuration.
- `data/` subpackage: `GroupedObservationDataset`, `GroupedMapDataset`,
  `GroupedBatchSampler`, `grouped_collate_fn`, `split_by_sample_id`,
  and a memory-mapped dataset loader (`data.memmap`).
- `objectives/` subpackage: `mean.py`, `covariance.py`, `penalties.py`,
  `quadrature.py` (including `UnitIntervalGridMeasure`,
  `WeightedDiscreteMeasure`, `MonteCarloMeasure`), and `losses.py`.
- `training/` subpackage: `engine.py` (sequential component training loop),
  `callbacks.py` (typed event callbacks), `checkpointing.py`, `devices.py`,
  and `metrics.py` (epoch timing and step counts).
- `models/` subpackage: `base.py`, `mlp.py` (default MLP), `factory.py`.
- `utils/` subpackage: `random.py` (seed control), `validation.py`
  (centralized input and hyperparameter validation), `typing.py`.
- `IrregPCAResult` methods: `component_norms()`, `principal_values()`,
  `explained_variance_proxy()`, `orthogonality_matrix()`, `save()`, `load()`,
  `to(device)`.
- `LossHistory` fields: `best_epochs`, `best_valid_losses`.
- Training modes: `"full_batch"`, `"mini_batch"`, `"streaming"`.
- Integration modes: `"grid"` (d=1 quadrature), `"monte_carlo"`,
  `"weighted_discrete"`.
- Automated test suite: 76 tests across 9 modules.
- GitHub Actions CI: lint, type-check, test (Python 3.10–3.12), build.
- `examples/` directory: `basic_small_dataset.py`,
  `synthetic_irregular_1d.py`, `large_dataset_memmap.py`,
  `custom_measure.py`.
- `CONTRIBUTING.md`, `.editorconfig`, `pytest.ini`, `ruff.toml`.
- `legacy/` directory preserving the pre-refactor implementation.

### Fixed
- **Early-stopping restoration bug**: best model state is now always restored
  after the epoch loop, not only when patience is exhausted mid-run.
- Domain/integration assumptions centralised; `[0, 1]` grid no longer
  hard-coded outside `UnitIntervalGridMeasure`.
- Hyperparameter validation raises descriptive `ValueError` before training.
- Centralized device handling; `device_` exposed as `torch.device`.

### Changed
- Package restructured from a single-file prototype to a `src/` layout with
  five subpackages (`data`, `objectives`, `training`, `models`, `utils`).
- Training loop refactored to consume `GroupedObservationDataset` rather than
  raw tensors.
- Loss functions accept grouped batch dicts instead of the full training tensor.
- Public API exports: `IrregPCA`, `IrregPCAConfig`, `IrregPCAResult`,
  `fit_irreg_pca`.
- `pyproject.toml` updated with complete metadata, classifiers, URLs, and
  optional dependency groups (`dev`, `docs`).

### Deprecated
- `core.py` legacy fitting path (kept in `legacy/` for reference; will be
  removed in 0.3.0).

---

## [0.1.0] — 2026-02-26

Initial research prototype.

### Added
- Basic estimator: `IrregPCA`, `fit_irreg_pca`.
- Default MLP model.
- Sequential component fitting with early stopping.
- Packed-tensor input format `(N, d+2)`.
- MIT License.
