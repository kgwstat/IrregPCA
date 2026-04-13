# CLAUDE.md — IrregPCA Operating Manual

## Mission

Maintain **IrregPCA** as a research-grade Python package that is:

- scientifically reproducible,
- easy to install and use,
- internally coherent,
- well tested,
- well documented,
- citable and releasable.

---

## Definition of "research-grade"

A change is complete only when it improves one or more of the following without
regressing the others:

1. **Scientific correctness** — results are numerically sane, assumptions are
   documented, evaluation metrics are justified, limitations are stated.
2. **Reproducibility** — experiments can be rerun from the repository, seeds are
   controlled, benchmark results are regenerable from committed scripts.
3. **Software quality** — public API is stable and documented, tests cover core
   behaviors and failure modes, CI enforces quality gates.
4. **Usability** — new users can install, run an example, and understand outputs
   quickly; README matches code; errors are descriptive.
5. **Maintainability** — one canonical implementation path, modules have clear
   responsibilities, legacy code is removed or explicitly deprecated.

---

## Non-negotiable repository standards

### 1. README and code must agree

Any time the public API changes, update all of the following together:
`README.md`, docstrings, examples, tests, and `CHANGELOG.md`.

If README examples do not run against the current package, that is a release
blocker.

### 2. One canonical API

There must be exactly **one recommended user-facing fit path**.

- `IrregPCA(...).fit(...)` — primary object-oriented interface.
- `fit_irreg_pca(...)` — thin functional convenience wrapper.

Do not maintain multiple overlapping fitting paths unless one is explicitly
marked legacy and scheduled for removal.

### 3. No silent shape ambiguity

For every accepted input format: document exact array shape, validate shape at
runtime, raise precise error messages, test both valid and invalid cases.

### 4. No feature flags without implementation

Do not expose configuration options unless they are implemented end-to-end and
tested.

### 5. Research claims require evidence

Claims about consistency, scalability, robustness, dimensional support,
performance, or numerical stability must be backed by tests, benchmarks,
experiments, or an explicit statement that the claim is a hypothesis.

---

## Current architecture

```
src/irregpca/
├── __init__.py          # Public exports
├── config.py            # IrregPCAConfig dataclass
├── estimator.py         # IrregPCA, fit_irreg_pca
├── result.py            # IrregPCAResult, LossHistory
├── data/                # Dataset, DataLoader, split, memmap
├── models/              # MLP, factory, base
├── objectives/          # mean, covariance, penalties, quadrature, losses
├── training/            # engine, callbacks, checkpointing, devices, metrics
└── utils/               # random seeding, validation, typing
```

Public API: `IrregPCA`, `IrregPCAConfig`, `IrregPCAResult`, `fit_irreg_pca`.

---

## Integration modes

| Mode | Description | d support | Exact? |
|------|-------------|-----------|--------|
| `"grid"` | Uniform quadrature on [0,1] | d=1 only | Quadrature approximation |
| `"monte_carlo"` | Monte Carlo from a sampler | arbitrary d | Unbiased stochastic |
| `"weighted_discrete"` | User-supplied nodes + weights | arbitrary d | Exact under given measure |

Do not claim general d support until `d > 1` utilities are tested.

---

## Training modes

| Mode | Description |
|------|-------------|
| `"full_batch"` | Exact empirical objective, all samples each epoch. |
| `"mini_batch"` | Grouped mini-batches by sample ID. Approximate, scalable. |
| `"streaming"` | Memory-mapped / disk-backed datasets. Approximate. |

Mini-batch and streaming results are stochastic approximations; document this.

---

## Testing standards

Run the test suite with:

```bash
pytest tests/ -m "not slow"
```

Add `@pytest.mark.slow` to tests that take more than a few seconds.

Required test categories:
- **Unit** — validation, parsing, config, split logic, result accessors.
- **Integration** — end-to-end fit on a tiny synthetic dataset.
- **Property/invariant** — orthogonality, non-negative norms, loss decreases.
- **Regression** — one targeted test per fixed bug.

Coverage target: high on validation, parsing, and result utilities; meaningful
on training orchestration; smoke coverage on numerical training loops.

---

## CI checklist (GitHub Actions)

Every PR must pass:

1. `ruff check src/ tests/` — zero errors.
2. `mypy src/irregpca --ignore-missing-imports` — no new errors.
3. `pytest tests/ -m "not slow"` — all pass.
4. `python -m build` — wheel and sdist build cleanly.

Python version matrix: 3.10, 3.11, 3.12.

---

## Serialisation policy

Preferred pattern: save config, model state dicts, metadata, and package
version. Provide `IrregPCAResult.save(path)` / `IrregPCAResult.load(path)`.

Whole-object `torch.save(self)` is a convenience; label it as such and warn
about cross-version compatibility limits.

---

## Reproducibility standards

- Expose `random_state` in all public entry points.
- Document that exact GPU reproducibility may vary by hardware/backend.
- Every figure or benchmark in docs should be regenerable from a script in
  `scripts/` or `examples/`.

---

## Versioning policy

- **patch** — bug fix / non-breaking internal improvement.
- **minor** — backward-compatible feature or API extension.
- **major** — breaking API change.

Update `CHANGELOG.md` with every release.

---

## Change discipline

Before changing code, ask:

1. Is this fixing a real bug or missing requirement?
2. Does this simplify the public story?
3. Is there one clear acceptance test?

Do not combine architecture rewrites, API redesigns, numerical objective
changes, and docs rewrites in a single PR.

---

## Anti-goals

- Do not add features before tests and docs exist.
- Do not claim general `d` support where only `d=1` utilities are mature.
- Do not keep legacy interfaces alive without a deprecation warning and removal
  target version.
- Do not add configuration options without implementation.
- Do not merge large numerical changes without benchmark evidence.
- Do not use notebooks as the only record of experiments.

---

## Preferred style

- Prefer small, reviewable commits.
- Write docstrings before exposing new API.
- Add a regression test with each bug fix.
- Keep public names minimal.
- Choose explicitness over cleverness.
- Preserve scientific clarity over abstraction.

When uncertain, prefer: correctness → clarity → reproducibility → performance →
convenience.

---

## Done criteria

The repository is research-grade when:

- README matches the code exactly.
- Public API is coherent and stable.
- Legacy/duplicate fit paths are removed or clearly deprecated.
- Dimensional support is accurately documented.
- Core functionality is covered by tests.
- CI is active and blocking.
- Reproducible scripts exist for key examples/benchmarks.
- Packaging and release metadata are complete.
- Users can install, fit, inspect, save, and reload results reliably.
- Scientific limitations are explicit rather than implied away.
