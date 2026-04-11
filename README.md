# IrregPCA

Neural-network functional PCA for irregularly observed data.

IrregPCA fits a mean function and a sequence of orthogonal principal component
functions to scalar functional data observed at **irregular, possibly
subject-specific** locations. All functions are represented by small multilayer
perceptrons trained sequentially via gradient descent, based on a risk
minimisation methodology derived from the Eckart–Young–Mirsky theorem for the
Hilbert–Schmidt norm.

## Installation

```bash
pip install irregpca
```

For development:

```bash
git clone https://github.com/kgwaghmare/irregpca
cd irregpca
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.22.

## Quick start

### Split-input format

```python
import torch
from irregpca import fit_irreg_pca

# sample_ids: (N,)   — integer sample/curve identifier per observation
# locations:  (N, d) — observation location in R^d
# values:     (N,)   — scalar observation value

result = fit_irreg_pca(
    sample_ids=sample_ids,
    locations=locations,
    values=values,
    n_components=3,
    epochs=600,
    random_state=42,
)

grid = torch.linspace(0, 1, 500).unsqueeze(-1)
mu        = result.mean(grid)           # (500,)
phi1      = result.component(0, grid)   # (500,)
all_comps = result.components(grid)     # (500, 3)
```

### Packed-input format

```python
# data: (N, d+2) — columns [sample_id, loc_1, ..., loc_d, value]
result = fit_irreg_pca(data=data, n_components=2)
```

### Using a config object

```python
from irregpca import IrregPCA, IrregPCAConfig

cfg = IrregPCAConfig(
    n_components=3,
    epochs=600,
    lr=1e-3,
    patience=300,
    valid_split=0.2,
    random_state=0,
    verbose=True,
)
est = IrregPCA(config=cfg)
result = est.fit(sample_ids=..., locations=..., values=...)
```

## Training modes

| Mode | Description |
|------|-------------|
| `"full_batch"` | Default. All training samples used each epoch. Exact empirical objective. |
| `"mini_batch"` | Grouped mini-batches by sample ID. Scalable to large datasets. |
| `"streaming"` | For memory-mapped / disk-backed datasets. |

```python
result = fit_irreg_pca(..., training_mode="full_batch")
```

## Integration / inner-product modes

Inner products between functions (for regularisation and orthogonality) are
computed using a configurable integration measure.

| Mode | Description | Exact? |
|------|-------------|--------|
| `"grid"` | Uniform quadrature on [0,1]. Default. d=1 only. | Quadrature approximation |
| `"monte_carlo"` | Monte Carlo draws from a sampler. Arbitrary d. | Unbiased stochastic estimator |
| `"weighted_discrete"` | User-supplied nodes + weights. Arbitrary d. | Exact under the given measure |

```python
# Grid (default, d=1)
result = fit_irreg_pca(..., integration_mode="grid", quadrature_points=4096)

# Custom weighted discrete measure
from irregpca.objectives.quadrature import WeightedDiscreteMeasure
pts = torch.tensor([[0.25], [0.5], [0.75]])
wts = torch.tensor([1.0, 2.0, 1.0])
measure = WeightedDiscreteMeasure(pts, wts)
result = fit_irreg_pca(..., measure=measure)
```

## Result summaries

```python
# Norms of component functions under the integration measure
result.component_norms()            # (n_components,)

# Proxy for principal values (squared norms)
result.principal_values()           # (n_components,)

# Fraction of total variance explained per component
result.explained_variance_proxy()   # (n_components,), sums to 1

# Gram matrix of inner products between component functions
result.orthogonality_matrix()       # (n_components, n_components)

# Best epoch per model (mean + components)
result.history.best_epochs          # list of ints
```

## Serialisation

```python
result.save("my_result.pt")
result = IrregPCAResult.load("my_result.pt")

# Move all models to a different device
result.to("cpu")
```

## Large datasets

For datasets too large to fit in GPU memory, use the `data` layer directly:

```python
from irregpca.data import (
    GroupedObservationDataset, GroupedMapDataset,
    grouped_collate_fn, GroupedBatchSampler,
)
from torch.utils.data import DataLoader

ds     = GroupedObservationDataset.from_split(sample_ids, locations, values)
map_ds = GroupedMapDataset(ds)
sampler = GroupedBatchSampler(n_samples=len(map_ds), batch_size=32)
loader  = DataLoader(map_ds, batch_sampler=sampler, collate_fn=grouped_collate_fn)

for batch in loader:
    # batch keys: "sample_ids", "locations", "values",
    #             "group_offsets", "group_lengths"
    ...
```

For on-disk memory-mapped data:

```python
from irregpca.data.memmap import load_memmap_dataset

ds = load_memmap_dataset(
    locations_path="locs.bin",
    values_path="vals.bin",
    sample_ids_path="ids.bin",
    input_dim=1,
    n_obs=1_000_000,
    n_samples=10_000,
)
```

## Current limitations

- The `"grid"` integration mode supports **d = 1** only. For `d > 1`, use
  `"monte_carlo"` or `"weighted_discrete"` with an appropriate domain sampler —
  and note these are approximate.
- Mini-batch and streaming modes use stochastic approximations to the full-batch
  objective. Results may differ from full-batch training on identical data.
- GPU reproducibility may vary across hardware and PyTorch versions even under a
  fixed seed.

## License

MIT. See [LICENSE](LICENSE).
