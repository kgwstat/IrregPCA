from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from irregpca import IrregPCAResult


class TestResultSummaries:
    def test_component_norms_shape(self, fitted_result):
        norms = fitted_result.component_norms()
        assert norms.shape == (2,)
        assert (norms >= 0).all()

    def test_principal_values_shape(self, fitted_result):
        pv = fitted_result.principal_values()
        assert pv.shape == (2,)

    def test_explained_variance_sums_to_one(self, fitted_result):
        ev = fitted_result.explained_variance_proxy()
        assert abs(float(ev.sum()) - 1.0) < 1e-5

    def test_orthogonality_matrix_shape(self, fitted_result):
        G = fitted_result.orthogonality_matrix()
        assert G.shape == (2, 2)

    def test_orthogonality_matrix_symmetric(self, fitted_result):
        G = fitted_result.orthogonality_matrix()
        assert torch.allclose(G, G.T, atol=1e-5)


class TestResultSerialisation:
    def test_save_and_load(self, fitted_result):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "result.pt"
            fitted_result.save(path)
            loaded = IrregPCAResult.load(path)
        assert loaded.n_components == fitted_result.n_components
        grid = torch.linspace(0, 1, 20).unsqueeze(-1)
        assert torch.allclose(
            loaded.mean(grid),
            fitted_result.mean(grid),
            atol=1e-5,
        )
