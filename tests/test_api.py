from __future__ import annotations

import pytest
import torch

from irregpca import IrregPCA, IrregPCAConfig, IrregPCAResult, fit_irreg_pca


class TestIrregPCAInit:
    def test_basic_init(self):
        est = IrregPCA(n_components=3)
        assert est.n_components == 3

    def test_config_init(self):
        cfg = IrregPCAConfig(n_components=2, epochs=10)
        est = IrregPCA(config=cfg)
        assert est.n_components == 2
        assert est.config.epochs == 10

    def test_missing_n_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            IrregPCA()

    def test_not_fitted_raises(self):
        est = IrregPCA(n_components=2)
        grid = torch.linspace(0, 1, 10).unsqueeze(-1)
        with pytest.raises(RuntimeError, match="not fitted"):
            est.mean(grid)


class TestFitAPI:
    def test_fit_split_input(self, small_dataset):
        sample_ids, locations, values = small_dataset
        result = IrregPCA(n_components=2, epochs=3, patience=100).fit(
            sample_ids=sample_ids, locations=locations, values=values
        )
        assert isinstance(result, IrregPCAResult)

    def test_fit_packed_input(self, packed_dataset):
        result = IrregPCA(n_components=1, epochs=3, patience=100).fit(data=packed_dataset)
        assert isinstance(result, IrregPCAResult)

    def test_fit_missing_args_raises(self):
        est = IrregPCA(n_components=1)
        with pytest.raises(ValueError, match="Provide either"):
            est.fit()

    def test_fit_irreg_pca_convenience(self, small_dataset):
        sample_ids, locations, values = small_dataset
        result = fit_irreg_pca(
            sample_ids=sample_ids, locations=locations, values=values,
            n_components=1, epochs=3, patience=100
        )
        assert isinstance(result, IrregPCAResult)


class TestResultShapes:
    def test_mean_shape(self, fitted_result):
        grid = torch.linspace(0, 1, 100).unsqueeze(-1)
        mu = fitted_result.mean(grid)
        assert mu.shape == (100,)

    def test_component_shape(self, fitted_result):
        grid = torch.linspace(0, 1, 50).unsqueeze(-1)
        phi = fitted_result.component(0, grid)
        assert phi.shape == (50,)

    def test_components_shape(self, fitted_result):
        grid = torch.linspace(0, 1, 50).unsqueeze(-1)
        all_c = fitted_result.components(grid)
        assert all_c.shape == (50, 2)

    def test_all_functions_shape(self, fitted_result):
        grid = torch.linspace(0, 1, 30).unsqueeze(-1)
        all_f = fitted_result.all_functions(grid)
        assert all_f.shape == (30, 3)  # mean + 2 components

    def test_component_invalid_index_raises(self, fitted_result):
        grid = torch.linspace(0, 1, 10).unsqueeze(-1)
        with pytest.raises(ValueError, match="out of range"):
            fitted_result.component(5, grid)

    def test_component_negative_index_raises(self, fitted_result):
        grid = torch.linspace(0, 1, 10).unsqueeze(-1)
        with pytest.raises(ValueError, match="out of range"):
            fitted_result.component(-1, grid)
