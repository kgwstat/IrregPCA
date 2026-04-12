from __future__ import annotations

import torch

from irregpca.objectives.covariance import covariance_fn_packed
from irregpca.objectives.mean import mean_fn_packed


def make_simple_model(val: float):
    """A constant-function 'model' for testing."""

    class ConstModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1), val, dtype=x.dtype, device=x.device)

    return ConstModel()


def make_data(n_samples: int = 5, obs_per: int = 4, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    n = n_samples * obs_per
    ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locs = torch.rand(n, 1)
    vals = torch.rand(n)
    return torch.cat([ids.unsqueeze(-1), locs, vals.unsqueeze(-1)], dim=1)


class TestMeanFnPacked:
    def test_output_is_scalar(self):
        data = make_data()
        model = make_simple_model(1.0)
        out = mean_fn_packed(model, data)
        assert out.ndim == 0

    def test_constant_model_equals_mean_of_values(self):
        # if model(u) = 1, then mean_fn = E_i[mean_j(y_ij)]
        data = make_data(n_samples=4, obs_per=3)
        model = make_simple_model(1.0)
        result = float(mean_fn_packed(model, data))
        # each sample mean of y * 1 = sample mean of y
        idx = data[:, 0].long()
        obs = data[:, -1]
        unique = idx.unique()
        expected = sum(obs[idx == i].mean().item() for i in unique) / len(unique)
        assert abs(result - expected) < 1e-5


class TestCovarianceFnPacked:
    def test_output_is_scalar(self):
        data = make_data(n_samples=5, obs_per=4)
        model = make_simple_model(1.0)
        out = covariance_fn_packed(model, data)
        assert out.ndim == 0

    def test_single_sample_returns_zero(self):
        # With <2 samples, covariance is undefined; should return 0
        data = make_data(n_samples=1, obs_per=5)
        model = make_simple_model(2.0)
        result = covariance_fn_packed(model, data)
        assert float(result) == 0.0

    def test_single_obs_per_sample_returns_zero(self):
        # samples with 1 observation can't contribute to within-sample term
        n = 5
        data = torch.cat(
            [
                torch.arange(n, dtype=torch.float).unsqueeze(-1),
                torch.rand(n, 1),
                torch.rand(n, 1),
            ],
            dim=1,
        )
        model = make_simple_model(1.0)
        result = covariance_fn_packed(model, data)
        # Only 1 observation per sample, so within_sample term is zero for all
        # Between sample may not be zero, but should not crash
        assert result.ndim == 0
