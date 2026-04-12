from __future__ import annotations

import math

import pytest
import torch

from irregpca.objectives.quadrature import (
    MonteCarloMeasure,
    UnitIntervalGridMeasure,
    WeightedDiscreteMeasure,
    make_measure,
)


def constant_model(c: float):
    """Wrapper that behaves like an nn.Module: takes (N,1) → (N,1)."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), c, dtype=x.dtype, device=x.device)

    return f


def sine_model():
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * math.pi * x)  # x is (N,1), result (N,1)

    return f


class TestUnitIntervalGridMeasure:
    def test_constant_one(self):
        m = UnitIntervalGridMeasure(num_points=4096)
        ip = m.integrate_product(constant_model(1.0), constant_model(1.0), torch.device("cpu"))
        assert abs(float(ip) - 1.0) < 1e-4

    def test_constant_times_constant(self):
        m = UnitIntervalGridMeasure(num_points=4096)
        ip = m.integrate_product(constant_model(2.0), constant_model(3.0), torch.device("cpu"))
        assert abs(float(ip) - 6.0) < 1e-4

    def test_sine_self_inner_product(self):
        # ∫_0^1 sin^2(2πx) dx = 0.5
        m = UnitIntervalGridMeasure(num_points=8192)
        ip = m.integrate_product(sine_model(), sine_model(), torch.device("cpu"))
        assert abs(float(ip) - 0.5) < 1e-3

    def test_invalid_num_points(self):
        with pytest.raises(ValueError):
            UnitIntervalGridMeasure(num_points=1)


class TestWeightedDiscreteMeasure:
    def test_constant_integration(self):
        pts = torch.tensor([[0.25], [0.50], [0.75]])
        wts = torch.tensor([1.0, 1.0, 1.0])
        m = WeightedDiscreteMeasure(pts, wts)
        ip = m.integrate_product(constant_model(2.0), constant_model(3.0), torch.device("cpu"))
        # weights normalised to [1/3, 1/3, 1/3], integral = 2*3 = 6
        assert abs(float(ip) - 6.0) < 1e-6

    def test_negative_weights_raises(self):
        pts = torch.tensor([[0.5]])
        wts = torch.tensor([-1.0])
        with pytest.raises(ValueError, match="non-negative"):
            WeightedDiscreteMeasure(pts, wts)


class TestMonteCarloMeasure:
    def test_constant_approx(self):
        torch.manual_seed(42)

        def sampler(n, device):
            return torch.rand(n, 1, device=device)

        m = MonteCarloMeasure(sampler=sampler, num_draws=10000)
        ip = m.integrate_product(constant_model(1.0), constant_model(1.0), torch.device("cpu"))
        assert abs(float(ip) - 1.0) < 0.05

    def test_invalid_num_draws(self):
        with pytest.raises(ValueError):
            MonteCarloMeasure(sampler=lambda n, d: torch.rand(n, 1), num_draws=0)


class TestMakeMeasure:
    def test_grid(self):
        m = make_measure("grid", quadrature_points=512)
        assert isinstance(m, UnitIntervalGridMeasure)

    def test_monte_carlo(self):
        m = make_measure("monte_carlo", quadrature_points=512)
        assert isinstance(m, MonteCarloMeasure)

    def test_weighted_discrete_raises(self):
        with pytest.raises(ValueError, match="explicit"):
            make_measure("weighted_discrete")

    def test_custom_measure_returned(self):
        pts = torch.tensor([[0.5]])
        wts = torch.tensor([1.0])
        custom = WeightedDiscreteMeasure(pts, wts)
        result = make_measure("grid", custom_measure=custom)
        assert result is custom
