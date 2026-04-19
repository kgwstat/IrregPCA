"""Tests for tensor-product quadrature for d > 1."""
from __future__ import annotations

import pytest
import torch

from irregpca.objectives.quadrature import UnitIntervalGridMeasure, make_measure

CPU = torch.device("cpu")


# ── d=1 regression ──────────────────────────────────────────────────────────

def test_d1_grid_shape():
    m = UnitIntervalGridMeasure(num_points=10, input_dim=1)
    nodes: list[torch.Tensor] = []

    def capture_f(x):
        nodes.append(x)
        return torch.ones(x.shape[0], 1)

    m.integrate_product(capture_f, lambda x: torch.ones(x.shape[0], 1), CPU)
    assert nodes[0].shape == (10, 1)


def test_d1_constant_one_integrates_to_one():
    m = UnitIntervalGridMeasure(num_points=1000, input_dim=1)
    f = lambda x: torch.ones(x.shape[0], 1)  # noqa: E731
    result = m.integrate_product(f, f, CPU)
    assert result.item() == pytest.approx(1.0, abs=1e-6)


# ── d=2 tensor-product grid ──────────────────────────────────────────────────

def test_d2_grid_node_count():
    """integrate_product for d=2 should evaluate f on (m**2, 2) points."""
    m_pts = 5
    m = UnitIntervalGridMeasure(num_points=m_pts, input_dim=2)
    nodes: list[torch.Tensor] = []

    def capture_f(x):
        nodes.append(x)
        return torch.ones(x.shape[0], 1)

    m.integrate_product(capture_f, lambda x: torch.ones(x.shape[0], 1), CPU)
    assert nodes[0].shape == (m_pts ** 2, 2)


def test_d2_constant_one_integrates_to_one():
    m = UnitIntervalGridMeasure(num_points=100, input_dim=2)
    f = lambda x: torch.ones(x.shape[0], 1)  # noqa: E731
    result = m.integrate_product(f, f, CPU)
    assert result.item() == pytest.approx(1.0, abs=1e-6)


def test_d2_grid_coordinates_in_unit_square():
    """All grid nodes should lie in [0, 1]^2."""
    nodes: list[torch.Tensor] = []
    m = UnitIntervalGridMeasure(num_points=10, input_dim=2)

    def capture_f(x):
        nodes.append(x)
        return torch.ones(x.shape[0], 1)

    m.integrate_product(capture_f, lambda x: torch.ones(x.shape[0], 1), CPU)
    pts = nodes[0]
    assert pts.min().item() >= 0.0
    assert pts.max().item() <= 1.0


# ── size guard ────────────────────────────────────────────────────────────────

def test_grid_size_guard_raises():
    with pytest.raises(ValueError, match="nodes"):
        UnitIntervalGridMeasure(num_points=1000, input_dim=3, max_nodes=100)


def test_grid_size_guard_exact_limit_ok():
    # num_points ** input_dim == max_nodes should succeed
    UnitIntervalGridMeasure(num_points=10, input_dim=2, max_nodes=100)


def test_make_measure_grid_d2():
    measure = make_measure("grid", quadrature_points=8, input_dim=2)
    assert isinstance(measure, UnitIntervalGridMeasure)
    assert measure.input_dim == 2
    assert measure.num_points == 8


def test_make_measure_grid_d1_default():
    measure = make_measure("grid", quadrature_points=16)
    assert isinstance(measure, UnitIntervalGridMeasure)
    assert measure.input_dim == 1
