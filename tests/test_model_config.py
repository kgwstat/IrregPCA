"""Tests for model configurability: config fields, factory, MLP, ResNet."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from irregpca import IrregPCAConfig, fit_irreg_pca
from irregpca.models.factory import build_model
from irregpca.models.mlp import DefaultModel
from irregpca.models.resnet import ResNetModel

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_dataset():
    """Minimal dataset: 20 samples, 8 obs each, 1-D locations."""
    torch.manual_seed(7)
    n_samples, obs_per = 20, 8
    n_obs = n_samples * obs_per
    sample_ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locations = torch.rand(n_obs, 1)
    values = torch.sin(locations.squeeze()) + 0.01 * torch.randn(n_obs)
    return sample_ids, locations, values


def _quick_fit(dataset, **kwargs):
    sample_ids, locations, values = dataset
    return fit_irreg_pca(
        sample_ids=sample_ids,
        locations=locations,
        values=values,
        n_components=1,
        epochs=3,
        patience=100,
        random_state=0,
        **kwargs,
    )


# ── IrregPCAConfig field acceptance ──────────────────────────────────────────

def test_config_accepts_hidden_width_depth():
    cfg = IrregPCAConfig(n_components=2, hidden_width=32, hidden_depth=3)
    assert cfg.hidden_width == 32
    assert cfg.hidden_depth == 3


def test_config_accepts_model_kind_resnet():
    cfg = IrregPCAConfig(n_components=2, model_kind="resnet")
    assert cfg.model_kind == "resnet"


def test_config_accepts_model_factory():
    factory = lambda i, o: nn.Linear(i, o)  # noqa: E731
    cfg = IrregPCAConfig(n_components=2, model_factory=factory)
    assert cfg.model_factory is factory


def test_config_model_kwargs_defaults_none():
    cfg = IrregPCAConfig(n_components=1)
    assert cfg.model_kwargs is None


# ── Validation errors ─────────────────────────────────────────────────────────

def test_config_invalid_model_kind_raises():
    with pytest.raises(ValueError, match="model_kind"):
        IrregPCAConfig(n_components=1, model_kind="transformer")


def test_config_invalid_hidden_width_raises():
    with pytest.raises(ValueError, match="hidden_width"):
        IrregPCAConfig(n_components=1, hidden_width=0)


def test_config_invalid_hidden_depth_raises():
    with pytest.raises(ValueError, match="hidden_depth"):
        IrregPCAConfig(n_components=1, hidden_depth=0)


def test_config_custom_factory_bypasses_model_kind_validation():
    factory = lambda i, o: nn.Linear(i, o)  # noqa: E731
    # model_kind="unknown" is OK when model_factory is provided
    cfg = IrregPCAConfig(n_components=1, model_factory=factory, model_kind="unknown_kind")
    assert cfg.model_factory is factory


# ── build_model ───────────────────────────────────────────────────────────────

def test_build_model_mlp_default():
    m = build_model(input_dim=2)
    assert isinstance(m, DefaultModel)


def test_build_model_resnet():
    m = build_model(input_dim=2, model_kind="resnet", width=32, depth=4)
    assert isinstance(m, ResNetModel)


def test_build_model_custom_factory():
    class MyModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.fc(x)

    m = build_model(input_dim=3, model_factory=lambda i, o: MyModel(i, o))
    assert isinstance(m, MyModel)


def test_build_model_invalid_kind_raises():
    with pytest.raises(ValueError, match="model_kind"):
        build_model(input_dim=1, model_kind="unknown")


def test_build_model_mlp_configurable_depth():
    m = build_model(input_dim=1, model_kind="mlp", width=16, depth=4)
    # A depth-4 MLP should have 4 hidden layers + 1 output = 5 linear layers
    linear_layers = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    # input→h1, h1→h2, h2→h3, h3→h4, h4→out  = 5
    assert len(linear_layers) == 5


# ── fit_irreg_pca model args ──────────────────────────────────────────────────

def test_functional_api_forwards_hidden_width_depth(tiny_dataset):
    result = _quick_fit(tiny_dataset, hidden_width=16, hidden_depth=3)
    assert result is not None


def test_functional_api_mlp_kind(tiny_dataset):
    result = _quick_fit(tiny_dataset, model_kind="mlp", hidden_width=16)
    assert result is not None


def test_functional_api_resnet_kind(tiny_dataset):
    result = _quick_fit(tiny_dataset, model_kind="resnet", hidden_width=16, hidden_depth=2)
    assert result is not None


def test_functional_api_custom_factory(tiny_dataset):
    class Tiny(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.fc(x)

    factory_calls: list[int] = []

    def factory(input_dim, output_dim):
        factory_calls.append(input_dim)
        return Tiny(input_dim, output_dim)

    result = _quick_fit(tiny_dataset, model_factory=factory)
    assert result is not None
    assert len(factory_calls) > 0  # factory was called


def test_functional_api_invalid_model_kind_raises(tiny_dataset):
    with pytest.raises(ValueError):
        _quick_fit(tiny_dataset, model_kind="bad_kind")


# ── DefaultModel forward shape ────────────────────────────────────────────────

def test_mlp_forward_shape():
    m = DefaultModel(input_dim=2, width=8, depth=3)
    x = torch.rand(10, 2)
    out = m(x)
    assert out.shape == (10, 1)


def test_resnet_forward_shape():
    m = ResNetModel(input_dim=2, width=8, depth=4)
    x = torch.rand(10, 2)
    out = m(x)
    assert out.shape == (10, 1)
