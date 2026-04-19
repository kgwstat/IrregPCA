"""Tests for plotting utilities and device safety."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from irregpca.utils.plotting import to_numpy_cpu

# ── to_numpy_cpu ─────────────────────────────────────────────────────────────

def test_to_numpy_cpu_from_cpu_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_numpy_cpu(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])


def test_to_numpy_cpu_from_numpy_array():
    a = np.array([4.0, 5.0])
    arr = to_numpy_cpu(a)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, a)


def test_to_numpy_cpu_from_scalar():
    arr = to_numpy_cpu(3.14)
    assert isinstance(arr, np.ndarray)
    assert float(arr) == pytest.approx(3.14)


def test_to_numpy_cpu_from_list():
    arr = to_numpy_cpu([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]


def test_to_numpy_cpu_detaches_gradient():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    arr = to_numpy_cpu(t)
    assert isinstance(arr, np.ndarray)


# ── LiveLossPlotCallback: no plt.show() in Jupyter ────────────────────────────

def test_on_train_end_no_show_in_jupyter():
    pytest.importorskip("matplotlib")
    from irregpca.training.callbacks import LiveLossPlotCallback

    cb = LiveLossPlotCallback()
    mock_plt = MagicMock()
    cb._plt = mock_plt
    cb._fig = MagicMock()
    cb._jupyter_handle = MagicMock()  # simulate Jupyter environment

    cb.on_train_end()

    mock_plt.show.assert_not_called()


def test_on_train_end_calls_show_outside_jupyter():
    pytest.importorskip("matplotlib")
    from irregpca.training.callbacks import LiveLossPlotCallback

    cb = LiveLossPlotCallback()
    mock_plt = MagicMock()
    cb._plt = mock_plt
    cb._fig = MagicMock()
    cb._jupyter_handle = None  # non-Jupyter environment

    cb.on_train_end()

    mock_plt.show.assert_called_once()


# ── Default training does not open figures ────────────────────────────────────

def test_default_training_no_figures(tmp_path):
    """fit_irreg_pca with no callbacks must not open any matplotlib figures."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from irregpca import fit_irreg_pca

    torch.manual_seed(0)
    n_samples, obs_per = 10, 5
    n_obs = n_samples * obs_per
    sample_ids = torch.repeat_interleave(torch.arange(n_samples, dtype=torch.float), obs_per)
    locations = torch.rand(n_obs, 1)
    values = torch.rand(n_obs)

    before = plt.get_fignums()
    fit_irreg_pca(
        sample_ids=sample_ids,
        locations=locations,
        values=values,
        n_components=1,
        epochs=2,
        patience=10,
        random_state=0,
    )
    after = plt.get_fignums()
    assert set(after) == set(before), "Default training must not open matplotlib figures."
