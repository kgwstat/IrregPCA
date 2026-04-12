"""Tests for LiveLossPlotCallback."""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import

from irregpca.training.callbacks import LiveLossPlotCallback


def _make_metrics(train_loss: float, valid_loss: float | None = None):
    """Minimal stand-in for EpochMetrics."""
    class M:
        pass
    m = M()
    m.train_loss = train_loss
    m.valid_loss = valid_loss
    return m


def _run_fake_training(callback, n_components: int = 2, epochs: int = 5):
    """Drive the callback through a complete fake training run."""
    callback.on_train_begin(n_components=n_components)
    for i in range(1 + n_components):  # mean + components
        callback.on_component_begin(component_index=i)
        for e in range(epochs):
            callback.on_epoch_end(metrics=_make_metrics(1.0 / (e + 1), 1.2 / (e + 1)))
        callback.on_component_end(best_epoch=epochs - 1)
    callback.on_train_end()


def test_callback_runs_without_error():
    cb = LiveLossPlotCallback()
    _run_fake_training(cb)


def test_figure_has_correct_number_of_panels():
    n_components = 3
    cb = LiveLossPlotCallback()
    cb.on_train_begin(n_components=n_components)
    assert len(cb._axes) == 1 + n_components


def test_update_every_is_respected():
    cb = LiveLossPlotCallback(update_every=3)
    cb.on_train_begin(n_components=1)
    cb.on_component_begin(component_index=0)
    for _e in range(6):
        cb.on_epoch_end(metrics=_make_metrics(1.0))
    # buffer should have 6 entries regardless of update_every
    assert len(cb._epoch_buffer_train) == 6


def test_save_path_writes_file(tmp_path):
    out = tmp_path / "loss.png"
    cb = LiveLossPlotCallback(save_path=str(out))
    _run_fake_training(cb, n_components=1, epochs=3)
    assert out.exists()


def test_no_valid_loss_hides_valid_line():
    cb = LiveLossPlotCallback()
    cb.on_train_begin(n_components=1)
    cb.on_component_begin(component_index=0)
    cb.on_epoch_end(metrics=_make_metrics(1.0, valid_loss=None))
    cb._redraw_current(best_epoch=None)
    assert not cb._valid_lines[0].get_visible()


def test_completed_panel_gets_grey_background():
    cb = LiveLossPlotCallback()
    cb.on_train_begin(n_components=1)
    cb.on_component_begin(component_index=0)
    cb.on_epoch_end(metrics=_make_metrics(1.0))
    cb.on_component_end(best_epoch=0)
    color = cb._axes[0].get_facecolor()
    # #f5f5f5 in RGBA is approximately (0.96, 0.96, 0.96, 1.0)
    assert color[0] > 0.9 and color[1] > 0.9 and color[2] > 0.9


def test_missing_matplotlib_raises_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pip install irregpca"):
        LiveLossPlotCallback()
