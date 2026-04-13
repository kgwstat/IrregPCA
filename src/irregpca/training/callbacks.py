from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingEvent:
    """Typed payload passed to callbacks at each event.

    Attributes
    ----------
    stage : str
        One of ``"train_start"``, ``"component_start"``, ``"epoch_end"``,
        ``"early_stop"``, ``"fit_end"``.
    component_index : int or None
        0-based component index, or ``None`` for the mean model.
    epoch : int or None
        Current epoch (0-based), or ``None`` for non-epoch events.
    train_loss : float or None
    valid_loss : float or None
    joint_train_loss : float or None
    joint_valid_loss : float or None
    is_mean : bool
        ``True`` if this event concerns the mean model.
    device : str
        Device string (e.g. ``"cpu"``).
    best_epoch : int or None
        Epoch with best validation loss so far (set at ``"early_stop"``
        and ``"fit_end"``).
    best_valid_loss : float or None
        Best validation loss seen so far.
    """

    stage: str
    component_index: int | None = None
    epoch: int | None = None
    train_loss: float | None = None
    valid_loss: float | None = None
    joint_train_loss: float | None = None
    joint_valid_loss: float | None = None
    is_mean: bool = False
    device: str = "cpu"
    best_epoch: int | None = None
    best_valid_loss: float | None = None

    def as_dict(self) -> dict:
        """Return a plain dict for backward-compatible callback payloads."""
        return {
            "stage": self.stage,
            "component_index": self.component_index,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "joint_train_loss": self.joint_train_loss,
            "joint_valid_loss": self.joint_valid_loss,
            "is_mean": self.is_mean,
            "device": self.device,
            "best_epoch": self.best_epoch,
            "best_valid_loss": self.best_valid_loss,
        }


class CallbackList:
    """Container for a list of callback functions.

    Each callback receives a :class:`TrainingEvent` instance. For backward
    compatibility, callbacks that accept a plain dict are also supported —
    the event is converted via :meth:`TrainingEvent.as_dict`.

    Parameters
    ----------
    callbacks : list of callable or None
    """

    def __init__(self, callbacks: list[Callable] | None = None) -> None:
        self._callbacks: list[Callable] = list(callbacks) if callbacks else []

    def fire(self, event: TrainingEvent) -> None:
        """Invoke all registered callbacks with the event.

        Skips class-based callbacks (those without ``__call__``); they are
        handled by :meth:`fire_lifecycle`.
        """
        for cb in self._callbacks:
            if callable(cb):
                cb(event.as_dict())

    def fire_lifecycle(self, method_name: str, **kwargs) -> None:
        """Call a named lifecycle method on class-based callbacks that have it."""
        for cb in self._callbacks:
            m = getattr(cb, method_name, None)
            if callable(m):
                m(**kwargs)

    def __bool__(self) -> bool:
        return bool(self._callbacks)


class LiveLossPlotCallback:
    """Display a live-updating loss curve during training.

    Shows one subplot per model (mean function + each principal component).
    Each subplot updates after every epoch; completed subplots are greyed out
    so progress across the full sequence is visible at a glance.

    Requires ``matplotlib``. Install via::

        pip install irregpca[viz]

    Parameters
    ----------
    update_every : int
        Redraw the plot every *update_every* epochs. Useful for large datasets
        where rendering on every epoch would slow training. Default: 1.
    figsize_per_panel : tuple[float, float]
        Width and height of each subplot panel in inches. Default: (3.5, 2.8).
    save_path : str or None
        If given, save the final figure to this path when training ends.
        The format is inferred from the file extension (e.g. ``"loss.png"``).
    """

    def __init__(
        self,
        update_every: int = 1,
        figsize_per_panel: tuple[float, float] = (3.5, 2.8),
        save_path: str | None = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            self._plt = plt
        except ImportError as exc:
            raise ImportError(
                "LiveLossPlotCallback requires matplotlib. "
                "Install it with:  pip install irregpca[viz]"
            ) from exc

        self.update_every = update_every
        self.figsize_per_panel = figsize_per_panel
        self.save_path = save_path

        self._fig: Any = None
        self._axes: list = []
        self._train_lines: list = []
        self._valid_lines: list = []
        self._best_vlines: list = []

        self._current_component: int = 0
        self._epoch_buffer_train: list[float] = []
        self._epoch_buffer_valid: list[float | None] = []

    # ── lifecycle hooks ───────────────────────────────────────────────────────

    def on_train_begin(self, n_components: int, **kwargs) -> None:
        """Called once before any component is trained."""
        plt = self._plt
        n_panels = 1 + n_components
        fig_w = self.figsize_per_panel[0] * n_panels
        fig_h = self.figsize_per_panel[1]

        plt.ion()
        self._fig, axes = plt.subplots(
            1, n_panels, figsize=(fig_w, fig_h), squeeze=False
        )
        self._axes = list(axes[0])

        for i, ax in enumerate(self._axes):
            label = "mean" if i == 0 else f"component {i}"
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("epoch", fontsize=8)
            ax.set_ylabel("loss", fontsize=8)
            ax.tick_params(labelsize=7)

            (train_line,) = ax.plot([], [], color="steelblue", lw=1.5, label="train")
            (valid_line,) = ax.plot(
                [], [], color="darkorange", lw=1.5, linestyle="--", label="valid"
            )
            best_vline = ax.axvline(
                x=0, color="crimson", lw=1.0, linestyle=":", visible=False
            )

            self._train_lines.append(train_line)
            self._valid_lines.append(valid_line)
            self._best_vlines.append(best_vline)
            ax.legend(fontsize=7, loc="upper right")

        self._fig.suptitle("IrregPCA — training loss", fontsize=10, y=1.01)
        self._fig.tight_layout()
        plt.pause(0.01)

    def on_component_begin(self, component_index: int, **kwargs) -> None:
        """Called before training each model (mean = index 0, PC k = index k)."""
        self._current_component = component_index
        self._epoch_buffer_train = []
        self._epoch_buffer_valid = []

    def on_epoch_end(self, metrics, **kwargs) -> None:
        """Called after each epoch; redraws the active subplot."""
        self._epoch_buffer_train.append(metrics.train_loss)
        self._epoch_buffer_valid.append(
            metrics.valid_loss if hasattr(metrics, "valid_loss") else None
        )

        epoch = len(self._epoch_buffer_train)
        if epoch % self.update_every != 0:
            return

        self._redraw_current(best_epoch=None)

    def on_component_end(self, best_epoch: int, **kwargs) -> None:
        """Called after a component finishes; marks best epoch, greys out panel."""
        self._redraw_current(best_epoch=best_epoch)

        ax = self._axes[self._current_component]
        ax.set_facecolor("#f5f5f5")
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")

        self._plt.pause(0.01)

    def on_train_end(self, **kwargs) -> None:
        """Called once after all components are trained."""
        plt = self._plt
        plt.ioff()

        if self.save_path is not None:
            self._fig.savefig(self.save_path, bbox_inches="tight", dpi=150)

        plt.show()

    # ── internal ──────────────────────────────────────────────────────────────

    def _redraw_current(self, best_epoch: int | None) -> None:
        idx = self._current_component
        train_line = self._train_lines[idx]
        valid_line = self._valid_lines[idx]
        best_vline = self._best_vlines[idx]
        ax = self._axes[idx]

        epochs = list(range(1, len(self._epoch_buffer_train) + 1))
        train_line.set_data(epochs, self._epoch_buffer_train)

        valid_data = [v for v in self._epoch_buffer_valid if v is not None]
        if valid_data:
            valid_line.set_data(epochs[: len(valid_data)], valid_data)
            valid_line.set_visible(True)
        else:
            valid_line.set_visible(False)

        if best_epoch is not None:
            best_vline.set_xdata([best_epoch, best_epoch])
            best_vline.set_visible(True)

        ax.relim()
        ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)
