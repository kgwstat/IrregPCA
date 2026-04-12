from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
        """Invoke all registered callbacks with the event."""
        for cb in self._callbacks:
            cb(event.as_dict())

    def __bool__(self) -> bool:
        return bool(self._callbacks)
