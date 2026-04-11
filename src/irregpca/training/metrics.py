from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EpochMetrics:
    """Metrics recorded at the end of a single training epoch.

    Attributes
    ----------
    epoch : int
    train_loss : float
    valid_loss : float
    wall_time_s : float
        Wall-clock seconds for the epoch.
    """

    epoch: int
    train_loss: float
    valid_loss: float
    wall_time_s: float = 0.0


@dataclass
class ComponentMetrics:
    """Metrics accumulated across all epochs for a single component.

    Attributes
    ----------
    epochs : list of EpochMetrics
    best_epoch : int
    best_valid_loss : float
    """

    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_valid_loss: float = float("inf")
