from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .utils.validation import validate_hyperparams


@dataclass
class IrregPCAConfig:
    """Configuration for :class:`IrregPCA`.

    All fields that control training, data loading, and integration are
    consolidated here so the estimator signature stays stable.

    Parameters
    ----------
    n_components : int
        Number of principal component functions to fit.
    epochs : int
        Maximum training epochs per component (default 600).
    lr : float
        Adam optimiser learning rate (default 1e-3).
    patience : int
        Early-stopping patience in epochs (default 300). Set to 0 to
        disable early stopping.
    valid_split : float
        Fraction of *samples* (not observations) to hold out for
        validation. Must be in ``(0, 1)`` (default 0.2).
    random_state : int or None
        Seed for the train/validation split and weight initialisation.
        ``None`` means no fixed seed.
    device : str, torch.device, or None
        Device to run training on. ``None`` auto-selects
        ``cuda > mps > cpu``.
    batch_size : int or None
        Number of *samples* (curves) per mini-batch. ``None`` uses the
        full training set (full-batch mode). Only relevant for
        ``training_mode="mini_batch"`` or ``"streaming"``.
    num_workers : int
        Number of DataLoader worker processes (default 0).
    training_mode : str
        One of ``"full_batch"``, ``"mini_batch"``, or ``"streaming"``
        (default ``"full_batch"``).
    integration_mode : str
        One of ``"grid"``, ``"monte_carlo"``, or ``"weighted_discrete"``
        (default ``"grid"``).
    quadrature_points : int
        Number of quadrature nodes used by grid integration
        (default 4096).
    validation_frequency : int
        Evaluate validation loss every this many epochs (default 1).
    measure : Any or None
        Custom :class:`InnerProductMeasure` instance. If provided,
        overrides ``integration_mode``.
    verbose : bool
        Print training progress (default ``False``).

    Examples
    --------
    ::

        cfg = IrregPCAConfig(n_components=3, epochs=200, lr=5e-4)
        est = IrregPCA(cfg)
    """

    n_components: int
    epochs: int = 600
    lr: float = 1e-3
    patience: int = 300
    valid_split: float = 0.2
    random_state: int | None = None
    device: str | torch.device | None = None
    batch_size: int | None = None
    num_workers: int = 0
    training_mode: str = "full_batch"
    integration_mode: str = "grid"
    quadrature_points: int = 4096
    validation_frequency: int = 1
    measure: Any | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        validate_hyperparams(
            n_components=self.n_components,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            valid_split=self.valid_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            quadrature_points=self.quadrature_points,
            validation_frequency=self.validation_frequency,
            training_mode=self.training_mode,
            integration_mode=self.integration_mode,
        )
