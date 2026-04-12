from __future__ import annotations


def validate_hyperparams(
    n_components: int,
    epochs: int,
    lr: float,
    patience: int,
    valid_split: float,
    batch_size: int | None,
    num_workers: int,
    quadrature_points: int,
    validation_frequency: int,
    training_mode: str,
    integration_mode: str,
) -> None:
    """Validate IrregPCA hyperparameters, raising ValueError for invalid values.

    Parameters
    ----------
    n_components : int
    epochs : int
    lr : float
    patience : int
    valid_split : float
    batch_size : int or None
    num_workers : int
    quadrature_points : int
    validation_frequency : int
    training_mode : str
    integration_mode : str

    Raises
    ------
    ValueError
        If any hyperparameter is out of the valid range or has an invalid value.
    """
    if n_components < 1:
        raise ValueError(f"`n_components` must be >= 1, got {n_components}.")
    if epochs < 1:
        raise ValueError(f"`epochs` must be >= 1, got {epochs}.")
    if lr <= 0:
        raise ValueError(f"`lr` must be > 0, got {lr}.")
    if patience < 0:
        raise ValueError(f"`patience` must be >= 0, got {patience}.")
    if not (0.0 < valid_split < 1.0):
        raise ValueError(f"`valid_split` must be in (0, 1), got {valid_split}.")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"`num_workers` must be >= 0, got {num_workers}.")
    if quadrature_points < 2:
        raise ValueError(f"`quadrature_points` must be >= 2, got {quadrature_points}.")
    if validation_frequency < 1:
        raise ValueError(f"`validation_frequency` must be >= 1, got {validation_frequency}.")
    valid_training_modes = {"full_batch", "mini_batch", "streaming"}
    if training_mode not in valid_training_modes:
        raise ValueError(
            f"`training_mode` must be one of {valid_training_modes}, got {training_mode!r}."
        )
    valid_integration_modes = {"grid", "monte_carlo", "weighted_discrete"}
    if integration_mode not in valid_integration_modes:
        raise ValueError(
            f"`integration_mode` must be one of {valid_integration_modes}, "
            f"got {integration_mode!r}."
        )
