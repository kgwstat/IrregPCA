from __future__ import annotations

from collections.abc import Callable

import torch

from .mlp import DefaultModel


def make_model(
    input_dim: int,
    factory: Callable[[int], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """Create a model for representing a scalar function.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the location space.
    factory : callable or None
        Optional user-supplied factory ``factory(input_dim) -> nn.Module``.
        If ``None``, returns a :class:`DefaultModel`.

    Returns
    -------
    torch.nn.Module
        An initialised model (not yet moved to a device).
    """
    if factory is not None:
        return factory(input_dim)
    return DefaultModel(input_dim=input_dim)
