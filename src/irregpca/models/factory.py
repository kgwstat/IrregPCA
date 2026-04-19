from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .mlp import DefaultModel
from .resnet import ResNetModel

# Maps model_kind strings to concrete subclass constructors.
_REGISTERED_KINDS: dict[str, type[DefaultModel] | type[ResNetModel]] = {  # noqa: UP007
    "mlp": DefaultModel,
    "resnet": ResNetModel,
}


def build_model(
    input_dim: int,
    output_dim: int = 1,
    model_kind: str = "mlp",
    width: int = 64,
    depth: int = 2,
    activation: str = "tanh",
    model_factory: Callable[..., nn.Module] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Create a model for representing a scalar function.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the location space.
    output_dim : int
        Output dimensionality (default 1).
    model_kind : str
        Built-in model type: ``"mlp"`` or ``"resnet"``.
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    activation : str
        Activation function name.
    model_factory : callable or None
        User-supplied factory ``factory(input_dim, output_dim, **kwargs) -> nn.Module``.
        Overrides ``model_kind`` when provided.
    model_kwargs : dict or None
        Extra keyword arguments forwarded to ``model_factory``.

    Returns
    -------
    torch.nn.Module
        An initialised model (not yet moved to a device).
    """
    kwargs = model_kwargs or {}
    if model_factory is not None:
        return model_factory(input_dim, output_dim, **kwargs)
    cls = _REGISTERED_KINDS.get(model_kind)
    if cls is None:
        raise ValueError(
            f"Unknown model_kind {model_kind!r}. Choose from {list(_REGISTERED_KINDS)}."
        )
    return cls(
        input_dim=input_dim, output_dim=output_dim, width=width, depth=depth, activation=activation
    )


def make_model(
    input_dim: int,
    factory: Callable[[int], nn.Module] | None = None,
) -> nn.Module:
    """Backward-compatible wrapper around :func:`build_model`.

    Parameters
    ----------
    input_dim : int
    factory : callable or None
        Legacy factory ``factory(input_dim) -> nn.Module``.

    Returns
    -------
    torch.nn.Module
    """
    if factory is not None:
        return factory(input_dim)
    return build_model(input_dim)
