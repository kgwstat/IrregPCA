from __future__ import annotations

import torch
from torch import nn

from .base import FunctionalModel

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}. Choose from {list(_ACTIVATIONS)}."
        ) from None


class DefaultModel(FunctionalModel):
    """Fully-connected MLP for representing scalar functions.

    Architecture: ``Linear(d, width) [→ act → Linear(width, width)] × depth
    → Linear(width, 1)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality *d* of the location space.
    width : int
        Hidden layer width (default 64).
    depth : int
        Number of hidden layers (default 2).
    activation : str
        Activation function name: ``"tanh"`` (default), ``"relu"``,
        ``"gelu"``, or ``"silu"``.
    output_dim : int
        Output dimensionality (default 1).
    """

    def __init__(
        self,
        input_dim: int = 1,
        width: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        output_dim: int = 1,
    ) -> None:
        super().__init__(input_dim=input_dim)
        layers: list[nn.Module] = [nn.Linear(input_dim, width), _get_activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), _get_activation(activation)]
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the MLP at input locations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, output_dim)``.
        """
        out: torch.Tensor = self.net(x)
        return out
