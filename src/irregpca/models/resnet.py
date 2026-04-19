from __future__ import annotations

import torch
from torch import nn

from .base import FunctionalModel
from .mlp import _get_activation


class _ResBlock(nn.Module):
    """Two-layer residual block: Linear → act → Linear + skip."""

    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layer2(self.act(self.layer1(x)))
        return out + x


class ResNetModel(FunctionalModel):
    """Residual MLP for representing scalar functions.

    Architecture: ``Linear(d, width) → act → [ResBlock] × (depth // 2)
    → Linear(width, output_dim)``, where each ResBlock is
    ``Linear → act → Linear + skip``.

    Parameters
    ----------
    input_dim : int
        Dimensionality *d* of the location space.
    width : int
        Hidden layer width (default 64).
    depth : int
        Total number of hidden layers (rounded to nearest even; default 2).
    activation : str
        Activation function name: ``"tanh"`` (default), ``"relu"``, etc.
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
        n_blocks = max(1, depth // 2)
        self.stem = nn.Sequential(nn.Linear(input_dim, width), _get_activation(activation))
        self.blocks = nn.Sequential(*[_ResBlock(width, activation) for _ in range(n_blocks)])
        self.head = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the ResNet at input locations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, output_dim)``.
        """
        out: torch.Tensor = self.head(self.blocks(self.stem(x)))
        return out
