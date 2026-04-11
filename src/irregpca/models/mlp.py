from __future__ import annotations

import torch
from torch import nn

from .base import FunctionalModel


class DefaultModel(FunctionalModel):
    """Default fully-connected MLP for representing scalar functions.

    Architecture: ``Linear(d, width) → tanh → Linear(width, width) → tanh
    → Linear(width, 1)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality *d* of the location space.
    width : int
        Hidden layer width (default 20).
    """

    def __init__(self, input_dim: int = 1, width: int = 20) -> None:
        super().__init__(input_dim=input_dim)
        self.layer1 = nn.Linear(input_dim, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the MLP at input locations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1)``.
        """
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)
