from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FunctionalModel(nn.Module, ABC):
    """Abstract base class for models representing scalar functions R^d → R.

    Subclasses must implement :meth:`forward`.

    Parameters
    ----------
    input_dim : int
        Dimensionality *d* of the location space.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at the given locations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1)``.
        """
        ...
