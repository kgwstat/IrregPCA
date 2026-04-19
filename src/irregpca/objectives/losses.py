from __future__ import annotations

from collections.abc import Callable

import torch

from .covariance import covariance_fn_packed
from .mean import mean_fn_packed
from .quadrature import InnerProductMeasure


def build_loss_fns(
    k: int,
    measure: InnerProductMeasure,
    device: torch.device,
) -> list[Callable[[list[torch.nn.Module], torch.Tensor], torch.Tensor]]:
    """Build a list of loss functions for sequential training.

    Returns a list of ``k+1`` callables:
    - Index 0: mean model loss.
    - Index j (j >= 1): j-th component model loss (0-based component j-1).

    Each callable has signature ``loss_fn(models, data) -> scalar Tensor``
    where ``data`` is a packed tensor of shape ``(N, d+2)``.

    The losses are:
    - Mean:        ``-⟨μ, Y⟩ + 0.5 * ⟨μ, μ⟩``
    - Component j: ``-cov_fn + 0.5 * ⟨e_j, e_j⟩² + Σ_{i<j} ⟨e_i, e_j⟩²``

    Parameters
    ----------
    k : int
        Number of components.
    measure : InnerProductMeasure
        Integration measure for inner products.
    device : torch.device

    Returns
    -------
    list of callable
        Length ``k+1``.
    """
    lossfns: list[Callable] = [None] * (k + 1)  # type: ignore[list-item]

    def _mean_loss(
        models: list[torch.nn.Module],
        data: torch.Tensor,
    ) -> torch.Tensor:
        norm_sq = measure.integrate_product(models[0], models[0], device)
        return -mean_fn_packed(models[0], data) + 0.5 * norm_sq

    lossfns[0] = _mean_loss

    for j in range(1, k + 1):

        def _make_component_loss(j: int) -> Callable:
            def _component_loss(
                models: list[torch.nn.Module],
                data: torch.Tensor,
            ) -> torch.Tensor:
                cov = covariance_fn_packed(models[j], data)
                norm_sq = measure.integrate_product(models[j], models[j], device)
                orth = torch.stack([
                    measure.integrate_product(models[j], prior, device).pow(2)
                    for prior in models[1:j]
                ]).sum() if j > 1 else torch.tensor(0.0, device=device)
                return -cov + 0.5 * norm_sq.pow(2) + orth

            return _component_loss

        lossfns[j] = _make_component_loss(j)

    return lossfns
