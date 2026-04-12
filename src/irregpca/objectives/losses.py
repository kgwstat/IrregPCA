from __future__ import annotations

from collections.abc import Callable

import torch

from .covariance import covariance_fn_packed
from .mean import mean_fn_packed
from .penalties import norm_penalty, orthogonality_penalty
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
    - Mean: ``-mean_fn + 0.5 * ‖μ‖²`` (exact full-batch, quadrature approx)
    - Component j: ``-cov_fn + 0.5 * ‖e_j‖⁴ + Σ_{i<j} ⟨e_i, e_j⟩²``
      (exact full-batch objective, quadrature inner products)

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

    # mean loss: -mean_fn + 0.5 * ||mu||^2
    def _mean_loss(
        models: list[torch.nn.Module],
        data: torch.Tensor,
    ) -> torch.Tensor:
        return -mean_fn_packed(models[0], data) + norm_penalty(models[0], measure, device)

    lossfns[0] = _mean_loss

    # component losses (closure over j)
    for j in range(1, k + 1):

        def _make_component_loss(j: int) -> Callable:
            def _component_loss(
                models: list[torch.nn.Module],
                data: torch.Tensor,
            ) -> torch.Tensor:
                cov = covariance_fn_packed(models[j], data)
                np_ = norm_penalty(models[j], measure, device)
                orth = orthogonality_penalty(
                    models[j],
                    prior_models=models[1:j],
                    measure=measure,
                    device=device,
                )
                return -cov + np_ + orth

            return _component_loss

        lossfns[j] = _make_component_loss(j)

    return lossfns
