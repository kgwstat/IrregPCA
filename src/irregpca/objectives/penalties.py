from __future__ import annotations

import torch

from .quadrature import InnerProductMeasure


def orthogonality_penalty(
    model: torch.nn.Module,
    prior_models: list[torch.nn.Module],
    measure: InnerProductMeasure,
    device: torch.device,
) -> torch.Tensor:
    """Compute the sum of squared inner products between model and prior models.

    Penalises ‖⟨e_j, e_i⟩‖² for i < j, pushing the j-th component to be
    orthogonal to all previously fitted components.

    Parameters
    ----------
    model : nn.Module
        The j-th component model being trained.
    prior_models : list of nn.Module
        Models e_1, ..., e_{j-1} (already fitted, no gradients needed).
    measure : InnerProductMeasure
        Integration measure.
    device : torch.device

    Returns
    -------
    torch.Tensor
        Scalar penalty (≥ 0).
    """
    if not prior_models:
        return torch.tensor(0.0, device=device)

    penalties = []
    for prior in prior_models:
        ip = measure.integrate_product(model, prior, device)
        penalties.append(ip.pow(2))

    return torch.stack(penalties).sum()


def norm_penalty(
    model: torch.nn.Module,
    measure: InnerProductMeasure,
    device: torch.device,
) -> torch.Tensor:
    """Compute 0.5 * ‖model‖² under the given measure.

    Used as the regularisation term for both mean and component models.

    Parameters
    ----------
    model : nn.Module
    measure : InnerProductMeasure
    device : torch.device

    Returns
    -------
    torch.Tensor
        Scalar ≥ 0.
    """
    ip = measure.integrate_product(model, model, device)
    return 0.5 * ip.pow(2)
