from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class InnerProductMeasure(Protocol):
    """Protocol for measures used to compute inner products of functions.

    All methods produce or consume tensors on the specified device.

    Integration mode: exact (grid/weighted) or approximate (monte_carlo).
    """

    def integrate_product(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the inner product ⟨f, g⟩ = ∫ f(x) g(x) dμ(x).

        Parameters
        ----------
        f, g : callable
            Functions R^d → R, accepting ``(N, d)`` tensors and returning
            ``(N, 1)`` tensors.
        device : torch.device
            Device on which to create quadrature nodes.

        Returns
        -------
        torch.Tensor
            Scalar tensor.
        """
        ...


_DEFAULT_MAX_GRID_NODES: int = 10_000_000


class UnitIntervalGridMeasure:
    """Inner product on ``[0,1]^d`` using a uniform tensor-product quadrature grid.

    For ``d = 1`` this produces a standard equally-spaced grid.  For ``d > 1``
    it forms the Cartesian product of 1-D grids, yielding
    ``num_points ** d`` nodes in total.

    Integration mode: quadrature approximation.

    Parameters
    ----------
    num_points : int
        Number of equally-spaced nodes *per dimension* (default 4096).
    input_dim : int
        Dimensionality of the location space (default 1).
    max_nodes : int
        Safety limit on the total number of grid nodes.  Raises
        ``ValueError`` if ``num_points ** input_dim`` would exceed this.
    """

    def __init__(
        self,
        num_points: int = 4096,
        input_dim: int = 1,
        max_nodes: int = _DEFAULT_MAX_GRID_NODES,
    ) -> None:
        if num_points < 2:
            raise ValueError(f"`num_points` must be >= 2, got {num_points}.")
        total = num_points ** input_dim
        if total > max_nodes:
            raise ValueError(
                f"Tensor-product grid for d={input_dim} with {num_points} points per "
                f"dimension would have {total:,} nodes (limit {max_nodes:,}). "
                "Reduce `quadrature_points` or use integration_mode='monte_carlo'."
            )
        self.num_points = num_points
        self.input_dim = input_dim

    def integrate_product(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute ⟨f, g⟩ on a uniform tensor-product grid over ``[0,1]^d``.

        For ``d = 1``: ``(1/N) Σ f(x_i) g(x_i)`` on a uniform [0,1] grid.
        For ``d > 1``: Cartesian product of 1-D grids, uniform weights.

        Parameters
        ----------
        f, g : callable
            ``model(x)`` where ``x`` has shape ``(N, d)`` and output is ``(N, 1)``.
        device : torch.device

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        if self.input_dim == 1:
            grid = torch.linspace(0.0, 1.0, steps=self.num_points, device=device).unsqueeze(-1)
            vals_f = f(grid).squeeze(-1)
            vals_g = g(grid).squeeze(-1)
            return (vals_f * vals_g).mean()

        # Tensor-product grid for d > 1
        axes = [torch.linspace(0.0, 1.0, steps=self.num_points, device=device)] * self.input_dim
        mesh = torch.meshgrid(*axes, indexing="ij")  # each: (m, m, ..., m)
        pts = torch.stack([ax.flatten() for ax in mesh], dim=-1)  # (m**d, d)
        vals_f = f(pts).squeeze(-1)
        vals_g = g(pts).squeeze(-1)
        return (vals_f * vals_g).mean()


class WeightedDiscreteMeasure:
    """Inner product defined by a finite set of points and weights.

    Integration mode: exact weighted discrete approximation.

    Suitable for arbitrary ``d``.

    Parameters
    ----------
    points : torch.Tensor
        Shape ``(M, d)``, the quadrature / support points.
    weights : torch.Tensor
        Shape ``(M,)``, non-negative weights (need not sum to 1; they
        will be normalised internally).
    """

    def __init__(self, points: torch.Tensor, weights: torch.Tensor) -> None:
        if points.ndim != 2:
            raise ValueError(
                f"`points` must be 2-D with shape (M, d), got shape {tuple(points.shape)}."
            )
        if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
            raise ValueError(
                f"`weights` must be 1-D with length {points.shape[0]}, "
                f"got shape {tuple(weights.shape)}."
            )
        if (weights < 0).any():
            raise ValueError("`weights` must be non-negative.")
        self.points = points
        self.weights = weights / weights.sum()  # normalise

    def integrate_product(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute ⟨f, g⟩ = Σ_i w_i f(x_i) g(x_i).

        Parameters
        ----------
        f, g : callable
        device : torch.device

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        pts = self.points.to(device)
        wts = self.weights.to(device)
        vals_f = f(pts).squeeze(-1)  # (M,)
        vals_g = g(pts).squeeze(-1)  # (M,)
        return (wts * vals_f * vals_g).sum()


class MonteCarloMeasure:
    """Inner product approximated by Monte Carlo integration.

    Integration mode: stochastic unbiased estimator.

    Suitable for arbitrary ``d``.

    Parameters
    ----------
    sampler : callable
        ``sampler(n: int, device: torch.device) -> Tensor`` of shape ``(n, d)``.
    num_draws : int
        Number of Monte Carlo draws per integration call (default 4096).
    """

    def __init__(
        self,
        sampler: Callable[[int, torch.device], torch.Tensor],
        num_draws: int = 4096,
    ) -> None:
        if num_draws < 1:
            raise ValueError(f"`num_draws` must be >= 1, got {num_draws}.")
        self.sampler = sampler
        self.num_draws = num_draws

    def integrate_product(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute ⟨f, g⟩ ≈ (1/M) Σ f(x_i) g(x_i) where x_i ~ μ.

        This is an unbiased estimator of the true inner product.

        Parameters
        ----------
        f, g : callable
        device : torch.device

        Returns
        -------
        torch.Tensor
            Scalar (stochastic — varies across calls).
        """
        pts = self.sampler(self.num_draws, device)  # (M, d)
        vals_f = f(pts).squeeze(-1)
        vals_g = g(pts).squeeze(-1)
        return (vals_f * vals_g).mean()


def make_measure(
    integration_mode: str,
    quadrature_points: int = 4096,
    custom_measure: InnerProductMeasure | None = None,
    input_dim: int = 1,
) -> InnerProductMeasure:
    """Build a default measure from configuration.

    Parameters
    ----------
    integration_mode : str
        One of ``"grid"``, ``"monte_carlo"``, ``"weighted_discrete"``.
    quadrature_points : int
        Passed to :class:`UnitIntervalGridMeasure` as nodes *per dimension*.
    custom_measure : InnerProductMeasure or None
        If provided, returned directly.
    input_dim : int
        Dimensionality of the location space. Used to construct the
        tensor-product grid when ``integration_mode="grid"`` and ``d > 1``.

    Returns
    -------
    InnerProductMeasure

    Raises
    ------
    ValueError
        If ``integration_mode`` requires additional arguments that are not
        provided, or if the resulting grid would be too large.
    """
    if custom_measure is not None:
        return custom_measure
    if integration_mode == "grid":
        return UnitIntervalGridMeasure(num_points=quadrature_points, input_dim=input_dim)
    if integration_mode == "monte_carlo":
        # Default: uniform [0,1]^1 sampler
        def _uniform_sampler(n: int, device: torch.device) -> torch.Tensor:
            return torch.rand(n, 1, device=device)

        return MonteCarloMeasure(sampler=_uniform_sampler, num_draws=quadrature_points)
    if integration_mode == "weighted_discrete":
        raise ValueError(
            "integration_mode='weighted_discrete' requires an explicit "
            "`measure=WeightedDiscreteMeasure(points, weights)` to be passed."
        )
    raise ValueError(f"Unknown integration_mode: {integration_mode!r}.")
