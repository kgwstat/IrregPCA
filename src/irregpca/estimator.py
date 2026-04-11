from __future__ import annotations

from typing import Callable

import torch

from .core import _parse_input, _fit_irreg_pca_core
from .result import IrregPCAResult


class IrregPCA:
    """Estimator for irregular functional PCA.

    Fits a mean function and a set of principal component functions to
    irregularly observed functional data using neural network models.

    Parameters
    ----------
    n_components : int
        Number of principal component functions to fit.
    epochs : int, optional
        Maximum number of training epochs per component (default 600).
    lr : float, optional
        Adam optimiser learning rate (default 1e-3).
    patience : int, optional
        Early-stopping patience in epochs (default 300).
    valid_split : float, optional
        Fraction of samples to hold out for validation (default 0.2).
    random_state : int or None, optional
        Seed for the train/validation split RNG. ``None`` means no fixed seed.
    device : str, torch.device, or None, optional
        Device to run training on. If ``None``, auto-selects CUDA > MPS > CPU.
    model_factory : callable or None, optional
        A callable ``model_factory(input_dim: int) -> torch.nn.Module`` used
        to create each fitted model. If ``None``, uses the built-in
        ``DefaultModel``.
    verbose : bool, optional
        If ``True``, print training progress. Default ``False``.
    callbacks : list of callable or None, optional
        List of callback functions invoked at the end of each training epoch.
        Each callback receives a single dictionary with keys:
        ``stage``, ``component_index``, ``epoch``, ``train_loss``,
        ``valid_loss``, ``joint_train_loss``, ``joint_valid_loss``,
        ``is_mean``, ``device``.

    Examples
    --------
    Split-input usage::

        est = IrregPCA(n_components=3, epochs=600)
        result = est.fit(sample_ids=idx, locations=U, values=Y)
        mu = result.mean(grid)

    Packed-input usage::

        est = IrregPCA(n_components=3)
        result = est.fit(data=data)
    """

    def __init__(
        self,
        n_components: int,
        *,
        epochs: int = 600,
        lr: float = 1e-3,
        patience: int = 300,
        valid_split: float = 0.2,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        model_factory: Callable[[int], torch.nn.Module] | None = None,
        verbose: bool = False,
        callbacks: list[Callable[[dict], None]] | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError(
                f"`n_components` must be at least 1, got {n_components}."
            )
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.valid_split = valid_split
        self.random_state = random_state
        self.device = device
        self.model_factory = model_factory
        self.verbose = verbose
        self.callbacks = callbacks

        # fitted attributes (set after .fit())
        self.mean_model_: torch.nn.Module | None = None
        self.component_models_: list[torch.nn.Module] | None = None
        self.history_ = None
        self.result_: IrregPCAResult | None = None
        self.input_dim_: int | None = None
        self.device_: str | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        data: torch.Tensor | None = None,
        *,
        sample_ids: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> IrregPCAResult:
        """Fit the model to irregularly observed functional data.

        Accepts either packed or split input format.

        **Packed input** — ``data`` of shape ``(N, d+2)``::

            result = est.fit(data=data)

        Columns: ``[sample_id, loc_1, ..., loc_d, value]``.

        **Split input** — three separate tensors::

            result = est.fit(sample_ids=idx, locations=U, values=Y)

        Parameters
        ----------
        data : torch.Tensor, optional
            Packed tensor of shape ``(N, d+2)``.
        sample_ids : torch.Tensor, optional
            1-D tensor of shape ``(N,)`` with integer sample identifiers.
        locations : torch.Tensor, optional
            2-D tensor of shape ``(N, d)`` with observation locations.
        values : torch.Tensor, optional
            1-D tensor of shape ``(N,)`` with observed scalar values.

        Returns
        -------
        IrregPCAResult
            The fitted result object, also stored as ``self.result_``.
        """
        sample_ids_t, locations_t, values_t = _parse_input(
            data,
            sample_ids=sample_ids,
            locations=locations,
            values=values,
        )

        result = _fit_irreg_pca_core(
            sample_ids=sample_ids_t,
            locations=locations_t,
            values=values_t,
            n_components=self.n_components,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            valid_split=self.valid_split,
            random_state=self.random_state,
            device=self.device,
            model_factory=self.model_factory,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        self.result_ = result
        self.mean_model_ = result.mean_model
        self.component_models_ = result.component_models
        self.history_ = result.history
        self.input_dim_ = result.input_dim
        self.device_ = result.device
        self.is_fitted_ = True

        return result

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This IrregPCA instance is not fitted yet. "
                "Call `.fit(...)` before using inference methods."
            )

    def mean(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted mean function.

        Parameters
        ----------
        locations : torch.Tensor
            Tensor of shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(N,)``.
        """
        self._check_fitted()
        return self.result_.mean(locations)

    def component(self, index: int, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate a single principal component function (0-based index).

        Parameters
        ----------
        index : int
            0-based component index.
        locations : torch.Tensor
            Tensor of shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(N,)``.
        """
        self._check_fitted()
        return self.result_.component(index, locations)

    def components(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate all component functions, stacked along the last dimension.

        Parameters
        ----------
        locations : torch.Tensor
            Tensor of shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(N, n_components)``.
        """
        self._check_fitted()
        return self.result_.components(locations)


def fit_irreg_pca(
    data: torch.Tensor | None = None,
    *,
    sample_ids: torch.Tensor | None = None,
    locations: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    n_components: int,
    epochs: int = 600,
    lr: float = 1e-3,
    patience: int = 300,
    valid_split: float = 0.2,
    random_state: int | None = None,
    device: str | torch.device | None = None,
    model_factory: Callable[[int], torch.nn.Module] | None = None,
    verbose: bool = False,
    callbacks: list[Callable[[dict], None]] | None = None,
) -> IrregPCAResult:
    """Fit IrregPCA and return the result in a single call.

    Functional convenience wrapper around :class:`IrregPCA`. Constructs an
    estimator with the given hyperparameters, calls :meth:`~IrregPCA.fit`,
    and returns the :class:`IrregPCAResult`.

    Parameters
    ----------
    data : torch.Tensor, optional
        Packed tensor of shape ``(N, d+2)``. Columns:
        ``[sample_id, loc_1, ..., loc_d, value]``.
    sample_ids : torch.Tensor, optional
        1-D tensor of shape ``(N,)`` with integer sample identifiers.
    locations : torch.Tensor, optional
        2-D tensor of shape ``(N, d)`` with observation locations.
    values : torch.Tensor, optional
        1-D tensor of shape ``(N,)`` with observed scalar values.
    n_components : int
        Number of principal component functions to fit.
    epochs : int, optional
        Maximum training epochs per component (default 600).
    lr : float, optional
        Adam learning rate (default 1e-3).
    patience : int, optional
        Early-stopping patience in epochs (default 300).
    valid_split : float, optional
        Fraction of samples used for validation (default 0.2).
    random_state : int or None, optional
        RNG seed for the train/validation split.
    device : str, torch.device, or None, optional
        Target device. ``None`` auto-selects CUDA > MPS > CPU.
    model_factory : callable or None, optional
        Factory ``model_factory(input_dim) -> torch.nn.Module``. If ``None``,
        uses ``DefaultModel``.
    verbose : bool, optional
        Print training progress (default ``False``).
    callbacks : list of callable or None, optional
        Epoch-end callbacks; each receives an info dictionary.

    Returns
    -------
    IrregPCAResult
        Fitted result with mean model, component models, and loss history.

    Examples
    --------
    ::

        result = fit_irreg_pca(sample_ids=idx, locations=U, values=Y, n_components=3)
        mu = result.mean(grid)
        phi1 = result.component(0, grid)
    """
    est = IrregPCA(
        n_components=n_components,
        epochs=epochs,
        lr=lr,
        patience=patience,
        valid_split=valid_split,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        verbose=verbose,
        callbacks=callbacks,
    )
    return est.fit(
        data,
        sample_ids=sample_ids,
        locations=locations,
        values=values,
    )
