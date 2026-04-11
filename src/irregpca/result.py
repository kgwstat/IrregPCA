from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LossHistory:
    """Container for training and validation loss history.

    Attributes
    ----------
    joint_train : list of float
        Joint training loss across all component models, recorded once per
        epoch per component during sequential training.
    joint_valid : list of float
        Joint validation loss, recorded at the same frequency as
        ``joint_train``.
    component_train : list of list of float
        Per-component training losses. ``component_train[0]`` holds the mean
        model's per-epoch losses; ``component_train[j]`` for ``j >= 1`` holds
        losses for the (j-1)-th component model.
    component_valid : list of list of float
        Per-component validation losses, mirroring ``component_train``.
    """

    joint_train: list[float] = field(default_factory=list)
    joint_valid: list[float] = field(default_factory=list)
    component_train: list[list[float]] = field(default_factory=list)
    component_valid: list[list[float]] = field(default_factory=list)


@dataclass
class IrregPCAResult:
    """Fitted result returned by :class:`IrregPCA` and :func:`fit_irreg_pca`.

    Attributes
    ----------
    mean_model : torch.nn.Module
        Fitted neural network representing the mean function.
    component_models : list of torch.nn.Module
        Fitted neural networks for each principal component function, in order.
        ``component_models[0]`` corresponds to the first component (0-based).
    history : LossHistory
        Loss history recorded during training.
    n_components : int
        Number of fitted component models.
    input_dim : int
        Dimensionality *d* of the location space.
    device : str
        Device on which the models were trained (e.g. ``"cpu"``, ``"cuda"``).
    """

    mean_model: torch.nn.Module
    component_models: list[torch.nn.Module]
    history: LossHistory
    n_components: int
    input_dim: int
    device: str

    def mean(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted mean function at the given locations.

        Parameters
        ----------
        locations : torch.Tensor
            Tensor of shape ``(N, d)`` giving the query locations. The tensor
            is automatically moved to the model's device before evaluation.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(N,)`` with the mean function values.
        """
        locs = locations.to(torch.device(self.device))
        self.mean_model.eval()
        with torch.no_grad():
            return self.mean_model(locs).squeeze(-1)

    def component(self, index: int, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate a single principal component function at the given locations.

        Parameters
        ----------
        index : int
            0-based index of the component to evaluate.
        locations : torch.Tensor
            Tensor of shape ``(N, d)`` giving the query locations. The tensor
            is automatically moved to the model's device before evaluation.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(N,)`` with the component function values.

        Raises
        ------
        ValueError
            If ``index`` is out of the range ``[0, n_components - 1]``.
        """
        if index < 0 or index >= self.n_components:
            raise ValueError(
                f"Component index {index} is out of range. "
                f"Valid indices are 0 to {self.n_components - 1}."
            )
        locs = locations.to(torch.device(self.device))
        model = self.component_models[index]
        model.eval()
        with torch.no_grad():
            return model(locs).squeeze(-1)

    def components(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate all component functions at the given locations.

        Parameters
        ----------
        locations : torch.Tensor
            Tensor of shape ``(N, d)`` giving the query locations.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(N, n_components)`` where column ``j`` contains
            the values of the ``j``-th component function (0-based).
        """
        locs = locations.to(torch.device(self.device))
        parts = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                parts.append(model(locs).squeeze(-1))
        return torch.stack(parts, dim=-1)

    def all_functions(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate the mean and all component functions at the given locations.

        Parameters
        ----------
        locations : torch.Tensor
            Tensor of shape ``(N, d)`` giving the query locations.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(N, 1 + n_components)`` where column 0 is the
            mean function and columns ``1..n_components`` are the component
            functions in order.
        """
        locs = locations.to(torch.device(self.device))
        self.mean_model.eval()
        with torch.no_grad():
            mean_vals = self.mean_model(locs).squeeze(-1)
        comp_vals = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                comp_vals.append(model(locs).squeeze(-1))
        return torch.stack([mean_vals] + comp_vals, dim=-1)
