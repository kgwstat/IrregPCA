from __future__ import annotations

from typing import Callable

import torch
from torch import optim

from .model import DefaultModel, train_only
from .estimate import mean_fn, covariance_fn, inner_product
from .utils import split_data
from .result import IrregPCAResult, LossHistory


def _parse_input(
    data: torch.Tensor | None = None,
    *,
    sample_ids: torch.Tensor | None = None,
    locations: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize packed or split input into ``(sample_ids, locations, values)``.

    Parameters
    ----------
    data : torch.Tensor, optional
        Packed tensor of shape ``(N, d+2)``. Columns are
        ``[sample_id, loc_1, ..., loc_d, value]``.
    sample_ids : torch.Tensor, optional
        1-D tensor of shape ``(N,)`` with integer sample identifiers.
    locations : torch.Tensor, optional
        2-D tensor of shape ``(N, d)`` with observation locations.
    values : torch.Tensor, optional
        1-D tensor of shape ``(N,)`` with observed scalar values.

    Returns
    -------
    tuple of torch.Tensor
        Normalized ``(sample_ids, locations, values)`` tensors.

    Raises
    ------
    ValueError
        On invalid shape or missing arguments.
    """
    if data is not None:
        if data.ndim != 2:
            raise ValueError(
                f"Expected `data` to be a 2-D tensor, but got {data.ndim}-D. "
                "Expected shape is (N, d+2), with columns [sample_id, loc..., value]."
            )
        if data.shape[1] < 3:
            raise ValueError(
                f"Expected `data` to have at least 3 columns (got {data.shape[1]}). "
                "Expected shape is (N, d+2), with columns [sample_id, loc..., value]."
            )
        sample_ids = data[:, 0]
        locations = data[:, 1:-1]
        values = data[:, -1]
    else:
        if sample_ids is None or locations is None or values is None:
            raise ValueError(
                "Provide either `data` (packed mode) or all three of "
                "`sample_ids`, `locations`, and `values` (split mode)."
            )
        if sample_ids.ndim != 1:
            raise ValueError(
                f"Expected `sample_ids` to be 1-D, got shape {tuple(sample_ids.shape)}."
            )
        if locations.ndim != 2:
            raise ValueError(
                f"Expected `locations` to be 2-D with shape (N, d), "
                f"got shape {tuple(locations.shape)}."
            )
        if values.ndim != 1:
            raise ValueError(
                f"Expected `values` to be 1-D, got shape {tuple(values.shape)}."
            )
        n = sample_ids.shape[0]
        if not (locations.shape[0] == n and values.shape[0] == n):
            raise ValueError(
                f"Length mismatch: `sample_ids` has {n} rows, "
                f"`locations` has {locations.shape[0]}, "
                f"`values` has {values.shape[0]}. All must match."
            )

    return sample_ids, locations, values


def _fit_irreg_pca_core(
    *,
    sample_ids: torch.Tensor,
    locations: torch.Tensor,
    values: torch.Tensor,
    n_components: int,
    epochs: int,
    lr: float,
    patience: int,
    valid_split: float,
    random_state: int | None,
    device: str | torch.device | None,
    model_factory: Callable[[int], torch.nn.Module] | None,
    verbose: bool,
    callbacks: list[Callable[[dict], None]] | None,
) -> IrregPCAResult:
    """Core fitting routine for IrregPCA.

    Trains one mean model and ``n_components`` component models sequentially,
    preserving the mathematical method from the original implementation.

    Returns
    -------
    IrregPCAResult
    """
    # --- device resolution ---
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    device_str = str(device)

    # --- pack split inputs into (N, d+2) for internal helpers ---
    data = torch.cat(
        [
            sample_ids.float().unsqueeze(-1),
            locations.float(),
            values.float().unsqueeze(-1),
        ],
        dim=1,
    ).to(device)

    d = locations.shape[1]
    k = n_components

    # --- train / validation split ---
    train_ratio = 1.0 - valid_split
    data_train, data_valid = split_data(data, train_ratio, random_state=random_state)

    # --- model creation ---
    if model_factory is None:
        models = [DefaultModel(d=d).to(device=device) for _ in range(k + 1)]
    else:
        models = [model_factory(d).to(device=device) for _ in range(k + 1)]

    # --- loss functions ---
    lossfn = [None] * (k + 1)
    lossfn[0] = lambda models, data: (
        -mean_fn(models[0], data)
        + 0.5 * inner_product(models[0], models[0], device=device)
    )
    for j in range(1, k + 1):
        lossfn[j] = (
            lambda j: lambda models, data: (
                -covariance_fn(models[j], data)
                + 0.5 * inner_product(models[j], models[j], device=device).pow(2)
                + (
                    torch.stack(
                        [
                            inner_product(models[i], models[j], device=device).pow(2)
                            for i in range(1, j)
                        ]
                    ).sum()
                    if j > 1
                    else torch.tensor(0.0, device=device)
                )
            )
        )(j)

    def loss_joint(models, data):
        return lossfn[0](models, data) + torch.stack(
            [lossfn[j](models, data) for j in range(1, k + 1)]
        ).sum()

    # --- history accumulators ---
    history = LossHistory(
        joint_train=[],
        joint_valid=[],
        component_train=[[] for _ in range(k + 1)],
        component_valid=[[] for _ in range(k + 1)],
    )

    # --- sequential training ---
    for j, model in enumerate(models):
        is_mean = j == 0
        component_index = None if is_mean else (j - 1)

        if verbose:
            label = "mean" if is_mean else f"component {component_index}"
            print(f"\n===> Training {label}")

        train_only(j, models)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        wait = 0
        min_valid_loss = float("inf")
        best_state = {
            name: v.detach().clone() for name, v in model.state_dict().items()
        }

        for epoch in range(epochs):
            # train step
            model.train()
            optimizer.zero_grad()
            loss_train = lossfn[j](models, data_train)
            loss_train.backward()
            optimizer.step()

            # validation step
            model.eval()
            with torch.no_grad():
                loss_valid = lossfn[j](models, data_valid)

            train_loss_val = loss_train.item()
            valid_loss_val = loss_valid.item()

            # joint losses (preserved from original: outside no_grad)
            joint_train_val = loss_joint(models, data_train).item()
            joint_valid_val = loss_joint(models, data_valid).item()

            # record history
            history.component_train[j].append(train_loss_val)
            history.component_valid[j].append(valid_loss_val)
            history.joint_train.append(joint_train_val)
            history.joint_valid.append(joint_valid_val)

            # early stopping
            if loss_valid < min_valid_loss:
                min_valid_loss = loss_valid
                best_state = {
                    name: v.detach().clone()
                    for name, v in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1

            if wait > patience:
                if verbose:
                    print("Early stop.")
                model.load_state_dict(best_state)
                break

            # epoch callbacks
            if callbacks:
                info = {
                    "stage": "training",
                    "component_index": component_index,
                    "epoch": epoch,
                    "train_loss": train_loss_val,
                    "valid_loss": valid_loss_val,
                    "joint_train_loss": joint_train_val,
                    "joint_valid_loss": joint_valid_val,
                    "is_mean": is_mean,
                    "device": device_str,
                }
                for cb in callbacks:
                    cb(info)

    # restore all models to eval mode with no gradients
    train_only(-1, models)

    return IrregPCAResult(
        mean_model=models[0],
        component_models=models[1:],
        history=history,
        n_components=n_components,
        input_dim=d,
        device=device_str,
    )
