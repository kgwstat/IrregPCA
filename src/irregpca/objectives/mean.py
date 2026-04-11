from __future__ import annotations

import torch


def mean_fn_grouped(
    model: torch.nn.Module,
    locations: torch.Tensor,
    values: torch.Tensor,
    group_offsets: torch.Tensor,
    group_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean-function objective from a grouped batch.

    Objective (to maximise): E_i[ (1/n_i) Σ_j f(u_ij) * y_ij ]

    The loss used in training is the negation of this.

    Parameters
    ----------
    model : nn.Module
        Shape ``(N, d) → (N, 1)``.
    locations : torch.Tensor
        Flat concatenated observation locations. Shape ``(N_obs, d)``.
    values : torch.Tensor
        Flat concatenated observation values. Shape ``(N_obs,)``.
    group_offsets : torch.Tensor
        Start index of each sample's observations. Shape ``(n_samples,)``.
    group_lengths : torch.Tensor
        Number of observations per sample. Shape ``(n_samples,)``.

    Returns
    -------
    torch.Tensor
        Scalar — mean across samples of per-sample averages.
    """
    n_samples = group_offsets.shape[0]
    f_vals = model(locations).squeeze(-1)  # (N_obs,)
    weighted = f_vals * values  # (N_obs,)

    # compute per-sample means using offsets
    sample_means = torch.zeros(n_samples, dtype=weighted.dtype, device=weighted.device)
    for i in range(n_samples):
        start = group_offsets[i]
        length = group_lengths[i]
        sample_means[i] = weighted[start : start + length].mean()

    return sample_means.mean()


def mean_fn_packed(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean-function objective from a packed data tensor.

    Legacy full-batch interface. ``data`` shape ``(N, d+2)`` with columns
    ``[sample_id, loc..., value]``.

    Parameters
    ----------
    model : nn.Module
    data : torch.Tensor
        Shape ``(N, d+2)``.

    Returns
    -------
    torch.Tensor
        Scalar.
    """
    idx = data[:, 0].long()
    loc = data[:, 1:-1]
    obs = data[:, -1]

    unique_idx, inverse = torch.unique(idx, return_inverse=True)
    n = unique_idx.numel()

    vals = model(loc).squeeze(-1) * obs
    sample_sums = vals.new_zeros(n)
    sample_sums.scatter_add_(0, inverse, vals)
    sample_counts = torch.bincount(inverse).to(vals.dtype)

    return (sample_sums / sample_counts).mean()
