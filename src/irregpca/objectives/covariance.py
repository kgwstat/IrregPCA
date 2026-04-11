from __future__ import annotations

import torch


def covariance_fn_grouped(
    model: torch.nn.Module,
    locations: torch.Tensor,
    values: torch.Tensor,
    group_offsets: torch.Tensor,
    group_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the covariance-function objective from a grouped batch.

    Objective (to maximise):
        within_sample - between_sample

    where
        within_sample  = E_i[ (1/(n_i(n_i-1))) Σ_{j≠k} f(u_ij) y_ij f(u_ik) y_ik ]
        between_sample = E_{i≠j}[ mean_i * mean_j ]  (cross-sample term)

    Only samples with >= 2 observations contribute to within_sample.

    Parameters
    ----------
    model : nn.Module
    locations : torch.Tensor
        Shape ``(N_obs, d)``.
    values : torch.Tensor
        Shape ``(N_obs,)``.
    group_offsets : torch.Tensor
        Shape ``(n_samples,)``.
    group_lengths : torch.Tensor
        Shape ``(n_samples,)``.

    Returns
    -------
    torch.Tensor
        Scalar.
    """
    n_samples = group_offsets.shape[0]
    f_vals = model(locations).squeeze(-1)  # (N_obs,)
    weighted = f_vals * values  # (N_obs,)

    # per-sample sufficient statistics
    sums1 = torch.zeros(n_samples, dtype=weighted.dtype, device=weighted.device)
    sums2 = torch.zeros(n_samples, dtype=weighted.dtype, device=weighted.device)
    counts = group_lengths.to(weighted.dtype)

    for i in range(n_samples):
        start = group_offsets[i]
        length = group_lengths[i]
        chunk = weighted[start : start + length]
        sums1[i] = chunk.sum()
        sums2[i] = chunk.pow(2).sum()

    # within-sample term: only samples with >= 2 observations
    mask = group_lengths >= 2
    if mask.sum() < 2:
        return weighted.new_tensor(0.0)

    s1 = sums1[mask]
    s2 = sums2[mask]
    cnt = counts[mask]
    m = s1.shape[0]

    within_sample = ((s1.pow(2) - s2) / (cnt * (cnt - 1))).mean()

    # between-sample term
    cluster_means = s1 / cnt
    between_sample = (
        cluster_means.sum().pow(2) - cluster_means.pow(2).sum()
    ) / (m * (m - 1))

    return within_sample - between_sample


def covariance_fn_packed(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute the covariance-function objective from a packed data tensor.

    Legacy full-batch interface. ``data`` shape ``(N, d+2)``.

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

    if n < 2:
        return vals.new_tensor(0.0)

    sample_counts = torch.bincount(inverse).to(vals.dtype)

    sample_sums1 = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    sample_sums2 = torch.zeros(n, dtype=vals.dtype, device=vals.device)

    sample_sums1.scatter_add_(0, inverse, vals)
    sample_sums2.scatter_add_(0, inverse, vals.pow(2))

    mask = (sample_counts > 1)

    sample_sums1 = sample_sums1[mask]
    sample_sums2 = sample_sums2[mask]
    sample_counts = sample_counts[mask]

    m = sample_counts.shape[0]

    if m < 2:
        return vals.new_tensor(0.0)

    within_sample = (
        (sample_sums1.pow(2) - sample_sums2)
        / (sample_counts * (sample_counts - 1))
    ).mean()

    cluster_means = sample_sums1 / sample_counts

    between_sample = (
        cluster_means.sum().pow(2)
        - cluster_means.pow(2).sum()
    ) / (m * (m - 1))

    return within_sample - between_sample
