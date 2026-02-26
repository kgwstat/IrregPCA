
import torch

def inner_product(model1, model2, device):
    grid = torch.linspace(0, 1, steps=4096).to(device).unsqueeze(-1)
    vals1 = model1(grid).squeeze(-1)
    vals2 = model2(grid).squeeze(-1)
    return (vals1 * vals2).mean()

def mean_fn(model, data):
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

def covariance_fn(model, data):
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

