import torch


def split_data(data, train_ratio=0.8):
    idx = data[:, 0].long()
    unique_idx = idx.unique()
    
    n = unique_idx.numel()
    cut = int(train_ratio * n)
    permutation = torch.randperm(n, device=unique_idx.device)
    
    train_idx = unique_idx[permutation[:cut]]
    train_mask = torch.isin(idx, train_idx)
    valid_idx = unique_idx[permutation[cut:]]
    valid_mask = torch.isin(idx, valid_idx)

    return data[train_mask], data[valid_mask]