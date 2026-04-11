from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device specification to a ``torch.device``.

    Auto-selects ``cuda > mps > cpu`` when ``device=None``.

    Parameters
    ----------
    device : str, torch.device, or None

    Returns
    -------
    torch.device
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device
