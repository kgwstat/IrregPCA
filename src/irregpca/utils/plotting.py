from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_cpu(x: Any) -> np.ndarray:
    """Convert a tensor, array, or scalar to a CPU NumPy array.

    Parameters
    ----------
    x : torch.Tensor, numpy.ndarray, list, or scalar
        Input value to convert.

    Returns
    -------
    numpy.ndarray
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)
