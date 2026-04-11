from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Note: exact bit-for-bit reproducibility may vary across hardware and
    PyTorch versions, especially on GPU.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
