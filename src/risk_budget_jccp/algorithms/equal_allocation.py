from __future__ import annotations

import numpy as np


def equal_allocation(m: int, alpha: float) -> np.ndarray:
    if m <= 0:
        raise ValueError("m must be greater than 0")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    return np.full(m, alpha / m, dtype=float)
