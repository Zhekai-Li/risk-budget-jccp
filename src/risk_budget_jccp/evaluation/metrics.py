from __future__ import annotations

import math

import numpy as np


def normalized_entropy(alpha_vec: np.ndarray) -> float:
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_vec must be one-dimensional")
    if alpha_arr.size == 0:
        raise ValueError("alpha_vec must be non-empty")
    if not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha_vec must be finite")
    if np.any(alpha_arr <= 0.0):
        raise ValueError("alpha_vec must be strictly positive")

    if alpha_arr.size == 1:
        return 1.0

    scale = float(np.max(alpha_arr))
    scaled_alpha = alpha_arr / scale
    probabilities = scaled_alpha / scaled_alpha.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(alpha_arr.size)


def relative_improvement_percent(equal_value: float, optimized_value: float) -> float:
    if not np.isfinite(equal_value) or not np.isfinite(optimized_value):
        raise ValueError("values must be finite")
    if equal_value <= 0.0:
        raise ValueError("equal_value must be positive")

    improvement = equal_value - optimized_value
    return 100.0 * improvement / equal_value
