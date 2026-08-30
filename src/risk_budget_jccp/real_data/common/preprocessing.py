from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_tail_split(
    frame: pd.DataFrame,
    *,
    max_train: int,
    max_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_index(kind="stable")
    if len(ordered) < 2:
        raise ValueError("need at least two observations for calibration/held-out split")
    n_test = min(int(max_test), max(1, len(ordered) // 3))
    train_end = len(ordered) - n_test
    train_start = max(0, train_end - int(max_train))
    train = ordered.iloc[train_start:train_end].copy()
    test = ordered.iloc[train_end:].copy()
    if train.empty or test.empty:
        raise ValueError("split produced empty calibration or held-out set")
    return train, test


def cap_rows(array: np.ndarray, max_rows: int, *, from_tail: bool = True) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.shape[0] <= max_rows:
        return arr.copy()
    return arr[-max_rows:].copy() if from_tail else arr[:max_rows].copy()


def finite_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
