from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class FrenchInstance:
    returns_train: np.ndarray
    returns_test: np.ndarray
    industries: tuple[str, ...]
    metadata: pd.DataFrame
    alpha: float
    max_weight: float
    target_return: float

    @property
    def m(self) -> int:
        return int(self.returns_train.shape[1])


def _target_return(mean_returns: np.ndarray, max_weight: float, fraction: float) -> float:
    m = len(mean_returns)
    bounds = [(0.0, max_weight)] * m
    a_eq = np.ones((1, m))
    b_eq = np.array([1.0])
    min_res = linprog(mean_returns, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    max_res = linprog(-mean_returns, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not min_res.success or not max_res.success:
        raise RuntimeError("failed to compute French target-return LP bounds")
    r_min = float(mean_returns @ min_res.x)
    r_max = float(mean_returns @ max_res.x)
    return r_min + float(fraction) * (r_max - r_min)


def build_instance(
    processed_dir: str | Path,
    *,
    alpha: float,
    max_weight: float,
    target_return_fraction: float,
) -> FrenchInstance:
    root = Path(processed_dir)
    train = pd.read_csv(root / "returns_train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(root / "returns_test.csv", index_col=0, parse_dates=True)
    metadata = pd.read_csv(root / "industry_metadata.csv")
    returns_train = train.to_numpy(dtype=float)
    returns_test = test.to_numpy(dtype=float)
    industries = tuple(train.columns.astype(str).tolist())
    if returns_train.shape[1] != len(industries) or returns_test.shape[1] != len(industries):
        raise ValueError("French processed train/test shapes do not match industries")
    mean_returns = returns_train.mean(axis=0)
    return FrenchInstance(
        returns_train=returns_train,
        returns_test=returns_test,
        industries=industries,
        metadata=metadata,
        alpha=float(alpha),
        max_weight=float(max_weight),
        target_return=_target_return(mean_returns, float(max_weight), float(target_return_fraction)),
    )
