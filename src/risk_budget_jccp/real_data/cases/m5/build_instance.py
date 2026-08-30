from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class M5Instance:
    demand_train: np.ndarray
    demand_test: np.ndarray
    cost: np.ndarray
    metadata: pd.DataFrame
    alpha: float

    @property
    def m(self) -> int:
        return int(self.demand_train.shape[1])


def build_instance(processed_dir: str | Path, *, alpha: float) -> M5Instance:
    root = Path(processed_dir)
    train = pd.read_csv(root / "demand_train.csv", index_col=0)
    test = pd.read_csv(root / "demand_test.csv", index_col=0)
    metadata = pd.read_csv(root / "series_metadata.csv")
    cost = metadata["median_price"].to_numpy(dtype=float)
    demand_train = train.to_numpy(dtype=float)
    demand_test = test.to_numpy(dtype=float)
    if demand_train.shape[1] != len(metadata) or demand_test.shape[1] != len(metadata):
        raise ValueError("M5 processed demand and metadata column counts do not match")
    if np.any(cost <= 0.0) or np.any(~np.isfinite(cost)):
        raise ValueError("M5 cost vector must be finite and positive")
    return M5Instance(
        demand_train=demand_train,
        demand_test=demand_test,
        cost=cost,
        metadata=metadata,
        alpha=float(alpha),
    )
