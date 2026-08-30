from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PowerInstance:
    case_name: str
    alpha: float
    branch_ids: tuple[str, ...]
    from_bus: tuple[str, ...]
    to_bus: tuple[str, ...]
    directions: tuple[str, ...]
    line_limits: np.ndarray
    base_flow: np.ndarray
    flow_residual_train: np.ndarray
    flow_residual_test: np.ndarray
    generator_ids: tuple[str, ...]
    generator_cost: np.ndarray
    generator_pmin: np.ndarray
    generator_pmax: np.ndarray
    generator_bus_matrix: np.ndarray
    dispatch_flow_matrix: np.ndarray
    forecast_flow_offset: np.ndarray
    actual_flow_offset: np.ndarray
    load_forecast_total: np.ndarray
    renewable_forecast_total: np.ndarray
    load_actual_total: np.ndarray
    renewable_actual_total: np.ndarray
    selected_test_indices: np.ndarray
    cont_limits: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    lte_limits: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    ste_limits: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    primary_rating: str = "lte"
    slack_penalty: float = 1.0e6

    @property
    def m(self) -> int:
        return int(self.flow_residual_train.shape[1])


def build_instance(processed_dir: str | Path, *, alpha: float) -> PowerInstance:
    root = Path(processed_dir)
    arrays = np.load(root / "power_instance.npz", allow_pickle=True)
    metadata = pd.read_csv(root / "line_metadata.csv")
    gen = pd.read_csv(root / "generator_metadata.csv")
    risk_col = "risk_limit" if "risk_limit" in metadata.columns else "thermal_limit"
    cont = metadata["cont_rating"].to_numpy(dtype=float) if "cont_rating" in metadata.columns else metadata[risk_col].to_numpy(dtype=float)
    lte = metadata["lte_rating"].to_numpy(dtype=float) if "lte_rating" in metadata.columns else metadata[risk_col].to_numpy(dtype=float)
    ste = metadata["ste_rating"].to_numpy(dtype=float) if "ste_rating" in metadata.columns else lte.copy()
    primary = str(arrays["primary_rating"].item()) if "primary_rating" in arrays.files else "lte"
    return PowerInstance(
        case_name=str(arrays["case_name"].item()),
        alpha=float(alpha),
        branch_ids=tuple(metadata["branch_id"].astype(str)),
        from_bus=tuple(metadata["from_bus"].astype(str)),
        to_bus=tuple(metadata["to_bus"].astype(str)),
        directions=tuple(metadata["direction"].astype(str)),
        line_limits=metadata[risk_col].to_numpy(dtype=float),
        base_flow=metadata["base_flow"].to_numpy(dtype=float),
        cont_limits=cont,
        lte_limits=lte,
        ste_limits=ste,
        primary_rating=primary,
        flow_residual_train=arrays["flow_residual_train"].astype(float),
        flow_residual_test=arrays["flow_residual_test"].astype(float),
        generator_ids=tuple(gen["generator_id"].astype(str)),
        generator_cost=gen["cost"].to_numpy(dtype=float),
        generator_pmin=gen["pmin"].to_numpy(dtype=float),
        generator_pmax=gen["pmax"].to_numpy(dtype=float),
        generator_bus_matrix=arrays["generator_bus_matrix"].astype(float),
        dispatch_flow_matrix=arrays["dispatch_flow_matrix"].astype(float),
        forecast_flow_offset=arrays["forecast_flow_offset"].astype(float),
        actual_flow_offset=arrays["actual_flow_offset"].astype(float),
        load_forecast_total=arrays["load_forecast_total"].astype(float),
        renewable_forecast_total=arrays["renewable_forecast_total"].astype(float),
        load_actual_total=arrays["load_actual_total"].astype(float),
        renewable_actual_total=arrays["renewable_actual_total"].astype(float),
        selected_test_indices=arrays["selected_test_indices"].astype(int),
    )
