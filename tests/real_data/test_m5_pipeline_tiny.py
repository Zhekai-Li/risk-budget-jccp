from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from risk_budget_jccp.real_data.cases.m5.build_instance import M5Instance
from risk_budget_jccp.real_data.cases.m5.report import build_m5_report_assets
from risk_budget_jccp.real_data.cases.m5.solve import solve_m5_instance


def test_m5_tiny_pipeline(tmp_path: Path) -> None:
    demand_train = np.array(
        [
            [3, 4, 1],
            [4, 5, 2],
            [6, 4, 3],
            [5, 7, 1],
            [7, 5, 2],
            [8, 6, 3],
        ],
        dtype=float,
    )
    demand_test = np.array([[5, 6, 2], [9, 5, 4], [4, 8, 1]], dtype=float)
    metadata = pd.DataFrame(
        {
            "id": ["A_X", "B_X", "C_Y"],
            "item_id": ["A", "B", "C"],
            "store_id": ["X", "X", "Y"],
            "dept_id": ["D1", "D1", "D2"],
            "cat_id": ["C1", "C1", "C2"],
            "state_id": ["S", "S", "T"],
            "mean_demand": demand_train.mean(axis=0),
            "std_demand": demand_train.std(axis=0),
            "coefficient_of_variation": demand_train.std(axis=0) / demand_train.mean(axis=0),
            "median_price": [2.0, 3.0, 1.5],
        }
    )
    instance = M5Instance(demand_train, demand_test, np.array([2.0, 3.0, 1.5]), metadata, alpha=0.2)
    cfg = {"max_iter": 2, "tol": 1.0e-3, "proximal_weight": 1.0e-4, "min_alpha": 1.0e-6, "feasibility_tol": 1.0e-6}
    summary = solve_m5_instance(instance, tmp_path, cfg)
    assert set(summary["certificate"]) == {"cvar", "bernstein", "cantelli"}
    assert set(summary["allocation"]) == {"equal", "optimized"}
    build_m5_report_assets(instance, summary, tmp_path)
    assert (tmp_path / "tables" / "tab_m5_summary.tex").is_file()
