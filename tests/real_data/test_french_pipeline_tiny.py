from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from risk_budget_jccp.real_data.cases.french.build_instance import FrenchInstance
from risk_budget_jccp.real_data.cases.french.report import build_french_report_assets
from risk_budget_jccp.real_data.cases.french.solve import solve_french_instance


def test_french_tiny_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(0.0008, 0.01, size=(20, 4))
    test = rng.normal(0.0006, 0.012, size=(8, 4))
    metadata = pd.DataFrame(
        {
            "industry": ["A", "B", "C", "D"],
            "mean_return": train.mean(axis=0),
            "volatility": train.std(axis=0),
            "downside_cvar_95": np.abs(np.quantile(np.minimum(train, 0.0), 0.05, axis=0)),
        }
    )
    instance = FrenchInstance(
        returns_train=train,
        returns_test=test,
        industries=("A", "B", "C", "D"),
        metadata=metadata,
        alpha=0.25,
        max_weight=0.5,
        target_return=float(train.mean(axis=0).mean()),
    )
    cfg = {"max_iter": 2, "tol": 1.0e-3, "proximal_weight": 1.0e-4, "min_alpha": 1.0e-6, "feasibility_tol": 1.0e-6}
    summary = solve_french_instance(instance, tmp_path, cfg)
    assert len(summary) == 6
    build_french_report_assets(instance, summary, tmp_path)
    assert (tmp_path / "tables" / "tab_french_summary.tex").is_file()
