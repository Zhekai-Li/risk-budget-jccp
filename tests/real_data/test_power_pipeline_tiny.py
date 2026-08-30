from __future__ import annotations

from pathlib import Path

import numpy as np

from risk_budget_jccp.real_data.cases.power.build_instance import PowerInstance
from risk_budget_jccp.real_data.cases.power.solve import solve_power_instance


def test_power_tiny_pipeline(tmp_path: Path) -> None:
    instance = PowerInstance(
        case_name="tiny",
        alpha=0.2,
        branch_ids=("L1", "L1"),
        from_bus=("1", "1"),
        to_bus=("2", "2"),
        directions=("positive", "negative"),
        line_limits=np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        base_flow=np.zeros(6),
        flow_residual_train=np.tile(
            np.array([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0], [0.5, -0.5]]),
            3,
        ),
        flow_residual_test=np.tile(np.array([[1.5, -1.5], [-0.5, 0.5], [2.0, -2.0]]), 3),
        generator_ids=("G1",),
        generator_cost=np.array([20.0]),
        generator_pmin=np.array([0.0]),
        generator_pmax=np.array([200.0]),
        generator_bus_matrix=np.array([[1.0], [0.0]]),
        dispatch_flow_matrix=np.array([[0.1]]),
        forecast_flow_offset=np.array([[0.0], [0.0], [0.0]]),
        actual_flow_offset=np.array([[0.0], [0.0], [0.0]]),
        load_forecast_total=np.array([80.0, 90.0, 85.0]),
        renewable_forecast_total=np.array([10.0, 10.0, 10.0]),
        load_actual_total=np.array([82.0, 91.0, 85.0]),
        renewable_actual_total=np.array([10.0, 9.0, 10.0]),
        selected_test_indices=np.array([0, 1, 2]),
    )
    cfg = {"max_iter": 2, "tol": 1.0e-3, "proximal_weight": 1.0e-4, "min_alpha": 1.0e-6, "feasibility_tol": 1.0e-6}
    summary = solve_power_instance(instance, tmp_path, cfg)
    assert len(summary) == 6
    assert np.isfinite(summary["objective"]).all()
    for column in ("certificate_residual_max", "budget_residual", "normalized_certificate_residual_max"):
        assert column in summary.columns
    finite = summary.loc[summary["valid_certificate"], ["certificate_residual_max", "budget_residual"]]
    assert np.isfinite(finite.to_numpy(dtype=float)).all()
