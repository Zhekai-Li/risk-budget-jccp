from __future__ import annotations

import numpy as np

from risk_budget_jccp.real_data.common.metrics import normalized_entropy, violation_metrics
from risk_budget_jccp.real_data.common.result_status import status_fields


def test_normalized_entropy_equal_and_concentrated() -> None:
    assert np.isclose(normalized_entropy(np.array([1.0, 1.0, 1.0])), 1.0)
    assert normalized_entropy(np.array([0.98, 0.01, 0.01])) < 0.2


def test_violation_metrics_joint_and_counts() -> None:
    y = np.array([[-1.0, -2.0], [0.2, -1.0], [0.1, 0.3]])
    metrics = violation_metrics(y, reference_scale=np.array([1.0, 1.0]))
    assert np.isclose(metrics["empirical_joint_violation"], 2 / 3)
    assert np.isclose(metrics["average_scalar_violations"], 1.0)
    assert metrics["max_scalar_violations"] == 2


def test_violation_metrics_uses_per_constraint_relative_scale() -> None:
    y = np.array([[5.0e-5, 5.0e-8], [2.0e-4, 2.0e-6]])
    metrics = violation_metrics(y, reference_scale=np.array([100.0, 1.0]))
    assert np.allclose(metrics["violation_tolerance"], [1.00000001e-04, 1.000001e-06])
    assert np.isclose(metrics["empirical_joint_violation"], 0.5)
    assert metrics["scalar_violation_rates"] == [0.5, 0.5]


def test_result_status_acceptance_and_strict_tolerances() -> None:
    fields = status_fields(
        case="power",
        certificate="bernstein",
        allocation="optimized",
        solver_status="optimal",
        valid_certificate=True,
        valid_optimization=True,
        fallback_used=False,
        feasibility_residual=1.0e-5,
        calibration_joint_violation=0.02,
        alpha_total=0.05,
        dca_cfg={"certificate_accept_tol": 1.0e-4, "certificate_strict_tol": 1.0e-6},
    )
    assert fields["result_status"] == "success"
    assert fields["algorithm_class"] == "paper_dca"
    assert fields["passes_certificate_acceptance"] is True
    assert fields["passes_certificate_strict"] is False
    assert fields["calibration_jvp_contract"] == "not_implied_by_moment_certificate"

    fallback = status_fields(
        case="power",
        certificate="cvar",
        allocation="optimized",
        solver_status="cvar_optimized_fallback_equal_objective_dominates",
        valid_certificate=True,
        valid_optimization=False,
        fallback_used=True,
        feasibility_residual=0.0,
        calibration_joint_violation=0.02,
        alpha_total=0.05,
    )
    assert fallback["result_status"] == "fallback_equal"
