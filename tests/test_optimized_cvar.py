from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import risk_budget_jccp.algorithms.optimized_cvar as optimized_cvar
from risk_budget_jccp.algorithms.optimized_cvar import (
    finite_scenario_cvar,
    solve_equal_allocation_cvar,
    solve_optimized_allocation_cvar,
)


def test_equal_and_optimized_allocations_sum_to_alpha() -> None:
    scenario_losses = np.array(
        [
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3],
            [0.1, 0.2, 0.3, 0.35, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0],
            [0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.35, 1.5, 1.7, 1.85, 1.95, 2.1],
        ],
        dtype=float,
    )
    alpha = 0.3

    equal_result = solve_equal_allocation_cvar(scenario_losses=scenario_losses, alpha=alpha)
    optimized_result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=alpha,
    )

    assert np.isclose(equal_result.alpha_allocation.sum(), alpha, atol=1e-10)
    assert np.isclose(optimized_result.alpha_allocation.sum(), alpha, atol=1e-8)
    assert np.isclose(optimized_result.diagnostics.alpha_allocation.sum(), alpha, atol=1e-8)


def test_symmetric_inputs_recover_equal_allocation() -> None:
    scenario_losses = np.array(
        [
            [0.2, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8],
            [0.2, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8],
            [0.2, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8],
        ],
        dtype=float,
    )
    alpha = 0.36

    optimized_result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=alpha,
    )

    assert np.allclose(
        optimized_result.alpha_allocation,
        np.full(3, alpha / 3.0),
        atol=1e-6,
    )


def test_optimized_objective_is_no_worse_than_equal_allocation() -> None:
    scenario_losses = np.array(
        [
            [0.2, 0.25, 0.3, 0.35, 0.42, 0.5, 0.58, 0.65, 0.7, 0.74, 0.77, 0.8],
            [0.1, 0.14, 0.18, 0.22, 0.26, 0.33, 0.4, 0.46, 0.53, 0.6, 0.66, 0.72],
            [0.6, 0.75, 0.92, 1.1, 1.28, 1.45, 1.7, 1.95, 2.2, 2.55, 2.9, 3.2],
        ],
        dtype=float,
    )
    alpha = 0.3

    equal_result = solve_equal_allocation_cvar(scenario_losses=scenario_losses, alpha=alpha)
    optimized_result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=alpha,
    )

    assert optimized_result.objective_value <= equal_result.objective_value + 1e-8
    assert optimized_result.alpha_allocation[2] > equal_result.alpha_allocation[2]


def test_outputs_are_finite_and_diagnostics_are_populated() -> None:
    scenario_losses = np.array(
        [
            [-0.3, -0.1, 0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.05, 1.18, 1.3],
            [0.1, 0.12, 0.18, 0.22, 0.3, 0.36, 0.42, 0.5, 0.58, 0.65, 0.73, 0.8],
            [0.0, 0.15, 0.35, 0.55, 0.8, 1.0, 1.25, 1.5, 1.8, 2.05, 2.3, 2.5],
        ],
        dtype=float,
    )

    result = solve_optimized_allocation_cvar(scenario_losses=scenario_losses, alpha=0.27)

    assert np.isfinite(result.objective_value)
    assert np.all(np.isfinite(result.alpha_allocation))
    assert np.all(np.isfinite(result.per_constraint_cvar))
    assert result.diagnostics.runtime >= 0.0
    assert result.diagnostics.iterations >= 0
    assert np.isfinite(result.diagnostics.final_residual)
    assert isinstance(result.diagnostics.solver_status, str)
    assert result.diagnostics.solver_status
    assert np.all(np.isfinite(result.diagnostics.alpha_allocation))


def test_optimized_allocation_recovers_weighted_boundary_optimum() -> None:
    scenario_losses = np.array(
        [
            [-1.04300108, -1.02980444, -0.01391467, 0.26841708, 0.35867195, 1.32245747],
            [-2.36530391, 0.33962001, 1.04183976, 1.15016564, 1.22868372, 1.40226483],
        ],
        dtype=float,
    )
    scenario_probabilities = np.array(
        [0.11860498, 0.26549305, 0.01256536, 0.24003588, 0.12119401, 0.24210673],
        dtype=float,
    )
    alpha = 0.25

    result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=alpha,
        scenario_probabilities=scenario_probabilities,
    )

    assert np.allclose(result.alpha_allocation, np.array([alpha, 0.0]), atol=1e-6)
    assert np.isclose(result.objective_value, 2.6942926159193714, atol=1e-8)


def test_finite_scenario_cvar_alpha_zero_ignores_zero_probability_losses() -> None:
    losses = np.array([100.0, 2.0, 1.0], dtype=float)
    scenario_probabilities = np.array([0.0, 0.6, 0.4], dtype=float)

    result = finite_scenario_cvar(
        losses,
        alpha=0.0,
        scenario_probabilities=scenario_probabilities,
    )

    assert result == 2.0


def test_failed_slsqp_candidate_does_not_override_restart_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_losses = np.array(
        [
            [-1.04300108, -1.02980444, -0.01391467, 0.26841708, 0.35867195, 1.32245747],
            [-2.36530391, 0.33962001, 1.04183976, 1.15016564, 1.22868372, 1.40226483],
        ],
        dtype=float,
    )
    scenario_probabilities = np.array(
        [0.11860498, 0.26549305, 0.01256536, 0.24003588, 0.12119401, 0.24210673],
        dtype=float,
    )
    alpha = 0.25

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([alpha, 0.0], dtype=float),
            nit=99,
            success=False,
            message="forced failure",
        )

    monkeypatch.setattr(optimized_cvar, "minimize", fake_minimize)

    result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=alpha,
        scenario_probabilities=scenario_probabilities,
    )

    assert np.allclose(result.alpha_allocation, np.array([alpha, 0.0]), atol=1e-8)
    assert result.diagnostics.solver_status == "restart_candidate"


def test_limited_focused_restarts_bound_slsqp_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_losses = np.array(
        [
            [0.0, 0.1, 0.3, 0.7, 1.2],
            [0.0, 0.2, 0.4, 0.9, 1.7],
            [0.0, 0.05, 0.1, 0.2, 0.4],
            [0.0, 0.3, 0.8, 1.4, 2.4],
        ],
        dtype=float,
    )
    call_count = 0

    def fake_minimize(fun, x0, **kwargs):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            nit=1,
            success=True,
            message="fake success",
        )

    monkeypatch.setattr(optimized_cvar, "minimize", fake_minimize)

    solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=0.2,
        max_focused_constraints=2,
    )

    assert call_count == 5


def test_candidate_only_mode_skips_slsqp(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_losses = np.array(
        [
            [0.0, 0.1, 0.3, 0.7, 1.2],
            [0.0, 0.2, 0.4, 0.9, 1.7],
            [0.0, 0.05, 0.1, 0.2, 0.4],
        ],
        dtype=float,
    )

    def fail_minimize(*args, **kwargs):
        raise AssertionError("SLSQP should not be called in candidate-only mode")

    monkeypatch.setattr(optimized_cvar, "minimize", fail_minimize)

    result = solve_optimized_allocation_cvar(
        scenario_losses=scenario_losses,
        alpha=0.2,
        max_focused_constraints=2,
        run_slsqp=False,
    )

    assert np.isfinite(result.objective_value)
    assert np.isclose(result.alpha_allocation.sum(), 0.2)
