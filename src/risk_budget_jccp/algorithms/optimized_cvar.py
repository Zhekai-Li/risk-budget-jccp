from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

_FINITE_TOL = 1e-12


@dataclass(frozen=True)
class CvarOptimizationDiagnostics:
    """Diagnostics returned by the shared finite-scenario CVaR routines."""

    runtime: float
    iterations: int
    final_residual: float
    solver_status: str
    alpha_allocation: np.ndarray


@dataclass(frozen=True)
class FiniteScenarioCvarResult:
    """Result bundle for a finite-scenario CVaR budget-allocation workflow."""

    objective_value: float
    alpha_allocation: np.ndarray
    per_constraint_cvar: np.ndarray
    diagnostics: CvarOptimizationDiagnostics


@dataclass(frozen=True)
class _CvarProfile:
    losses: np.ndarray
    probabilities: np.ndarray
    cumulative_probabilities: np.ndarray
    cumulative_losses: np.ndarray


def _validate_scenario_losses(scenario_losses: np.ndarray | Iterable[float]) -> np.ndarray:
    losses = np.asarray(scenario_losses, dtype=float)
    if losses.ndim == 1:
        losses = losses[np.newaxis, :]
    if losses.ndim != 2:
        raise ValueError("scenario_losses must be one- or two-dimensional")
    if losses.shape[0] == 0 or losses.shape[1] == 0:
        raise ValueError("scenario_losses must be non-empty")
    if not np.all(np.isfinite(losses)):
        raise ValueError("scenario_losses must be finite")
    return losses


def _validate_probabilities(
    scenario_probabilities: np.ndarray | Iterable[float] | None,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_probabilities is None:
        return np.full(n_scenarios, 1.0 / n_scenarios, dtype=float)

    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("scenario_probabilities must be one-dimensional")
    if probabilities.shape[0] != n_scenarios:
        raise ValueError("scenario_probabilities length must match the number of scenarios")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("scenario_probabilities must be finite")
    if np.any(probabilities < 0.0):
        raise ValueError("scenario_probabilities must be nonnegative")

    total_probability = float(probabilities.sum())
    if total_probability <= 0.0:
        raise ValueError("scenario_probabilities must sum to a positive value")
    return probabilities / total_probability


def _validate_total_alpha(alpha: float) -> float:
    alpha_value = float(alpha)
    if not (0.0 < alpha_value <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    return alpha_value


def _build_profile(losses: np.ndarray, probabilities: np.ndarray) -> _CvarProfile:
    order = np.argsort(losses)[::-1]
    ordered_losses = np.asarray(losses[order], dtype=float)
    ordered_probabilities = np.asarray(probabilities[order], dtype=float)
    cumulative_probabilities = np.cumsum(ordered_probabilities)
    cumulative_losses = np.cumsum(ordered_probabilities * ordered_losses)
    cumulative_probabilities[-1] = 1.0
    return _CvarProfile(
        losses=ordered_losses,
        probabilities=ordered_probabilities,
        cumulative_probabilities=cumulative_probabilities,
        cumulative_losses=cumulative_losses,
    )


def _cvar_from_profile(profile: _CvarProfile, alpha: float) -> tuple[float, float]:
    alpha_value = float(np.clip(alpha, 0.0, 1.0))
    if alpha_value <= _FINITE_TOL:
        positive_probability = profile.probabilities > _FINITE_TOL
        first_supported_index = int(np.argmax(positive_probability))
        return float(profile.losses[first_supported_index]), 0.0
    if alpha_value >= 1.0 - _FINITE_TOL:
        mean_loss = float(np.dot(profile.probabilities, profile.losses))
        derivative = float(profile.losses[-1] - mean_loss)
        return mean_loss, derivative

    index = int(np.searchsorted(profile.cumulative_probabilities, alpha_value, side="left"))
    left_probability = 0.0 if index == 0 else float(profile.cumulative_probabilities[index - 1])
    left_loss = 0.0 if index == 0 else float(profile.cumulative_losses[index - 1])
    threshold = float(profile.losses[index])
    cvar = (left_loss + (alpha_value - left_probability) * threshold) / alpha_value
    derivative = (threshold - cvar) / alpha_value
    return float(cvar), float(derivative)


def finite_scenario_cvar(
    losses: np.ndarray | Iterable[float],
    alpha: float,
    *,
    scenario_probabilities: np.ndarray | Iterable[float] | None = None,
) -> float:
    """Return the empirical CVaR of a scalar loss under a finite scenario set.

    Parameters
    ----------
    losses:
        Scalar loss values for one safe constraint across scenarios.
    alpha:
        Tail probability in ``[0, 1]``. ``alpha = 0`` is interpreted in the
        right-limit sense and returns the worst-case scenario loss.
    scenario_probabilities:
        Optional nonnegative scenario weights. If omitted, scenarios are equally
        weighted.
    """

    loss_vector = np.asarray(losses, dtype=float)
    if loss_vector.ndim != 1 or loss_vector.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(loss_vector)):
        raise ValueError("losses must be finite")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha <= 1")

    probabilities = _validate_probabilities(scenario_probabilities, loss_vector.size)
    profile = _build_profile(loss_vector, probabilities)
    return _cvar_from_profile(profile, float(alpha))[0]


def _evaluate_allocation(
    profiles: list[_CvarProfile],
    alpha_allocation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    per_constraint_cvar = np.empty(len(profiles), dtype=float)
    gradient = np.empty(len(profiles), dtype=float)
    for index, (profile, allocation) in enumerate(zip(profiles, alpha_allocation, strict=False)):
        cvar, derivative = _cvar_from_profile(profile, float(allocation))
        per_constraint_cvar[index] = cvar
        gradient[index] = derivative
    return float(per_constraint_cvar.sum()), per_constraint_cvar, gradient


def _make_result(
    *,
    objective_value: float,
    alpha_allocation: np.ndarray,
    per_constraint_cvar: np.ndarray,
    runtime: float,
    iterations: int,
    final_residual: float,
    solver_status: str,
) -> FiniteScenarioCvarResult:
    allocation = np.asarray(alpha_allocation, dtype=float).copy()
    per_constraint = np.asarray(per_constraint_cvar, dtype=float).copy()
    diagnostics = CvarOptimizationDiagnostics(
        runtime=float(runtime),
        iterations=int(iterations),
        final_residual=float(final_residual),
        solver_status=solver_status,
        alpha_allocation=allocation.copy(),
    )
    return FiniteScenarioCvarResult(
        objective_value=float(objective_value),
        alpha_allocation=allocation,
        per_constraint_cvar=per_constraint,
        diagnostics=diagnostics,
    )


def solve_equal_allocation_cvar(
    *,
    scenario_losses: np.ndarray | Iterable[Iterable[float]] | Iterable[float],
    alpha: float,
    scenario_probabilities: np.ndarray | Iterable[float] | None = None,
) -> FiniteScenarioCvarResult:
    """Evaluate the finite-scenario CVaR objective under an equal alpha split."""

    start_time = time.perf_counter()
    losses = _validate_scenario_losses(scenario_losses)
    total_alpha = _validate_total_alpha(alpha)
    probabilities = _validate_probabilities(scenario_probabilities, losses.shape[1])

    alpha_allocation = np.full(losses.shape[0], total_alpha / losses.shape[0], dtype=float)
    profiles = [_build_profile(loss_row, probabilities) for loss_row in losses]
    objective_value, per_constraint_cvar, _ = _evaluate_allocation(profiles, alpha_allocation)
    runtime = time.perf_counter() - start_time

    return _make_result(
        objective_value=objective_value,
        alpha_allocation=alpha_allocation,
        per_constraint_cvar=per_constraint_cvar,
        runtime=runtime,
        iterations=0,
        final_residual=abs(float(alpha_allocation.sum()) - total_alpha),
        solver_status="equal_allocation",
    )


def _candidate_initial_allocations(
    n_constraints: int,
    total_alpha: float,
    *,
    max_focused_constraints: int | None = None,
    focus_scores: np.ndarray | None = None,
) -> list[np.ndarray]:
    equal = np.full(n_constraints, total_alpha / n_constraints, dtype=float)
    if n_constraints == 1:
        return [equal]

    if max_focused_constraints is None:
        focus_indices = list(range(n_constraints))
    else:
        if max_focused_constraints <= 0:
            raise ValueError("max_focused_constraints must be positive when provided")
        if focus_scores is None:
            focus_indices = list(range(min(n_constraints, max_focused_constraints)))
        else:
            score_array = np.asarray(focus_scores, dtype=float)
            if score_array.shape != (n_constraints,):
                raise ValueError("focus_scores must have one entry per constraint")
            if not np.all(np.isfinite(score_array)):
                raise ValueError("focus_scores must be finite")
            focus_indices = np.argsort(score_array)[::-1][:max_focused_constraints].tolist()

    candidates = [equal]
    for focus_index in focus_indices:
        vertex = np.zeros(n_constraints, dtype=float)
        vertex[focus_index] = total_alpha
        candidates.append(vertex)

        candidate = np.full(n_constraints, 0.2 * total_alpha / (n_constraints - 1), dtype=float)
        candidate[focus_index] = 0.8 * total_alpha
        candidates.append(candidate)
    return candidates


def solve_optimized_allocation_cvar(
    *,
    scenario_losses: np.ndarray | Iterable[Iterable[float]] | Iterable[float],
    alpha: float,
    scenario_probabilities: np.ndarray | Iterable[float] | None = None,
    maxiter: int = 300,
    tol: float = 1e-9,
    max_focused_constraints: int | None = None,
    run_slsqp: bool = True,
) -> FiniteScenarioCvarResult:
    """Optimize the alpha split for separable finite-scenario CVaR constraints.

    Parameters
    ----------
    scenario_losses:
        An array of shape ``(m, n_scenarios)`` with one scalar safe constraint
        per row, or a one-dimensional array for the single-constraint case.
    alpha:
        Total Bonferroni risk budget to allocate across constraints.
    scenario_probabilities:
        Optional nonnegative scenario weights. If omitted, scenarios are equally
        weighted.
    maxiter:
        Maximum number of SciPy solver iterations per start.
    tol:
        Solver tolerance passed to SLSQP.
    max_focused_constraints:
        Optional cap on focused restart constraints. If omitted, the solver uses
        the historical all-constraint restart set. If provided, it keeps the
        equal restart plus vertex and focused restarts for the constraints with
        largest empirical loss standard deviation.
    run_slsqp:
        If false, evaluate restart candidates only. This is useful for large
        real-data cases where high-dimensional SLSQP restarts are too expensive
        for the final experiment pipeline.
    """

    start_time = time.perf_counter()
    losses = _validate_scenario_losses(scenario_losses)
    total_alpha = _validate_total_alpha(alpha)
    probabilities = _validate_probabilities(scenario_probabilities, losses.shape[1])
    profiles = [_build_profile(loss_row, probabilities) for loss_row in losses]

    equal_result = solve_equal_allocation_cvar(
        scenario_losses=losses,
        alpha=total_alpha,
        scenario_probabilities=probabilities,
    )
    if np.allclose(losses, losses[0], atol=1e-12, rtol=1e-12):
        return _make_result(
            objective_value=equal_result.objective_value,
            alpha_allocation=equal_result.alpha_allocation,
            per_constraint_cvar=equal_result.per_constraint_cvar,
            runtime=time.perf_counter() - start_time,
            iterations=0,
            final_residual=0.0,
            solver_status="symmetric_equal_allocation",
        )
    if losses.shape[0] == 1:
        return _make_result(
            objective_value=equal_result.objective_value,
            alpha_allocation=equal_result.alpha_allocation,
            per_constraint_cvar=equal_result.per_constraint_cvar,
            runtime=time.perf_counter() - start_time,
            iterations=0,
            final_residual=0.0,
            solver_status="single_constraint",
        )

    def objective(alpha_allocation: np.ndarray) -> float:
        value, _, _ = _evaluate_allocation(profiles, alpha_allocation)
        return value

    def gradient(alpha_allocation: np.ndarray) -> np.ndarray:
        _, _, grad = _evaluate_allocation(profiles, alpha_allocation)
        return grad

    bounds = [(0.0, 1.0) for _ in range(losses.shape[0])]
    constraints = [
        {
            "type": "eq",
            "fun": lambda allocation: float(np.sum(allocation) - total_alpha),
            "jac": lambda allocation: np.ones_like(allocation),
        }
    ]

    best_result = equal_result
    best_objective = equal_result.objective_value
    best_iterations = 0
    best_status = "fallback_to_equal_allocation"

    def consider_candidate(
        alpha_allocation: np.ndarray,
        *,
        iterations: int,
        status: str,
    ) -> None:
        nonlocal best_objective, best_iterations, best_status, best_result

        candidate_allocation = np.asarray(alpha_allocation, dtype=float)
        if not np.all(np.isfinite(candidate_allocation)):
            return

        candidate_allocation = np.clip(candidate_allocation, 0.0, 1.0)
        allocation_sum = float(candidate_allocation.sum())
        if allocation_sum > 0.0:
            candidate_allocation = candidate_allocation * (total_alpha / allocation_sum)

        candidate_objective, candidate_cvar, _ = _evaluate_allocation(profiles, candidate_allocation)
        candidate_residual = abs(float(candidate_allocation.sum()) - total_alpha)

        if not math.isfinite(candidate_objective):
            return
        if candidate_residual > 1e-6:
            return

        if candidate_objective <= best_objective + 1e-10:
            best_objective = candidate_objective
            best_iterations = iterations
            best_status = status
            best_result = _make_result(
                objective_value=candidate_objective,
                alpha_allocation=candidate_allocation,
                per_constraint_cvar=candidate_cvar,
                runtime=0.0,
                iterations=best_iterations,
                final_residual=candidate_residual,
                solver_status=best_status,
            )

    focus_scores = np.std(losses, axis=1, ddof=0)
    for initial_allocation in _candidate_initial_allocations(
        losses.shape[0],
        total_alpha,
        max_focused_constraints=max_focused_constraints,
        focus_scores=focus_scores,
    ):
        consider_candidate(
            initial_allocation,
            iterations=0,
            status="restart_candidate",
        )

        if run_slsqp:
            solve_result = minimize(
                objective,
                initial_allocation,
                method="SLSQP",
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": tol, "maxiter": maxiter, "disp": False},
            )
            if bool(getattr(solve_result, "success", False)):
                consider_candidate(
                    solve_result.x,
                    iterations=int(getattr(solve_result, "nit", 0)),
                    status=str(getattr(solve_result, "message", "optimized")),
                )

    runtime = time.perf_counter() - start_time
    return _make_result(
        objective_value=best_result.objective_value,
        alpha_allocation=best_result.alpha_allocation,
        per_constraint_cvar=best_result.per_constraint_cvar,
        runtime=runtime,
        iterations=best_iterations,
        final_residual=abs(float(best_result.alpha_allocation.sum()) - total_alpha),
        solver_status=best_status,
    )


__all__ = [
    "CvarOptimizationDiagnostics",
    "FiniteScenarioCvarResult",
    "finite_scenario_cvar",
    "solve_equal_allocation_cvar",
    "solve_optimized_allocation_cvar",
]
