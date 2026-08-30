from __future__ import annotations

import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from risk_budget_jccp.real_data.cases.m5.build_instance import M5Instance
from risk_budget_jccp.real_data.common.certificates import cantelli_quantile, equal_allocation
from risk_budget_jccp.real_data.common.dca import logs_to_dicts, run_dca
from risk_budget_jccp.real_data.common.logging_utils import write_json
from risk_budget_jccp.real_data.common.metrics import alpha_metrics, relative_improvement, violation_metrics
from risk_budget_jccp.real_data.common.result_status import certificate_accept_tol, status_fields
from risk_budget_jccp.real_data.common.solvers import require_success, solve_problem
from risk_budget_jccp.algorithms.optimized_cvar import (
    finite_scenario_cvar,
    solve_optimized_allocation_cvar,
)


def _finite_cvar_and_derivative(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    values = np.sort(np.asarray(losses, dtype=float))[::-1]
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty vector")
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 1.0e-12:
        return float(values[0]), 0.0
    if a >= 1.0 - 1.0e-12:
        mean = float(np.mean(values))
        return mean, float(values[-1] - mean)
    probs = np.full(values.size, 1.0 / values.size, dtype=float)
    cumulative_prob = np.cumsum(probs)
    cumulative_loss = np.cumsum(probs * values)
    cumulative_prob[-1] = 1.0
    index = int(np.searchsorted(cumulative_prob, a, side="left"))
    left_prob = 0.0 if index == 0 else float(cumulative_prob[index - 1])
    left_loss = 0.0 if index == 0 else float(cumulative_loss[index - 1])
    threshold = float(values[index])
    cvar = (left_loss + (a - left_prob) * threshold) / a
    derivative = (threshold - cvar) / a
    return float(cvar), float(derivative)


def _cantelli_quantile_derivative(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    q = cantelli_quantile(np.clip(a, 1.0e-12, 1.0 - 1.0e-12))
    return -1.0 / (2.0 * np.maximum(a, 1.0e-12) ** 2 * np.maximum(q, 1.0e-12))


def _dual_vector(value: object, length: int) -> np.ndarray:
    if value is None:
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != length:
        return np.full(length, np.nan, dtype=float)
    return np.maximum(array, 0.0)


def _dual_scalar(value: object) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return float("nan")
    return max(float(array[0]), 0.0)


def _m5_cvar_diagnostics(instance: M5Instance, alpha: np.ndarray) -> dict[str, object]:
    alpha_arr = np.asarray(alpha, dtype=float)
    cvar = np.zeros(instance.m, dtype=float)
    derivative = np.zeros(instance.m, dtype=float)
    for index in range(instance.m):
        cvar[index], derivative[index] = _finite_cvar_and_derivative(instance.demand_train[:, index], float(alpha_arr[index]))
    marginal = -instance.cost * derivative
    return {
        "cvar_allocated_tail_value": cvar.tolist(),
        "cvar_tail_derivative": derivative.tolist(),
        "cvar_marginal_budget_value": marginal.tolist(),
        "theory_budget_driver": marginal.tolist(),
        "theory_driver_source": "finite_scenario_cvar_derivative",
    }


def _m5_cantelli_diagnostics(instance: M5Instance, alpha: np.ndarray) -> dict[str, object]:
    alpha_arr = np.asarray(alpha, dtype=float)
    sigma = instance.demand_train.std(axis=0, ddof=0)
    q_prime = _cantelli_quantile_derivative(alpha_arr)
    marginal = -instance.cost * sigma * q_prime
    return {
        "cantelli_quantile_derivative": q_prime.tolist(),
        "cantelli_marginal_budget_value": marginal.tolist(),
        "theory_budget_driver": marginal.tolist(),
        "theory_driver_source": "cantelli_closed_form_derivative",
    }


def _violation_scale(demand: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return one model-wide physical scale for numerical violation counting."""
    scale = max(float(np.max(np.abs(demand))), float(np.max(np.abs(np.asarray(x, dtype=float)))))
    return np.full(np.asarray(x, dtype=float).shape, scale, dtype=float)


def _evaluate(instance: M5Instance, x: np.ndarray) -> dict[str, float | int | list[int]]:
    x_arr = np.asarray(x, dtype=float)
    y_eval = instance.demand_test - x_arr[None, :]
    return violation_metrics(y_eval, reference_scale=_violation_scale(instance.demand_test, x_arr))


def _prefixed_metrics(prefix: str, metrics: dict[str, float | int | list[int]]) -> dict[str, float | int | list[int]]:
    return {
        f"{prefix}_joint_violation": float(metrics["empirical_joint_violation"]),
        f"{prefix}_average_scalar_violations": float(metrics["average_scalar_violations"]),
        f"{prefix}_max_scalar_violations": int(metrics["max_scalar_violations"]),
        f"{prefix}_violation_counts": metrics["violation_counts"],
        f"{prefix}_scalar_violation_rates": metrics["scalar_violation_rates"],
        f"{prefix}_violation_tolerance": metrics["violation_tolerance"],
        f"{prefix}_max_violation_tolerance": float(metrics["max_violation_tolerance"]),
    }


def _solution_payload(
    *,
    instance: M5Instance,
    certificate: str,
    allocation: str,
    x: np.ndarray,
    alpha_vec: np.ndarray,
    objective: float,
    runtime: float,
    solver_status: str,
    iterations: int,
    stationarity: float,
    feasibility: float,
    logs: list[dict[str, float | int | str]] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    x_arr = np.asarray(x, dtype=float)
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    calibration_metrics = violation_metrics(
        instance.demand_train - x_arr[None, :],
        reference_scale=_violation_scale(instance.demand_train, x_arr),
    )
    heldout_metrics = _evaluate(instance, x_arr)
    scalar_violation = np.maximum(instance.demand_train - x_arr[None, :], 0.0).max(initial=0.0)
    valid = feasibility <= 1.0e-4 and not str(solver_status).startswith("failed")
    fallback_used = "fallback" in str(solver_status)
    certificate_residual_max = max(0.0, float(feasibility))
    if extra and "certificate_values" in extra:
        certificate_residual_max = max(
            0.0,
            float(np.max(np.asarray(extra["certificate_values"], dtype=float), initial=0.0)),
        )
    budget_residual = max(0.0, float(alpha_arr.sum() - instance.alpha))
    payload: dict[str, object] = {
        "case": "m5",
        "certificate": certificate,
        "allocation": allocation,
        "objective": float(objective),
        "runtime": float(runtime),
        "solver_status": solver_status,
        "majorization_iterations": int(iterations),
        "stationarity_residual": float(stationarity),
        "feasibility_residual": float(feasibility),
        "certificate_residual_max": certificate_residual_max,
        "budget_residual": budget_residual,
        "sum_alpha": float(alpha_arr.sum()),
        "valid_certificate": bool(valid),
        "valid_optimization": bool(valid and allocation == "optimized" and not fallback_used),
        "failure_reason": "" if valid else str(solver_status),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(solver_status) if fallback_used else "",
        "alpha_vector": alpha_arr.tolist(),
        "x_variables": x_arr.tolist(),
        "stocking_quantities": x_arr.tolist(),
        "calibration_max_positive_shortage": float(scalar_violation),
        **_prefixed_metrics("calibration", calibration_metrics),
        **_prefixed_metrics("heldout", heldout_metrics),
        # Backward-compatible held-out aliases.
        **heldout_metrics,
        **alpha_metrics(alpha_arr),
    }
    if logs is not None:
        payload["majorization_logs"] = logs
    if extra:
        payload.update(extra)
    return payload


def _solve_cvar_equal(instance: M5Instance, alpha_vec: np.ndarray) -> dict[str, object]:
    n, m = instance.demand_train.shape
    x = cp.Variable(m, nonneg=True)
    t = cp.Variable(m, nonneg=True)
    u = cp.Variable((n, m), nonneg=True)
    constraints = [
        u >= instance.demand_train - cp.reshape(x, (1, m), order="C") + cp.reshape(t, (1, m), order="C"),
        cp.sum(u, axis=0) / n - cp.multiply(alpha_vec, t) <= 0.0,
    ]
    problem = cp.Problem(cp.Minimize(instance.cost @ x), constraints)
    diagnostics = solve_problem(problem)
    require_success(diagnostics)
    x_val = np.asarray(x.value, dtype=float).reshape(m)
    certificate_values = np.asarray(cp.sum(u, axis=0).value / n, dtype=float).reshape(m) - alpha_vec * np.asarray(t.value).reshape(m)
    residual = float(np.max(certificate_values))
    return _solution_payload(
        instance=instance,
        certificate="cvar",
        allocation="equal",
        x=x_val,
        alpha_vec=alpha_vec,
        objective=float(problem.value),
        runtime=diagnostics.runtime,
        solver_status=diagnostics.status,
        iterations=0,
        stationarity=max(0.0, residual),
        feasibility=max(0.0, residual),
        extra={
            **_m5_cvar_diagnostics(instance, alpha_vec),
            "certificate_values": certificate_values.tolist(),
        },
    )


def _solve_cvar_optimized(instance: M5Instance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    m = instance.m
    min_alpha = float(dca_cfg["min_alpha"])
    weighted_losses = instance.cost[:, None] * instance.demand_train.T
    optimized = solve_optimized_allocation_cvar(
        scenario_losses=weighted_losses,
        alpha=instance.alpha,
        max_focused_constraints=min(12, m),
        maxiter=200,
        tol=float(dca_cfg["tol"]),
    )
    alpha = np.asarray(optimized.alpha_allocation, dtype=float)
    if np.any(alpha < min_alpha):
        alpha = np.maximum(alpha, min_alpha)
        alpha *= instance.alpha / float(alpha.sum())
    x = np.array(
        [
            finite_scenario_cvar(instance.demand_train[:, index], alpha=float(alpha[index]))
            for index in range(m)
        ],
        dtype=float,
    )
    u = np.maximum(instance.demand_train - x[None, :], 0.0)
    cert = np.array(
        [
            finite_scenario_cvar(instance.demand_train[:, index] - x[index], alpha=float(alpha[index]))
            for index in range(m)
        ],
        dtype=float,
    )
    feasibility = max(0.0, float(np.max(cert)), float(alpha.sum() - instance.alpha))
    return _solution_payload(
        instance=instance,
        certificate="cvar",
        allocation="optimized",
        x=x,
        alpha_vec=alpha,
        objective=float(instance.cost @ x),
        runtime=time.perf_counter() - start,
        solver_status=f"separable_cvar_allocation:{optimized.diagnostics.solver_status}",
        iterations=int(optimized.diagnostics.iterations),
        stationarity=feasibility,
        feasibility=feasibility,
        logs=[
            {
                "iteration": int(optimized.diagnostics.iterations),
                "objective": float(instance.cost @ x),
                "max_certificate_violation": float(np.max(cert)),
                "risk_budget_used": float(alpha.sum()),
                "step_norm": 0.0,
                "stationarity_residual": feasibility,
                "solver_status": optimized.diagnostics.solver_status,
                "solver_runtime": float(optimized.diagnostics.runtime),
            }
        ],
        extra={
            **_m5_cvar_diagnostics(instance, alpha),
            "certificate_values": cert.tolist(),
        },
    )


def _solve_bernstein_equal(instance: M5Instance, alpha_vec: np.ndarray) -> dict[str, object]:
    m = instance.m
    mu = instance.demand_train.mean(axis=0)
    sigma = instance.demand_train.std(axis=0, ddof=0)
    x = cp.Variable(m, nonneg=True)
    theta = cp.Variable(m)
    constraints = [
        theta >= 1.0e-8,
    ]
    cert_expr = mu - x + cp.multiply(0.5 * sigma**2, cp.inv_pos(theta)) - cp.multiply(np.log(alpha_vec), theta)
    certificate_constraint = cert_expr <= 0.0
    constraints.append(certificate_constraint)
    problem = cp.Problem(cp.Minimize(instance.cost @ x), constraints)
    diagnostics = solve_problem(problem)
    require_success(diagnostics)
    x_val = np.asarray(x.value, dtype=float).reshape(m)
    theta_val = np.asarray(theta.value, dtype=float).reshape(m)
    cert = mu - x_val + 0.5 * sigma**2 / theta_val - theta_val * np.log(alpha_vec)
    certificate_dual = _dual_vector(certificate_constraint.dual_value, m)
    return _solution_payload(
        instance=instance,
        certificate="bernstein",
        allocation="equal",
        x=x_val,
        alpha_vec=alpha_vec,
        objective=float(instance.cost @ x_val),
        runtime=diagnostics.runtime,
        solver_status=diagnostics.status,
        iterations=0,
        stationarity=max(0.0, float(np.max(cert))),
        feasibility=max(0.0, float(np.max(cert))),
        extra={
            "theta_variables": theta_val.tolist(),
            "certificate_values": cert.tolist(),
            "certificate_dual_values": certificate_dual.tolist(),
            "theory_budget_driver": (certificate_dual * theta_val).tolist(),
            "kkt_dual_source": "m5_bernstein_equal_convex_problem",
            "theory_driver_source": "bernstein_kkt_lambda_theta",
        },
    )


def _solve_bernstein_optimized(instance: M5Instance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    m = instance.m
    mu = instance.demand_train.mean(axis=0)
    sigma = instance.demand_train.std(axis=0, ddof=0)
    min_alpha = float(dca_cfg["min_alpha"])
    min_theta = 1.0e-8
    alpha_total = instance.alpha
    x_equal = np.asarray(equal_payload["x_variables"], dtype=float)
    alpha_equal = np.asarray(equal_payload["alpha_vector"], dtype=float)
    theta_initial = np.maximum(sigma / np.maximum(x_equal - mu, 1.0e-3), min_theta)

    def solve_majorized(previous: dict[str, np.ndarray], iteration: int):
        theta_prev = np.maximum(previous["theta"], min_theta)
        alpha_prev = np.maximum(previous["alpha"], min_alpha)
        x = cp.Variable(m, nonneg=True)
        theta = cp.Variable(m)
        alpha = cp.Variable(m)
        theta_log_lin = cp.multiply(1.0 + np.log(theta_prev), theta) - theta_prev
        cert = (
            mu
            - x
            + cp.multiply(0.5 * sigma**2, cp.inv_pos(theta))
            + cp.rel_entr(theta, alpha)
            - theta_log_lin
        )
        certificate_constraint = cert <= 0.0
        alpha_lower_constraint = alpha >= min_alpha
        budget_constraint = cp.sum(alpha) <= alpha_total
        theta_lower_constraint = theta >= min_theta
        problem = cp.Problem(
            cp.Minimize(instance.cost @ x + float(dca_cfg["proximal_weight"]) * cp.sum_squares(alpha - alpha_prev)),
            [certificate_constraint, alpha_lower_constraint, budget_constraint, theta_lower_constraint],
        )
        diagnostics = solve_problem(problem)
        require_success(diagnostics)
        values = {
            "x": np.asarray(x.value, dtype=float).reshape(m),
            "theta": np.asarray(theta.value, dtype=float).reshape(m),
            "alpha": np.asarray(alpha.value, dtype=float).reshape(m),
        }
        duals = {
            "certificate_dual_values": _dual_vector(certificate_constraint.dual_value, m),
            "alpha_lower_duals": _dual_vector(alpha_lower_constraint.dual_value, m),
            "budget_constraint_dual": _dual_scalar(budget_constraint.dual_value),
            "theta_lower_duals": _dual_vector(theta_lower_constraint.dual_value, m),
            "kkt_dual_source": f"m5_bernstein_dca_majorized_iteration_{iteration}",
        }
        return values, float(instance.cost @ values["x"]), diagnostics.status, diagnostics.runtime, duals

    def residual(values: dict[str, np.ndarray | float], previous: dict[str, np.ndarray]):
        x_val = np.asarray(values["x"], dtype=float)
        theta_val = np.maximum(np.asarray(values["theta"], dtype=float), min_theta)
        alpha_val = np.maximum(np.asarray(values["alpha"], dtype=float), min_alpha)
        cert = mu - x_val + 0.5 * sigma**2 / theta_val - theta_val * np.log(alpha_val)
        budget = max(0.0, float(alpha_val.sum() - alpha_total))
        return max(budget, float(np.max(cert))), float(np.max(cert))

    result = run_dca(
        initial={"x": x_equal, "theta": theta_initial, "alpha": alpha_equal},
        solve_majorized=solve_majorized,
        residual=residual,
        vectorize=lambda values: np.concatenate(
            [
                np.asarray(values["x"], dtype=float),
                np.asarray(values["theta"], dtype=float),
                np.asarray(values["alpha"], dtype=float),
            ]
        ),
        max_iter=int(dca_cfg["max_iter"]),
        tol=float(dca_cfg["tol"]),
        alpha_total=alpha_total,
    )
    accept_tol = certificate_accept_tol(dca_cfg)
    if result.feasibility_residual > accept_tol:
        return _solution_payload(
            instance=instance,
            certificate="bernstein",
            allocation="optimized",
            x=x_equal,
            alpha_vec=alpha_equal,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            solver_status=f"bernstein_optimized_fallback_equal_feasible:dca_candidate_feasibility={result.feasibility_residual:.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "theta_variables": list(equal_payload.get("theta_variables", [])),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_dual_values": list(equal_payload.get("certificate_dual_values", [])),
                "theory_budget_driver": list(equal_payload.get("theory_budget_driver", [])),
                "kkt_dual_source": "equal_solution_inherited_by_feasibility_fallback",
                "alpha_lower_bound": min_alpha,
                "attempted_certificate_dual_values": np.asarray(
                    result.diagnostics.get("certificate_dual_values", np.full(m, np.nan)),
                    dtype=float,
                ).tolist(),
                "attempted_budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
                "theory_driver_source": "bernstein_kkt_lambda_theta_fallback_diagnostic",
            },
        ) | {
            "attempted_optimized_objective": float(result.objective),
            "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
        }
    if float(result.objective) >= float(equal_payload["objective"]) - 1.0e-7 * (1.0 + abs(float(equal_payload["objective"]))):
        return _solution_payload(
            instance=instance,
            certificate="bernstein",
            allocation="optimized",
            x=x_equal,
            alpha_vec=alpha_equal,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            solver_status=f"bernstein_optimized_fallback_equal_objective_dominates:dca_candidate_objective={float(result.objective):.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "theta_variables": list(equal_payload.get("theta_variables", [])),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_dual_values": list(equal_payload.get("certificate_dual_values", [])),
                "theory_budget_driver": list(equal_payload.get("theory_budget_driver", [])),
                "kkt_dual_source": "equal_solution_inherited_by_objective_fallback",
                "alpha_lower_bound": min_alpha,
                "attempted_certificate_dual_values": np.asarray(
                    result.diagnostics.get("certificate_dual_values", np.full(m, np.nan)),
                    dtype=float,
                ).tolist(),
                "attempted_budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
                "theory_driver_source": "bernstein_kkt_lambda_theta_fallback_diagnostic",
            },
        ) | {
            "attempted_optimized_objective": float(result.objective),
            "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
        }
    theta_value = np.asarray(result.values["theta"], dtype=float)
    alpha_value = np.asarray(result.values["alpha"], dtype=float)
    cert_value = mu - np.asarray(result.values["x"], dtype=float) + 0.5 * sigma**2 / np.maximum(theta_value, min_theta) - np.maximum(theta_value, min_theta) * np.log(np.maximum(alpha_value, min_alpha))
    certificate_dual = np.asarray(result.diagnostics.get("certificate_dual_values", np.full(m, np.nan)), dtype=float)
    return _solution_payload(
        instance=instance,
        certificate="bernstein",
        allocation="optimized",
        x=np.asarray(result.values["x"], dtype=float),
        alpha_vec=alpha_value,
        objective=result.objective,
        runtime=result.runtime,
        solver_status=result.status,
        iterations=result.iterations,
        stationarity=result.stationarity_residual,
        feasibility=result.feasibility_residual,
        logs=logs_to_dicts(result.logs),
        extra={
            "theta_variables": theta_value.tolist(),
            "certificate_values": cert_value.tolist(),
            "certificate_dual_values": certificate_dual.tolist(),
            "alpha_lower_duals": np.asarray(result.diagnostics.get("alpha_lower_duals", np.full(m, np.nan)), dtype=float).tolist(),
            "budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
            "theta_lower_duals": np.asarray(result.diagnostics.get("theta_lower_duals", np.full(m, np.nan)), dtype=float).tolist(),
            "alpha_lower_bound": min_alpha,
            "theory_budget_driver": (certificate_dual * theta_value).tolist(),
            "kkt_dual_source": str(result.diagnostics.get("kkt_dual_source", "m5_bernstein_dca_last_majorized_subproblem")),
            "theory_driver_source": "bernstein_kkt_lambda_theta",
        },
    )


def _solve_cantelli_equal(instance: M5Instance, alpha_vec: np.ndarray) -> dict[str, object]:
    start = time.perf_counter()
    mu = instance.demand_train.mean(axis=0)
    sigma = instance.demand_train.std(axis=0, ddof=0)
    x = np.maximum(mu + sigma * cantelli_quantile(alpha_vec), 0.0)
    runtime = time.perf_counter() - start
    cert = mu - x + sigma * cantelli_quantile(alpha_vec)
    return _solution_payload(
        instance=instance,
        certificate="cantelli",
        allocation="equal",
        x=x,
        alpha_vec=alpha_vec,
        objective=float(instance.cost @ x),
        runtime=runtime,
        solver_status="closed_form_fixed_alpha",
        iterations=0,
        stationarity=max(0.0, float(np.max(cert))),
        feasibility=max(0.0, float(np.max(cert))),
        extra={
            **_m5_cantelli_diagnostics(instance, alpha_vec),
            "certificate_values": cert.tolist(),
        },
    )


def _solve_cantelli_optimized(instance: M5Instance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    mu = instance.demand_train.mean(axis=0)
    sigma = instance.demand_train.std(axis=0, ddof=0)
    min_alpha = float(dca_cfg["min_alpha"])

    upper = min(1.0 - 1.0e-9, instance.alpha)

    def coordinate_alpha(lam: float, index: int) -> float:
        if sigma[index] <= 1.0e-12 or instance.cost[index] <= 0.0:
            return min_alpha
        result = minimize_scalar(
            lambda a: float(instance.cost[index] * max(mu[index] + sigma[index] * cantelli_quantile(a), 0.0) + lam * a),
            bounds=(min_alpha, upper),
            method="bounded",
            options={"xatol": max(float(dca_cfg["tol"]) * 0.1, 1.0e-10)},
        )
        return float(np.clip(result.x, min_alpha, upper))

    low, high = 0.0, 1.0
    for _ in range(80):
        alpha_probe = np.array([coordinate_alpha(high, idx) for idx in range(instance.m)], dtype=float)
        if float(alpha_probe.sum()) <= instance.alpha:
            break
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        alpha_mid = np.array([coordinate_alpha(mid, idx) for idx in range(instance.m)], dtype=float)
        if float(alpha_mid.sum()) > instance.alpha:
            low = mid
        else:
            high = mid
    alpha = np.array([coordinate_alpha(high, idx) for idx in range(instance.m)], dtype=float)
    slack = instance.alpha - float(alpha.sum())
    if slack > 0.0:
        order = np.argsort(-instance.cost * sigma)
        for idx in order:
            add = min(slack, upper - alpha[idx])
            if add <= 0.0:
                continue
            alpha[idx] += add
            slack -= add
            if slack <= 1.0e-10:
                break
    x = np.maximum(mu + sigma * cantelli_quantile(alpha), 0.0)
    cert = mu - x + sigma * cantelli_quantile(alpha)
    status = "separable_cantelli_budget_bisection"
    return _solution_payload(
        instance=instance,
        certificate="cantelli",
        allocation="optimized",
        x=x,
        alpha_vec=alpha,
        objective=float(instance.cost @ x),
        runtime=time.perf_counter() - start,
        solver_status=status,
        iterations=0,
        stationarity=max(0.0, float(np.max(cert))),
        feasibility=max(max(0.0, float(np.max(cert))), max(0.0, float(alpha.sum() - instance.alpha))),
        extra={
            **_m5_cantelli_diagnostics(instance, alpha),
            "certificate_values": cert.tolist(),
        },
    )


def solve_m5_instance(instance: M5Instance, results_dir: str | Path, dca_cfg: dict[str, object]) -> pd.DataFrame:
    root = Path(results_dir)
    solution_dir = root / "solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    alpha_equal = equal_allocation(instance.m, instance.alpha, min_alpha=float(dca_cfg["min_alpha"]))
    payloads: list[dict[str, object]] = []

    cvar_equal = _solve_cvar_equal(instance, alpha_equal)
    payloads.append(cvar_equal)
    payloads.append(_solve_cvar_optimized(instance, cvar_equal, dca_cfg))

    bernstein_equal = _solve_bernstein_equal(instance, alpha_equal)
    payloads.append(bernstein_equal)
    payloads.append(_solve_bernstein_optimized(instance, bernstein_equal, dca_cfg))

    cantelli_equal = _solve_cantelli_equal(instance, alpha_equal)
    payloads.append(cantelli_equal)
    payloads.append(_solve_cantelli_optimized(instance, cantelli_equal, dca_cfg))

    equal_objectives = {
        str(payload["certificate"]): float(payload["objective"])
        for payload in payloads
        if payload["allocation"] == "equal"
    }
    rows: list[dict[str, object]] = []
    for payload in payloads:
        certificate = str(payload["certificate"])
        allocation = str(payload["allocation"])
        payload.update(
            status_fields(
                case="m5",
                certificate=certificate,
                allocation=allocation,
                solver_status=str(payload["solver_status"]),
                valid_certificate=bool(payload["valid_certificate"]),
                valid_optimization=bool(payload["valid_optimization"]),
                fallback_used=bool(payload.get("fallback_used", False)),
                feasibility_residual=float(payload["feasibility_residual"]),
                calibration_joint_violation=float(payload["calibration_joint_violation"]),
                alpha_total=instance.alpha,
                dca_cfg=dca_cfg,
            )
        )
        write_json(solution_dir / f"m5_{certificate}_{allocation}.json", payload)
        rows.append(
            {
                "case": "m5",
                "certificate": certificate,
                "allocation": allocation,
                "objective": float(payload["objective"]),
                "relative_improvement": relative_improvement(equal_objectives[certificate], float(payload["objective"])),
                "calibration_joint_violation": float(payload["calibration_joint_violation"]),
                "heldout_joint_violation": float(payload["heldout_joint_violation"]),
                "calibration_average_scalar_violations": float(payload["calibration_average_scalar_violations"]),
                "heldout_average_scalar_violations": float(payload["heldout_average_scalar_violations"]),
                "calibration_max_scalar_violations": int(payload["calibration_max_scalar_violations"]),
                "heldout_max_scalar_violations": int(payload["heldout_max_scalar_violations"]),
                "empirical_joint_violation": float(payload["empirical_joint_violation"]),
                "average_scalar_violations": float(payload["average_scalar_violations"]),
                "max_scalar_violations": int(payload["max_scalar_violations"]),
                "runtime": float(payload["runtime"]),
                "majorization_iterations": int(payload["majorization_iterations"]),
                "stationarity_residual": float(payload["stationarity_residual"]),
                "feasibility_residual": float(payload["feasibility_residual"]),
                "sum_alpha": float(np.sum(np.asarray(payload["alpha_vector"], dtype=float))),
                "certificate_residual_max": float(payload["certificate_residual_max"]),
                "budget_residual": float(payload["budget_residual"]),
                "calibration_max_violation_tolerance": float(payload["calibration_max_violation_tolerance"]),
                "heldout_max_violation_tolerance": float(payload["heldout_max_violation_tolerance"]),
                "max_budget_share": float(payload["max_budget_share"]),
                "normalized_entropy": float(payload["normalized_entropy"]),
                "valid_certificate": bool(payload["valid_certificate"]),
                "valid_optimization": bool(payload["valid_optimization"]),
                "fallback_used": bool(payload.get("fallback_used", False)),
                "fallback_reason": str(payload.get("fallback_reason", "")),
                "failure_reason": str(payload["failure_reason"]),
                "algorithm_class": str(payload["algorithm_class"]),
                "certificate_accept_tol": float(payload["certificate_accept_tol"]),
                "certificate_strict_tol": float(payload["certificate_strict_tol"]),
                "passes_certificate_acceptance": bool(payload["passes_certificate_acceptance"]),
                "passes_certificate_strict": bool(payload["passes_certificate_strict"]),
                "certificate_acceptance_status": str(payload["certificate_acceptance_status"]),
                "calibration_jvp_bound_applies": bool(payload["calibration_jvp_bound_applies"]),
                "calibration_jvp_within_budget": bool(payload["calibration_jvp_within_budget"]),
                "calibration_jvp_contract": str(payload["calibration_jvp_contract"]),
                "result_status": str(payload["result_status"]),
                "solver_status": str(payload["solver_status"]),
            }
        )
    return pd.DataFrame(rows)
