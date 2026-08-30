from __future__ import annotations

import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.cases.power.build_instance import PowerInstance
from risk_budget_jccp.real_data.common.certificates import cantelli_quantile, equal_allocation, validate_cantelli_budget
from risk_budget_jccp.real_data.common.dca import logs_to_dicts, run_dca
from risk_budget_jccp.real_data.common.logging_utils import write_json
from risk_budget_jccp.real_data.common.metrics import alpha_metrics, relative_improvement, violation_metrics
from risk_budget_jccp.real_data.common.result_status import certificate_accept_tol, status_fields
from risk_budget_jccp.real_data.common.solvers import require_success, solve_problem


def _dims(instance: PowerInstance) -> tuple[int, int, int, int]:
    snapshots = int(instance.forecast_flow_offset.shape[0])
    lines = int(instance.dispatch_flow_matrix.shape[0])
    generators = int(len(instance.generator_ids))
    side = 2 * lines
    if instance.m != snapshots * side:
        raise ValueError("PowerInstance scalar constraint count must equal snapshots * 2 * selected lines")
    return snapshots, lines, generators, side


def _snapshot_demands(instance: PowerInstance) -> np.ndarray:
    indices = np.asarray(instance.selected_test_indices, dtype=int)
    return instance.load_forecast_total[indices] - instance.renewable_forecast_total[indices]


def _base_dispatch(instance: PowerInstance) -> np.ndarray:
    snapshots, _, generators, _ = _dims(instance)
    demands = _snapshot_demands(instance)
    result = np.zeros((snapshots, generators), dtype=float)
    order = np.argsort(instance.generator_cost)
    for snapshot in range(snapshots):
        remaining = float(demands[snapshot])
        p = np.asarray(instance.generator_pmin, dtype=float).copy()
        remaining -= float(p.sum())
        for generator in order:
            if remaining <= 0.0:
                break
            extra = min(float(instance.generator_pmax[generator] - p[generator]), remaining)
            if extra > 0.0:
                p[generator] += extra
                remaining -= extra
        if abs(remaining) > 1.0e-6:
            raise RuntimeError("base dispatch cannot meet forecast demand within generator bounds")
        result[snapshot] = p
    return result


def _base_constraints(instance: PowerInstance, p: cp.Variable) -> list[cp.Constraint]:
    snapshots, _, _, _ = _dims(instance)
    demands = _snapshot_demands(instance)
    constraints: list[cp.Constraint] = [
        p >= instance.generator_pmin[None, :],
        p <= instance.generator_pmax[None, :],
    ]
    for snapshot in range(snapshots):
        constraints.append(cp.sum(p[snapshot, :]) == float(demands[snapshot]))
    return constraints


def _mean_generation_cost(instance: PowerInstance, p: cp.Expression) -> cp.Expression:
    snapshots, _, _, _ = _dims(instance)
    return cp.sum(p @ instance.generator_cost) / snapshots


def _flow_expr(instance: PowerInstance, p: cp.Expression, snapshot: int) -> cp.Expression:
    return instance.dispatch_flow_matrix @ p[snapshot, :] + instance.forecast_flow_offset[snapshot]


def _positive_limits(values: np.ndarray) -> np.ndarray:
    limits = np.asarray(values, dtype=float)
    if np.any(limits <= 0.0):
        raise ValueError("Power line-flow limits must be strictly positive")
    return limits


def _y_expr(instance: PowerInstance, p: cp.Expression, residuals: np.ndarray) -> cp.Expression:
    """Normalized scalar overload Y = (scenario flow - risk limit) / risk limit."""
    snapshots, lines, _, side = _dims(instance)
    limits = _positive_limits(instance.line_limits)
    blocks: list[cp.Expression] = []
    for snapshot in range(snapshots):
        start = snapshot * side
        flow = _flow_expr(instance, p, snapshot)
        flow_row = cp.reshape(flow, (1, lines), order="C")
        pos_limits = limits[start : start + lines]
        neg_limits = limits[start + lines : start + side]
        pos_raw = flow_row + residuals[:, start : start + lines] - pos_limits[None, :]
        neg_raw = -flow_row + residuals[:, start + lines : start + side] - neg_limits[None, :]
        pos = cp.multiply(1.0 / pos_limits[None, :], pos_raw)
        neg = cp.multiply(1.0 / neg_limits[None, :], neg_raw)
        blocks.append(cp.hstack([pos, neg]))
    return cp.hstack(blocks)


def _y_values(instance: PowerInstance, p_value: np.ndarray, residuals: np.ndarray, *, limit_name: str = "line_limits") -> np.ndarray:
    """Normalized scalar overload values; positive entries indicate violations."""
    snapshots, lines, _, side = _dims(instance)
    limits = _positive_limits(np.asarray(getattr(instance, limit_name), dtype=float))
    if limits.size != instance.m:
        limits = _positive_limits(np.asarray(instance.line_limits, dtype=float))
    blocks: list[np.ndarray] = []
    for snapshot in range(snapshots):
        start = snapshot * side
        flow = instance.dispatch_flow_matrix @ p_value[snapshot] + instance.forecast_flow_offset[snapshot]
        pos_limits = limits[start : start + lines]
        neg_limits = limits[start + lines : start + side]
        pos = (flow[None, :] + residuals[:, start : start + lines] - pos_limits[None, :]) / pos_limits[None, :]
        neg = (-flow[None, :] + residuals[:, start + lines : start + side] - neg_limits[None, :]) / neg_limits[None, :]
        blocks.append(np.hstack([pos, neg]))
    return np.hstack(blocks)


def _limits_for(instance: PowerInstance, name: str) -> np.ndarray:
    values = np.asarray(getattr(instance, name), dtype=float)
    if values.size == instance.m:
        return values
    return np.asarray(instance.line_limits, dtype=float)


def _evaluation_metrics(instance: PowerInstance, p_value: np.ndarray) -> dict[str, object]:
    normalized_scale = np.ones(instance.m, dtype=float)
    cal = violation_metrics(
        _y_values(instance, p_value, instance.flow_residual_train),
        reference_scale=normalized_scale,
    )
    held = violation_metrics(
        _y_values(instance, p_value, instance.flow_residual_test),
        reference_scale=normalized_scale,
    )
    cal_cont = violation_metrics(
        _y_values(instance, p_value, instance.flow_residual_train, limit_name="cont_limits"),
        reference_scale=normalized_scale,
    )
    held_cont = violation_metrics(
        _y_values(instance, p_value, instance.flow_residual_test, limit_name="cont_limits"),
        reference_scale=normalized_scale,
    )
    return {
        "calibration_emergency_joint_violation": float(cal["empirical_joint_violation"]),
        "heldout_emergency_joint_violation": float(held["empirical_joint_violation"]),
        "calibration_cont_overload_rate": float(cal_cont["empirical_joint_violation"]),
        "heldout_cont_overload_rate": float(held_cont["empirical_joint_violation"]),
        "calibration_average_scalar_violations": float(cal["average_scalar_violations"]),
        "heldout_average_scalar_violations": float(held["average_scalar_violations"]),
        "calibration_max_scalar_violations": int(cal["max_scalar_violations"]),
        "heldout_max_scalar_violations": int(held["max_scalar_violations"]),
        "calibration_violation_counts": cal["violation_counts"],
        "heldout_violation_counts": held["violation_counts"],
        "calibration_scalar_violation_rates": cal["scalar_violation_rates"],
        "heldout_scalar_violation_rates": held["scalar_violation_rates"],
        "calibration_violation_tolerance": cal["violation_tolerance"],
        "heldout_violation_tolerance": held["violation_tolerance"],
        "calibration_max_violation_tolerance": float(cal["max_violation_tolerance"]),
        "heldout_max_violation_tolerance": float(held["max_violation_tolerance"]),
        "violation_counts": held["violation_counts"],
        "empirical_joint_violation": float(held["empirical_joint_violation"]),
        "average_scalar_violations": float(held["average_scalar_violations"]),
        "max_scalar_violations": int(held["max_scalar_violations"]),
    }


def _residual_stats(instance: PowerInstance) -> tuple[np.ndarray, np.ndarray]:
    limits = _positive_limits(instance.line_limits)
    normalized = instance.flow_residual_train / limits[None, :]
    return normalized.mean(axis=0), normalized.std(axis=0, ddof=0)


def _affine_mean_terms(instance: PowerInstance, p: cp.Expression, residual_mean: np.ndarray) -> cp.Expression:
    """Normalized affine mean of Y_i(p, xi)."""
    snapshots, lines, _, side = _dims(instance)
    limits = _positive_limits(instance.line_limits)
    terms: list[cp.Expression] = []
    for snapshot in range(snapshots):
        start = snapshot * side
        flow = _flow_expr(instance, p, snapshot)
        pos_limits = limits[start : start + lines]
        neg_limits = limits[start + lines : start + side]
        pos = cp.multiply(1.0 / pos_limits, flow - pos_limits) + residual_mean[start : start + lines]
        neg = cp.multiply(1.0 / neg_limits, -flow - neg_limits) + residual_mean[start + lines : start + side]
        terms.extend([pos, neg])
    return cp.hstack(terms)


def _affine_mean_values(instance: PowerInstance, p_value: np.ndarray, residual_mean: np.ndarray) -> np.ndarray:
    snapshots, lines, _, side = _dims(instance)
    limits = _positive_limits(instance.line_limits)
    terms: list[np.ndarray] = []
    for snapshot in range(snapshots):
        start = snapshot * side
        flow = instance.dispatch_flow_matrix @ p_value[snapshot] + instance.forecast_flow_offset[snapshot]
        pos_limits = limits[start : start + lines]
        neg_limits = limits[start + lines : start + side]
        pos = (flow - pos_limits) / pos_limits + residual_mean[start : start + lines]
        neg = (-flow - neg_limits) / neg_limits + residual_mean[start + lines : start + side]
        terms.extend([pos, neg])
    return np.concatenate(terms)


def _objective_value(instance: PowerInstance, p_value: np.ndarray) -> float:
    return float(np.mean(p_value @ instance.generator_cost))


def _residual_components(certificate_values: np.ndarray | None, alpha: np.ndarray, alpha_total: float) -> tuple[float, float, float]:
    certificate_residual = 0.0
    if certificate_values is not None and len(certificate_values) > 0:
        certificate_residual = max(0.0, float(np.max(np.asarray(certificate_values, dtype=float))))
    budget_residual = max(0.0, float(np.sum(np.asarray(alpha, dtype=float)) - alpha_total))
    return certificate_residual, budget_residual, max(certificate_residual, budget_residual)


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


def _payload(
    instance: PowerInstance,
    *,
    certificate: str,
    allocation: str,
    p_value: np.ndarray,
    alpha: np.ndarray,
    objective: float,
    runtime: float,
    status: str,
    iterations: int,
    stationarity: float,
    feasibility: float,
    logs: list[dict[str, float | int | str]] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    alpha = np.asarray(alpha, dtype=float)
    p_value = np.asarray(p_value, dtype=float)
    certificate_residual = float(feasibility)
    budget_residual = max(0.0, float(alpha.sum() - instance.alpha))
    if extra:
        certificate_residual = float(extra.get("certificate_residual_max", certificate_residual))
        budget_residual = float(extra.get("budget_residual", budget_residual))
    valid = feasibility <= 1.0e-4 and not str(status).startswith(("failed", "unsupported"))
    fallback_used = "fallback" in str(status)
    abs_base = np.abs(np.asarray(instance.base_flow, dtype=float))
    result: dict[str, object] = {
        "case": "power",
        "certificate": certificate,
        "allocation": allocation,
        "objective": float(objective),
        "runtime": float(runtime),
        "solver_status": status,
        "majorization_iterations": int(iterations),
        "stationarity_residual": float(stationarity),
        "feasibility_residual": float(feasibility),
        "certificate_residual_max": float(certificate_residual),
        "budget_residual": float(budget_residual),
        "normalized_certificate_residual_max": float(certificate_residual),
        "certificate_scale": "normalized_overload_fraction",
        "valid_certificate": bool(valid),
        "valid_optimization": bool(valid and allocation == "optimized" and not fallback_used),
        "failure_reason": "" if valid else str(status),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(status) if fallback_used else "",
        "primary_rating": instance.primary_rating,
        "alpha_vector": alpha.tolist(),
        "sum_alpha": float(alpha.sum()),
        "x_variables": p_value.mean(axis=0).tolist(),
        "dispatch_mw": p_value.tolist(),
        "mean_lte_multiplier": float(np.mean(_limits_for(instance, "lte_limits") / _limits_for(instance, "cont_limits"))),
        "mean_ste_multiplier": float(np.mean(_limits_for(instance, "ste_limits") / _limits_for(instance, "cont_limits"))),
        "n_nominal_cont_overloaded": int(np.sum(abs_base > _limits_for(instance, "cont_limits"))),
        "n_nominal_lte_overloaded": int(np.sum(abs_base > _limits_for(instance, "lte_limits"))),
        **_evaluation_metrics(instance, p_value),
        **alpha_metrics(alpha),
    }
    if logs is not None:
        result["majorization_logs"] = logs
    if extra:
        result.update(extra)
    return result


def _failed_payload(instance: PowerInstance, certificate: str, allocation: str, alpha: np.ndarray, status: str, runtime: float) -> dict[str, object]:
    p0 = _base_dispatch(instance)
    return _payload(
        instance,
        certificate=certificate,
        allocation=allocation,
        p_value=p0,
        alpha=alpha,
        objective=_objective_value(instance, p0),
        runtime=runtime,
        status=f"failed_{status}",
        iterations=0,
        stationarity=float("inf"),
        feasibility=float("inf"),
    )


def _solve_cvar_equal(instance: PowerInstance, alpha: np.ndarray) -> dict[str, object]:
    start = time.perf_counter()
    n = instance.flow_residual_train.shape[0]
    p = cp.Variable((instance.forecast_flow_offset.shape[0], len(instance.generator_ids)))
    t = cp.Variable(instance.m, nonneg=True)
    u = cp.Variable((n, instance.m), nonneg=True)
    y = _y_expr(instance, p, instance.flow_residual_train)
    constraints = _base_constraints(instance, p) + [
        u >= y + cp.reshape(t, (1, instance.m), order="C"),
        cp.sum(u, axis=0) / n - cp.multiply(alpha, t) <= 0.0,
    ]
    problem = cp.Problem(cp.Minimize(_mean_generation_cost(instance, p)), constraints)
    diag = solve_problem(problem)
    if diag.status not in {"optimal", "optimal_inaccurate"}:
        return _failed_payload(instance, "cvar", "equal", alpha, diag.status, time.perf_counter() - start)
    p_value = np.asarray(p.value, dtype=float)
    cert_value = np.sum(np.asarray(u.value), axis=0) / n - alpha * np.asarray(t.value)
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha, instance.alpha)
    return _payload(
        instance,
        certificate="cvar",
        allocation="equal",
        p_value=p_value,
        alpha=alpha,
        objective=_objective_value(instance, p_value),
        runtime=diag.runtime,
        status=diag.status,
        iterations=0,
        stationarity=feasibility,
        feasibility=feasibility,
        extra={
            "t_variables": np.asarray(t.value, dtype=float).tolist(),
            "certificate_values": cert_value.tolist(),
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def _solve_cvar_optimized(instance: PowerInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    n = instance.flow_residual_train.shape[0]
    min_alpha = float(dca_cfg["min_alpha"])
    accept_tol = certificate_accept_tol(dca_cfg)
    alpha_total = instance.alpha
    p0 = np.asarray(equal_payload["dispatch_mw"], dtype=float)
    alpha0 = np.asarray(equal_payload["alpha_vector"], dtype=float)
    t0 = np.maximum(np.asarray(equal_payload.get("t_variables", np.ones(instance.m)), dtype=float), 1.0e-6)
    if not bool(equal_payload.get("valid_certificate", False)):
        return _failed_payload(instance, "cvar", "optimized", alpha0, "equal_warm_start_invalid", 0.0)
    base_objective = _objective_value(instance, _base_dispatch(instance))
    if bool(equal_payload.get("valid_certificate", False)) and float(equal_payload["objective"]) <= base_objective + 1.0e-7 * (1.0 + abs(base_objective)):
        return _payload(
            instance,
            certificate="cvar",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=0.0,
            status="optimal_unconstrained_lower_bound",
            iterations=0,
            stationarity=0.0,
            feasibility=float(equal_payload["feasibility_residual"]),
            extra={
                "t_variables": np.asarray(equal_payload.get("t_variables", t0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "optimization_note": "equal allocation already attains the unconstrained economic dispatch lower bound",
            },
        )

    def solve_majorized(previous: dict[str, np.ndarray], iteration: int):
        p = cp.Variable(p0.shape)
        t = cp.Variable(instance.m, nonneg=True)
        u = cp.Variable((n, instance.m), nonneg=True)
        alpha = cp.Variable(instance.m)
        y = _y_expr(instance, p, instance.flow_residual_train)
        z_prev = np.maximum(previous["t"] + previous["alpha"], min_alpha)
        concave_majorant = 0.25 * z_prev**2 + cp.multiply(0.5 * z_prev, t + alpha - z_prev)
        cert = cp.sum(u, axis=0) / n + 0.25 * cp.square(t - alpha) - concave_majorant
        problem = cp.Problem(
            cp.Minimize(
                _mean_generation_cost(instance, p)
                + float(dca_cfg["proximal_weight"]) * cp.sum_squares(alpha - previous["alpha"])
            ),
            _base_constraints(instance, p)
            + [
                u >= y + cp.reshape(t, (1, instance.m), order="C"),
                cert <= 0.0,
                alpha >= min_alpha,
                cp.sum(alpha) <= alpha_total,
            ],
        )
        diag = solve_problem(problem)
        require_success(diag)
        values = {
            "p": np.asarray(p.value, dtype=float),
            "t": np.asarray(t.value, dtype=float),
            "alpha": np.asarray(alpha.value, dtype=float),
        }
        return values, _objective_value(instance, values["p"]), diag.status, diag.runtime

    def residual(values: dict[str, np.ndarray | float], previous: dict[str, np.ndarray]):
        p_value = np.asarray(values["p"], dtype=float)
        t_value = np.asarray(values["t"], dtype=float)
        alpha_value = np.maximum(np.asarray(values["alpha"], dtype=float), min_alpha)
        y_value = _y_values(instance, p_value, instance.flow_residual_train)
        u_value = np.maximum(y_value + t_value[None, :], 0.0)
        cert = u_value.mean(axis=0) - alpha_value * t_value
        feas = max(float(np.max(cert)), float(alpha_value.sum() - alpha_total))
        return max(0.0, feas), float(np.max(cert))

    try:
        result = run_dca(
            initial={"p": p0, "t": t0, "alpha": alpha0},
            solve_majorized=solve_majorized,
            residual=residual,
            vectorize=lambda values: np.concatenate(
                [
                    np.asarray(values["p"], dtype=float).ravel(),
                    np.asarray(values["t"], dtype=float),
                    np.asarray(values["alpha"], dtype=float),
                ]
            ),
            max_iter=min(int(dca_cfg["max_iter"]), 4),
            tol=float(dca_cfg["tol"]),
            alpha_total=alpha_total,
        )
    except Exception as exc:
        return _failed_payload(instance, "cvar", "optimized", alpha0, str(exc), 0.0)
    if result.feasibility_residual > accept_tol:
        return _payload(
            instance,
            certificate="cvar",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"cvar_optimized_fallback_equal_feasible:dca_candidate_feasibility={result.feasibility_residual:.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "t_variables": np.asarray(equal_payload.get("t_variables", t0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "attempted_optimized_objective": float(result.objective),
                "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
            },
        )
    if float(result.objective) >= float(equal_payload["objective"]) - 1.0e-7 * (1.0 + abs(float(equal_payload["objective"]))):
        return _payload(
            instance,
            certificate="cvar",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"cvar_optimized_fallback_equal_objective_dominates:dca_candidate_objective={float(result.objective):.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "t_variables": np.asarray(equal_payload.get("t_variables", t0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "attempted_optimized_objective": float(result.objective),
                "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
            },
        )
    alpha_value = np.maximum(np.asarray(result.values["alpha"], dtype=float), min_alpha)
    if float(alpha_value.sum()) > alpha_total:
        alpha_value *= alpha_total / float(alpha_value.sum())
    p_value = np.asarray(result.values["p"], dtype=float)
    t_value = np.asarray(result.values["t"], dtype=float)
    y_value = _y_values(instance, p_value, instance.flow_residual_train)
    u_value = np.maximum(y_value + t_value[None, :], 0.0)
    cert_value = u_value.mean(axis=0) - alpha_value * t_value
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha_value, alpha_total)
    return _payload(
        instance,
        certificate="cvar",
        allocation="optimized",
        p_value=p_value,
        alpha=alpha_value,
        objective=result.objective,
        runtime=result.runtime,
        status=result.status,
        iterations=result.iterations,
        stationarity=result.stationarity_residual,
        feasibility=feasibility,
        logs=logs_to_dicts(result.logs),
        extra={
            "t_variables": t_value.tolist(),
            "certificate_values": cert_value.tolist(),
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def _solve_bernstein_equal(instance: PowerInstance, alpha: np.ndarray) -> dict[str, object]:
    residual_mean, residual_std = _residual_stats(instance)
    p = cp.Variable((instance.forecast_flow_offset.shape[0], len(instance.generator_ids)))
    theta = cp.Variable(instance.m)
    mu = _affine_mean_terms(instance, p, residual_mean)
    cert = mu + cp.multiply(0.5 * residual_std**2, cp.inv_pos(theta)) - cp.multiply(np.log(alpha), theta)
    certificate_constraint = cert <= 0.0
    problem = cp.Problem(
        cp.Minimize(_mean_generation_cost(instance, p)),
        _base_constraints(instance, p) + [theta >= 1.0e-8, certificate_constraint],
    )
    start = time.perf_counter()
    diag = solve_problem(problem)
    if diag.status not in {"optimal", "optimal_inaccurate"}:
        return _failed_payload(instance, "bernstein", "equal", alpha, diag.status, time.perf_counter() - start)
    p_value = np.asarray(p.value, dtype=float)
    theta_value = np.asarray(theta.value, dtype=float)
    cert_value = _affine_mean_values(instance, p_value, residual_mean) + 0.5 * residual_std**2 / theta_value - theta_value * np.log(alpha)
    certificate_dual = _dual_vector(certificate_constraint.dual_value, instance.m)
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha, instance.alpha)
    return _payload(
        instance,
        certificate="bernstein",
        allocation="equal",
        p_value=p_value,
        alpha=alpha,
        objective=_objective_value(instance, p_value),
        runtime=diag.runtime,
        status=diag.status,
        iterations=0,
        stationarity=feasibility,
        feasibility=feasibility,
        extra={
            "theta_variables": theta_value.tolist(),
            "certificate_values": cert_value.tolist(),
            "certificate_dual_values": certificate_dual.tolist(),
            "kkt_dual_source": "bernstein_equal_convex_problem",
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def _solve_bernstein_optimized(instance: PowerInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    residual_mean, residual_std = _residual_stats(instance)
    min_alpha = float(dca_cfg["min_alpha"])
    accept_tol = certificate_accept_tol(dca_cfg)
    min_theta = 1.0e-8
    p0 = np.asarray(equal_payload["dispatch_mw"], dtype=float)
    alpha0 = np.asarray(equal_payload["alpha_vector"], dtype=float)
    theta0 = np.maximum(np.asarray(equal_payload.get("theta_variables", np.ones(instance.m)), dtype=float), min_theta)
    if not bool(equal_payload.get("valid_certificate", False)):
        return _failed_payload(instance, "bernstein", "optimized", alpha0, "equal_warm_start_invalid", 0.0)
    base_objective = _objective_value(instance, _base_dispatch(instance))
    if bool(equal_payload.get("valid_certificate", False)) and float(equal_payload["objective"]) <= base_objective + 1.0e-7 * (1.0 + abs(base_objective)):
        return _payload(
            instance,
            certificate="bernstein",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=0.0,
            status="optimal_unconstrained_lower_bound",
            iterations=0,
            stationarity=0.0,
            feasibility=float(equal_payload["feasibility_residual"]),
            extra={
                "theta_variables": np.asarray(equal_payload.get("theta_variables", theta0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_dual_values": list(equal_payload.get("certificate_dual_values", [])),
                "kkt_dual_source": "equal_solution_inherited_by_optimized_lower_bound",
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "optimization_note": "equal allocation already attains the unconstrained economic dispatch lower bound",
            },
        )

    def solve_majorized(previous: dict[str, np.ndarray], iteration: int):
        p = cp.Variable(p0.shape)
        theta = cp.Variable(instance.m)
        alpha = cp.Variable(instance.m)
        theta_prev = np.maximum(previous["theta"], min_theta)
        theta_log_lin = cp.multiply(1.0 + np.log(theta_prev), theta) - theta_prev
        mu = _affine_mean_terms(instance, p, residual_mean)
        cert = mu + cp.multiply(0.5 * residual_std**2, cp.inv_pos(theta)) + cp.rel_entr(theta, alpha) - theta_log_lin
        certificate_constraint = cert <= 0.0
        alpha_lower_constraint = alpha >= min_alpha
        budget_constraint = cp.sum(alpha) <= instance.alpha
        theta_lower_constraint = theta >= min_theta
        problem = cp.Problem(
            cp.Minimize(
                _mean_generation_cost(instance, p)
                + float(dca_cfg["proximal_weight"]) * cp.sum_squares(alpha - previous["alpha"])
            ),
            _base_constraints(instance, p)
            + [certificate_constraint, alpha_lower_constraint, budget_constraint, theta_lower_constraint],
        )
        diag = solve_problem(problem)
        require_success(diag)
        values = {
            "p": np.asarray(p.value, dtype=float),
            "theta": np.asarray(theta.value, dtype=float),
            "alpha": np.asarray(alpha.value, dtype=float),
        }
        diagnostics = {
            "certificate_dual_values": _dual_vector(certificate_constraint.dual_value, instance.m),
            "alpha_lower_duals": _dual_vector(alpha_lower_constraint.dual_value, instance.m),
            "budget_constraint_dual": _dual_scalar(budget_constraint.dual_value),
            "theta_lower_duals": _dual_vector(theta_lower_constraint.dual_value, instance.m),
            "kkt_dual_source": f"bernstein_dca_majorized_iteration_{iteration}",
        }
        return values, _objective_value(instance, values["p"]), diag.status, diag.runtime, diagnostics

    def residual(values: dict[str, np.ndarray | float], previous: dict[str, np.ndarray]):
        p_value = np.asarray(values["p"], dtype=float)
        theta_value = np.maximum(np.asarray(values["theta"], dtype=float), min_theta)
        alpha_value = np.maximum(np.asarray(values["alpha"], dtype=float), min_alpha)
        mu_value = _affine_mean_values(instance, p_value, residual_mean)
        cert = mu_value + 0.5 * residual_std**2 / theta_value - theta_value * np.log(alpha_value)
        feas = max(float(np.max(cert)), float(alpha_value.sum() - instance.alpha))
        return max(0.0, feas), float(np.max(cert))

    try:
        result = run_dca(
            initial={"p": p0, "theta": theta0, "alpha": alpha0},
            solve_majorized=solve_majorized,
            residual=residual,
            vectorize=lambda values: np.concatenate(
                [
                    np.asarray(values["p"], dtype=float).ravel(),
                    np.asarray(values["theta"], dtype=float),
                    np.asarray(values["alpha"], dtype=float),
                ]
            ),
            max_iter=min(int(dca_cfg["max_iter"]), 4),
            tol=float(dca_cfg["tol"]),
            alpha_total=instance.alpha,
        )
    except Exception as exc:
        return _failed_payload(instance, "bernstein", "optimized", alpha0, str(exc), 0.0)
    if result.feasibility_residual > accept_tol:
        return _payload(
            instance,
            certificate="bernstein",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"bernstein_optimized_fallback_equal_feasible:dca_candidate_feasibility={result.feasibility_residual:.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "theta_variables": np.asarray(equal_payload.get("theta_variables", theta0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_dual_values": list(equal_payload.get("certificate_dual_values", [])),
                "kkt_dual_source": "equal_solution_inherited_by_feasibility_fallback",
                "attempted_certificate_dual_values": np.asarray(
                    result.diagnostics.get("certificate_dual_values", np.full(instance.m, np.nan)),
                    dtype=float,
                ).tolist(),
                "attempted_budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
                "alpha_lower_bound": min_alpha,
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "attempted_optimized_objective": float(result.objective),
                "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
            },
        )
    if float(result.objective) >= float(equal_payload["objective"]) - 1.0e-7 * (1.0 + abs(float(equal_payload["objective"]))):
        return _payload(
            instance,
            certificate="bernstein",
            allocation="optimized",
            p_value=p0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"bernstein_optimized_fallback_equal_objective_dominates:dca_candidate_objective={float(result.objective):.6g}",
            iterations=result.iterations,
            stationarity=float(equal_payload["stationarity_residual"]),
            feasibility=float(equal_payload["feasibility_residual"]),
            logs=logs_to_dicts(result.logs),
            extra={
                "theta_variables": np.asarray(equal_payload.get("theta_variables", theta0), dtype=float).tolist(),
                "certificate_values": list(equal_payload.get("certificate_values", [])),
                "certificate_dual_values": list(equal_payload.get("certificate_dual_values", [])),
                "kkt_dual_source": "equal_solution_inherited_by_objective_fallback",
                "attempted_certificate_dual_values": np.asarray(
                    result.diagnostics.get("certificate_dual_values", np.full(instance.m, np.nan)),
                    dtype=float,
                ).tolist(),
                "attempted_budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
                "alpha_lower_bound": min_alpha,
                "certificate_residual_max": float(equal_payload.get("certificate_residual_max", equal_payload["feasibility_residual"])),
                "budget_residual": float(equal_payload.get("budget_residual", 0.0)),
                "attempted_optimized_objective": float(result.objective),
                "attempted_optimized_feasibility_residual": float(result.feasibility_residual),
            },
        )
    alpha_value = np.maximum(np.asarray(result.values["alpha"], dtype=float), min_alpha)
    if float(alpha_value.sum()) > instance.alpha:
        alpha_value *= instance.alpha / float(alpha_value.sum())
    p_value = np.asarray(result.values["p"], dtype=float)
    theta_value = np.maximum(np.asarray(result.values["theta"], dtype=float), min_theta)
    cert_value = _affine_mean_values(instance, p_value, residual_mean) + 0.5 * residual_std**2 / theta_value - theta_value * np.log(alpha_value)
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha_value, instance.alpha)
    return _payload(
        instance,
        certificate="bernstein",
        allocation="optimized",
        p_value=p_value,
        alpha=alpha_value,
        objective=result.objective,
        runtime=result.runtime,
        status=result.status,
        iterations=result.iterations,
        stationarity=result.stationarity_residual,
        feasibility=feasibility,
        logs=logs_to_dicts(result.logs),
        extra={
            "theta_variables": theta_value.tolist(),
            "certificate_values": cert_value.tolist(),
            "certificate_dual_values": np.asarray(
                result.diagnostics.get("certificate_dual_values", np.full(instance.m, np.nan)),
                dtype=float,
            ).tolist(),
            "budget_constraint_dual": float(result.diagnostics.get("budget_constraint_dual", np.nan)),
            "alpha_lower_duals": np.asarray(
                result.diagnostics.get("alpha_lower_duals", np.full(instance.m, np.nan)),
                dtype=float,
            ).tolist(),
            "theta_lower_duals": np.asarray(
                result.diagnostics.get("theta_lower_duals", np.full(instance.m, np.nan)),
                dtype=float,
            ).tolist(),
            "alpha_lower_bound": min_alpha,
            "kkt_dual_source": str(result.diagnostics.get("kkt_dual_source", "bernstein_dca_last_majorized_subproblem")),
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def _solve_cantelli_equal(instance: PowerInstance, alpha: np.ndarray) -> dict[str, object]:
    residual_mean, residual_std = _residual_stats(instance)
    q = cantelli_quantile(alpha)
    p = cp.Variable((instance.forecast_flow_offset.shape[0], len(instance.generator_ids)))
    cert = _affine_mean_terms(instance, p, residual_mean) + residual_std * q
    problem = cp.Problem(cp.Minimize(_mean_generation_cost(instance, p)), _base_constraints(instance, p) + [cert <= 0.0])
    start = time.perf_counter()
    diag = solve_problem(problem)
    if diag.status not in {"optimal", "optimal_inaccurate"}:
        return _failed_payload(instance, "cantelli", "equal", alpha, diag.status, time.perf_counter() - start)
    p_value = np.asarray(p.value, dtype=float)
    cert_value = _affine_mean_values(instance, p_value, residual_mean) + residual_std * q
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha, instance.alpha)
    return _payload(
        instance,
        certificate="cantelli",
        allocation="equal",
        p_value=p_value,
        alpha=alpha,
        objective=_objective_value(instance, p_value),
        runtime=diag.runtime,
        status=diag.status,
        iterations=0,
        stationarity=feasibility,
        feasibility=feasibility,
        extra={
            "certificate_values": cert_value.tolist(),
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def _solve_cantelli_optimized(instance: PowerInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    residual_mean, residual_std = _residual_stats(instance)
    p_value = np.asarray(equal_payload["dispatch_mw"], dtype=float)
    alpha = np.asarray(equal_payload["alpha_vector"], dtype=float)
    if not bool(equal_payload.get("valid_certificate", False)):
        return _failed_payload(instance, "cantelli", "optimized", alpha, "equal_warm_start_invalid", 0.0)
    mu = _affine_mean_values(instance, p_value, residual_mean)
    ok, used, _ = validate_cantelli_budget(mu, residual_std, alpha, float(dca_cfg["feasibility_tol"]))
    status = "unsupported_coupled_cantelli_optimized" if not ok else "cantelli_optimized_equal_validated"
    q = cantelli_quantile(alpha)
    cert_value = mu + residual_std * q
    certificate_residual, budget_residual, feasibility = _residual_components(cert_value, alpha, instance.alpha)
    return _payload(
        instance,
        certificate="cantelli",
        allocation="optimized",
        p_value=p_value,
        alpha=alpha,
        objective=float(equal_payload["objective"]),
        runtime=0.0,
        status=status,
        iterations=0,
        stationarity=0.0 if ok else float(used - instance.alpha),
        feasibility=0.0 if ok else max(feasibility, float(used - instance.alpha)),
        extra={
            "cantelli_budget_used": float(used),
            "certificate_values": cert_value.tolist(),
            "certificate_residual_max": certificate_residual,
            "budget_residual": budget_residual,
        },
    )


def solve_power_instance(instance: PowerInstance, results_dir: str | Path, dca_cfg: dict[str, object]) -> pd.DataFrame:
    root = Path(results_dir)
    solution_dir = root / "solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    min_alpha = float(dca_cfg["min_alpha"])
    alpha_equal = equal_allocation(instance.m, instance.alpha, min_alpha=min_alpha)
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
                case="power",
                certificate=certificate,
                allocation=allocation,
                solver_status=str(payload["solver_status"]),
                valid_certificate=bool(payload["valid_certificate"]),
                valid_optimization=bool(payload["valid_optimization"]),
                fallback_used=bool(payload.get("fallback_used", False)),
                feasibility_residual=float(payload["feasibility_residual"]),
                calibration_joint_violation=float(payload["calibration_emergency_joint_violation"]),
                alpha_total=instance.alpha,
                dca_cfg=dca_cfg,
            )
        )
        write_json(solution_dir / f"power_{certificate}_{allocation}.json", payload)
        rows.append(
            {
                "case": "power",
                "certificate": certificate,
                "allocation": allocation,
                "objective": float(payload["objective"]),
                "relative_improvement": relative_improvement(equal_objectives[certificate], float(payload["objective"])),
                "calibration_joint_violation": float(payload["calibration_emergency_joint_violation"]),
                "heldout_joint_violation": float(payload["heldout_emergency_joint_violation"]),
                "calibration_emergency_joint_violation": float(payload["calibration_emergency_joint_violation"]),
                "heldout_emergency_joint_violation": float(payload["heldout_emergency_joint_violation"]),
                "calibration_cont_overload_rate": float(payload["calibration_cont_overload_rate"]),
                "heldout_cont_overload_rate": float(payload["heldout_cont_overload_rate"]),
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
                "sum_alpha": float(payload["sum_alpha"]),
                "certificate_residual_max": float(payload.get("certificate_residual_max", payload["feasibility_residual"])),
                "budget_residual": float(payload.get("budget_residual", 0.0)),
                "normalized_certificate_residual_max": float(
                    payload.get("normalized_certificate_residual_max", payload.get("certificate_residual_max", payload["feasibility_residual"]))
                ),
                "calibration_max_violation_tolerance": float(payload["calibration_max_violation_tolerance"]),
                "heldout_max_violation_tolerance": float(payload["heldout_max_violation_tolerance"]),
                "max_budget_share": float(payload["max_budget_share"]),
                "normalized_entropy": float(payload["normalized_entropy"]),
                "valid_certificate": bool(payload["valid_certificate"]),
                "valid_optimization": bool(payload["valid_optimization"]),
                "fallback_used": bool(payload.get("fallback_used", False)),
                "fallback_reason": str(payload.get("fallback_reason", "")),
                "optimization_note": str(payload.get("optimization_note", "")),
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
                "primary_rating": str(payload["primary_rating"]),
                "mean_lte_multiplier": float(payload["mean_lte_multiplier"]),
                "mean_ste_multiplier": float(payload["mean_ste_multiplier"]),
                "n_nominal_cont_overloaded": int(payload["n_nominal_cont_overloaded"]),
                "n_nominal_lte_overloaded": int(payload["n_nominal_lte_overloaded"]),
                "solver_status": str(payload["solver_status"]),
            }
        )
    return pd.DataFrame(rows)
