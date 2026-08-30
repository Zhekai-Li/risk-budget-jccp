from __future__ import annotations

import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from risk_budget_jccp.real_data.cases.french.build_instance import FrenchInstance
from risk_budget_jccp.real_data.common.certificates import cantelli_quantile, equal_allocation, validate_cantelli_budget
from risk_budget_jccp.real_data.common.dca import logs_to_dicts, run_dca
from risk_budget_jccp.real_data.common.logging_utils import write_json
from risk_budget_jccp.real_data.common.metrics import alpha_metrics, relative_improvement, violation_metrics
from risk_budget_jccp.real_data.common.result_status import certificate_accept_tol, status_fields
from risk_budget_jccp.real_data.common.solvers import require_success, solve_problem
from risk_budget_jccp.algorithms.optimized_cvar import finite_scenario_cvar


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


def _french_cvar_diagnostics(instance: FrenchInstance, w: np.ndarray, x: np.ndarray, alpha: np.ndarray) -> dict[str, object]:
    alpha_arr = np.asarray(alpha, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    q = np.zeros(instance.m, dtype=float)
    derivative = np.zeros(instance.m, dtype=float)
    for index in range(instance.m):
        q[index], derivative[index] = _finite_cvar_and_derivative(-instance.returns_train[:, index], float(alpha_arr[index]))
    allocated_tail = w_arr * q
    marginal = -w_arr * derivative
    marginal = np.where(x_arr > 1.0e-12, marginal, 0.0)
    return {
        "allocated_tail_cvar": allocated_tail.tolist(),
        "cvar_tail_derivative": derivative.tolist(),
        "cvar_marginal_budget_value": marginal.tolist(),
        "margin_contribution": x_arr.tolist(),
        "theory_budget_driver": marginal.tolist(),
        "theory_driver_source": "finite_scenario_cvar_derivative_fixed_weight",
    }


def _french_cantelli_diagnostics(instance: FrenchInstance, w: np.ndarray, x: np.ndarray, alpha: np.ndarray) -> dict[str, object]:
    alpha_arr = np.asarray(alpha, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    sigma = instance.returns_train.std(axis=0, ddof=0)
    q_prime = _cantelli_quantile_derivative(alpha_arr)
    marginal = -w_arr * sigma * q_prime
    marginal = np.where(x_arr > 1.0e-12, marginal, 0.0)
    return {
        "cantelli_quantile_derivative": q_prime.tolist(),
        "cantelli_marginal_budget_value": marginal.tolist(),
        "margin_contribution": x_arr.tolist(),
        "theory_budget_driver": marginal.tolist(),
        "theory_driver_source": "cantelli_closed_form_derivative_fixed_weight",
    }


def _violation_scale(returns: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return one model-wide loss scale for numerical violation counting."""
    weighted_returns = returns * np.asarray(w, dtype=float)[None, :]
    scale = max(float(np.max(np.abs(weighted_returns))), float(np.max(np.abs(np.asarray(x, dtype=float)))))
    return np.full(np.asarray(x, dtype=float).shape, scale, dtype=float)


def _evaluate(instance: FrenchInstance, w: np.ndarray, x: np.ndarray) -> dict[str, float | int | list[int]]:
    w_arr = np.asarray(w, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_eval = -instance.returns_test * w_arr[None, :] - x_arr[None, :]
    return violation_metrics(
        y_eval,
        reference_scale=_violation_scale(instance.returns_test, w_arr, x_arr),
    )


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


def _payload(
    *,
    instance: FrenchInstance,
    certificate: str,
    allocation: str,
    w: np.ndarray,
    x: np.ndarray,
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
    alpha_arr = np.asarray(alpha, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    calibration_metrics = violation_metrics(
        -instance.returns_train * w_arr[None, :] - x_arr[None, :],
        reference_scale=_violation_scale(instance.returns_train, w_arr, x_arr),
    )
    heldout_metrics = _evaluate(instance, w_arr, x_arr)
    valid = (
        feasibility <= 1.0e-4
        and abs(float(w_arr.sum()) - 1.0) <= 1.0e-5
        and not str(status).startswith("failed")
    )
    fallback_used = "fallback" in str(status)
    certificate_residual_max = max(0.0, float(feasibility))
    if extra and "certificate_values" in extra:
        certificate_residual_max = max(
            0.0,
            float(np.max(np.asarray(extra["certificate_values"], dtype=float), initial=0.0)),
        )
    budget_residual = max(0.0, float(alpha_arr.sum() - instance.alpha))
    result: dict[str, object] = {
        "case": "french",
        "certificate": certificate,
        "allocation": allocation,
        "objective": float(objective),
        "runtime": float(runtime),
        "solver_status": status,
        "majorization_iterations": int(iterations),
        "stationarity_residual": float(stationarity),
        "feasibility_residual": float(feasibility),
        "certificate_residual_max": certificate_residual_max,
        "budget_residual": budget_residual,
        "sum_alpha": float(alpha_arr.sum()),
        "valid_certificate": bool(valid),
        "valid_optimization": bool(valid and allocation == "optimized" and not fallback_used),
        "failure_reason": "" if valid else str(status),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(status) if fallback_used else "",
        "alpha_vector": alpha_arr.tolist(),
        "x_variables": x_arr.tolist(),
        "weights": w_arr.tolist(),
        "margin_buffers": x_arr.tolist(),
        "target_return": instance.target_return,
        "realized_calibration_return": float(instance.returns_train.mean(axis=0) @ w_arr),
        "weight_sum": float(w_arr.sum()),
        **_prefixed_metrics("calibration", calibration_metrics),
        **_prefixed_metrics("heldout", heldout_metrics),
        # Backward-compatible held-out aliases.
        **heldout_metrics,
        **alpha_metrics(alpha_arr),
    }
    if logs is not None:
        result["majorization_logs"] = logs
    if extra:
        result.update(extra)
    return result


def _base_constraints(instance: FrenchInstance, w: cp.Variable, x: cp.Variable) -> list[cp.Constraint]:
    mu = instance.returns_train.mean(axis=0)
    return [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= instance.max_weight,
        x >= 0.0,
        mu @ w >= instance.target_return,
    ]


def _loss_cvar_coefficients(returns: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            finite_scenario_cvar(-returns[:, index], alpha=float(alpha_vec[index]))
            for index in range(returns.shape[1])
        ],
        dtype=float,
    )


def _solve_weight_lp(instance: FrenchInstance, coefficients: np.ndarray) -> np.ndarray:
    mu = instance.returns_train.mean(axis=0)
    result = linprog(
        coefficients,
        A_ub=-mu[np.newaxis, :],
        b_ub=np.array([-instance.target_return]),
        A_eq=np.ones((1, instance.m), dtype=float),
        b_eq=np.array([1.0]),
        bounds=[(0.0, instance.max_weight)] * instance.m,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"French CVaR weight LP failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _solve_cvar_equal(instance: FrenchInstance, alpha_vec: np.ndarray) -> dict[str, object]:
    start = time.perf_counter()
    q = _loss_cvar_coefficients(instance.returns_train, alpha_vec)
    w_val = _solve_weight_lp(instance, q)
    x_val = np.maximum(w_val * q, 0.0)
    cert = np.array(
        [
            finite_scenario_cvar(
                -w_val[index] * instance.returns_train[:, index] - x_val[index],
                alpha=float(alpha_vec[index]),
            )
            for index in range(instance.m)
        ],
        dtype=float,
    )
    return _payload(
        instance=instance,
        certificate="cvar",
        allocation="equal",
        w=w_val,
        x=x_val,
        alpha=alpha_vec,
        objective=float(x_val.sum()),
        runtime=time.perf_counter() - start,
        status="separable_cvar_fixed_alpha_lp",
        iterations=0,
        stationarity=max(0.0, float(np.max(cert))),
        feasibility=max(0.0, float(np.max(cert))),
        extra={
            **_french_cvar_diagnostics(instance, w_val, x_val, alpha_vec),
            "certificate_values": cert.tolist(),
        },
    )


def _solve_cvar_optimized(instance: FrenchInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    m = instance.m
    min_alpha = float(dca_cfg["min_alpha"])
    w0 = np.asarray(equal_payload["weights"], dtype=float)
    alpha0 = np.asarray(equal_payload["alpha_vector"], dtype=float)

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:m], np.clip(z[m:], min_alpha, instance.alpha)

    def objective(z: np.ndarray) -> float:
        w, alpha = unpack(z)
        q = _loss_cvar_coefficients(instance.returns_train, alpha)
        return float(np.maximum(w, 0.0) @ q)

    def ineq_constraints_fun(z: np.ndarray) -> np.ndarray:
        w, alpha = unpack(z)
        mu = instance.returns_train.mean(axis=0)
        return np.array([float(mu @ w - instance.target_return), instance.alpha - float(alpha.sum())])

    result = minimize(
        objective,
        np.concatenate([w0, alpha0]),
        method="SLSQP",
        bounds=[(0.0, instance.max_weight)] * m + [(min_alpha, instance.alpha)] * m,
        constraints=[
            {"type": "eq", "fun": lambda z: float(unpack(z)[0].sum() - 1.0)},
            {"type": "ineq", "fun": ineq_constraints_fun},
        ],
        options={"ftol": float(dca_cfg["tol"]), "maxiter": 300, "disp": False},
    )
    w_val, alpha_val = unpack(np.asarray(result.x if result.success else np.concatenate([w0, alpha0]), dtype=float))
    if alpha_val.sum() > instance.alpha:
        alpha_val *= instance.alpha / float(alpha_val.sum())
    q = _loss_cvar_coefficients(instance.returns_train, alpha_val)
    x_val = np.maximum(w_val * q, 0.0)
    cert = np.array(
        [
            finite_scenario_cvar(
                -w_val[index] * instance.returns_train[:, index] - x_val[index],
                alpha=float(alpha_val[index]),
            )
            for index in range(m)
        ],
        dtype=float,
    )
    feasibility = max(
        max(0.0, float(np.max(cert))),
        max(0.0, float(alpha_val.sum() - instance.alpha)),
        abs(float(w_val.sum()) - 1.0),
    )
    status = "separable_cvar_scipy_slsqp" if result.success and feasibility <= 1.0e-5 else f"failed_cvar_validation:{result.message}"
    return _payload(
        instance=instance,
        certificate="cvar",
        allocation="optimized",
        w=w_val,
        x=x_val,
        alpha=alpha_val,
        objective=float(x_val.sum()),
        runtime=time.perf_counter() - start,
        status=status,
        iterations=int(getattr(result, "nit", 0)),
        stationarity=feasibility,
        feasibility=feasibility,
        logs=[
            {
                "iteration": int(getattr(result, "nit", 0)),
                "objective": float(x_val.sum()),
                "max_certificate_violation": float(np.max(cert)),
                "risk_budget_used": float(alpha_val.sum()),
                "step_norm": 0.0,
                "stationarity_residual": feasibility,
                "solver_status": str(getattr(result, "message", "")),
                "solver_runtime": time.perf_counter() - start,
            }
        ],
        extra={
            **_french_cvar_diagnostics(instance, w_val, x_val, alpha_val),
            "certificate_values": cert.tolist(),
        },
    )


def _solve_bernstein_equal(instance: FrenchInstance, alpha_vec: np.ndarray) -> dict[str, object]:
    m = instance.m
    mu = instance.returns_train.mean(axis=0)
    sigma = instance.returns_train.std(axis=0, ddof=0)
    w = cp.Variable(m)
    x = cp.Variable(m)
    theta = cp.Variable(m)
    variance_terms = cp.hstack(
        [0.5 * sigma[i] ** 2 * cp.quad_over_lin(w[i], theta[i]) for i in range(m)]
    )
    cert = -cp.multiply(mu, w) - x + variance_terms - cp.multiply(np.log(alpha_vec), theta)
    certificate_constraint = cert <= 0.0
    problem = cp.Problem(cp.Minimize(cp.sum(x)), _base_constraints(instance, w, x) + [theta >= 1.0e-8, certificate_constraint])
    diag = solve_problem(problem)
    require_success(diag)
    wv = np.asarray(w.value, dtype=float).reshape(m)
    xv = np.asarray(x.value, dtype=float).reshape(m)
    tv = np.asarray(theta.value, dtype=float).reshape(m)
    cert_val = -mu * wv - xv + 0.5 * sigma**2 * wv**2 / tv - tv * np.log(alpha_vec)
    certificate_dual = _dual_vector(certificate_constraint.dual_value, m)
    return _payload(
        instance=instance,
        certificate="bernstein",
        allocation="equal",
        w=wv,
        x=xv,
        alpha=alpha_vec,
        objective=float(xv.sum()),
        runtime=diag.runtime,
        status=diag.status,
        iterations=0,
        stationarity=max(0.0, float(np.max(cert_val))),
        feasibility=max(0.0, float(np.max(cert_val))),
        extra={
            "theta_variables": tv.tolist(),
            "certificate_values": cert_val.tolist(),
            "certificate_dual_values": certificate_dual.tolist(),
            "theory_budget_driver": (certificate_dual * tv).tolist(),
            "kkt_dual_source": "french_bernstein_equal_convex_problem",
            "theory_driver_source": "bernstein_kkt_lambda_theta",
        },
    )


def _solve_bernstein_optimized(instance: FrenchInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    m = instance.m
    mu = instance.returns_train.mean(axis=0)
    sigma = instance.returns_train.std(axis=0, ddof=0)
    min_alpha = float(dca_cfg["min_alpha"])
    min_theta = 1.0e-8
    w0 = np.asarray(equal_payload["weights"], dtype=float)
    x0 = np.asarray(equal_payload["x_variables"], dtype=float)
    alpha0 = np.asarray(equal_payload["alpha_vector"], dtype=float)
    theta0 = np.maximum(np.abs(w0) * sigma / np.maximum(x0 + np.abs(mu * w0), 1.0e-5), min_theta)

    def solve_majorized(previous: dict[str, np.ndarray], iteration: int):
        alpha_prev = np.maximum(previous["alpha"], min_alpha)
        theta_prev = np.maximum(previous["theta"], min_theta)
        w = cp.Variable(m)
        x = cp.Variable(m)
        theta = cp.Variable(m)
        alpha = cp.Variable(m)
        theta_log_lin = cp.multiply(1.0 + np.log(theta_prev), theta) - theta_prev
        variance_terms = cp.hstack(
            [0.5 * sigma[i] ** 2 * cp.quad_over_lin(w[i], theta[i]) for i in range(m)]
        )
        cert = (
            -cp.multiply(mu, w)
            - x
            + variance_terms
            + cp.rel_entr(theta, alpha)
            - theta_log_lin
        )
        certificate_constraint = cert <= 0.0
        alpha_lower_constraint = alpha >= min_alpha
        budget_constraint = cp.sum(alpha) <= instance.alpha
        theta_lower_constraint = theta >= min_theta
        problem = cp.Problem(
            cp.Minimize(cp.sum(x) + float(dca_cfg["proximal_weight"]) * cp.sum_squares(alpha - alpha_prev)),
            _base_constraints(instance, w, x)
            + [certificate_constraint, alpha_lower_constraint, budget_constraint, theta_lower_constraint],
        )
        diag = solve_problem(problem)
        require_success(diag)
        values = {
            "w": np.asarray(w.value, dtype=float).reshape(m),
            "x": np.asarray(x.value, dtype=float).reshape(m),
            "theta": np.asarray(theta.value, dtype=float).reshape(m),
            "alpha": np.asarray(alpha.value, dtype=float).reshape(m),
        }
        diagnostics = {
            "certificate_dual_values": _dual_vector(certificate_constraint.dual_value, m),
            "alpha_lower_duals": _dual_vector(alpha_lower_constraint.dual_value, m),
            "budget_constraint_dual": _dual_scalar(budget_constraint.dual_value),
            "theta_lower_duals": _dual_vector(theta_lower_constraint.dual_value, m),
            "kkt_dual_source": f"french_bernstein_dca_majorized_iteration_{iteration}",
        }
        return values, float(values["x"].sum()), diag.status, diag.runtime, diagnostics

    def residual(values: dict[str, np.ndarray | float], previous: dict[str, np.ndarray]):
        w = np.asarray(values["w"], dtype=float)
        x = np.asarray(values["x"], dtype=float)
        theta = np.maximum(np.asarray(values["theta"], dtype=float), min_theta)
        alpha = np.maximum(np.asarray(values["alpha"], dtype=float), min_alpha)
        cert = -mu * w - x + 0.5 * sigma**2 * w**2 / theta - theta * np.log(alpha)
        return max(max(0.0, float(alpha.sum() - instance.alpha)), float(np.max(cert))), float(np.max(cert))

    result = run_dca(
        initial={"w": w0, "x": x0, "theta": theta0, "alpha": alpha0},
        solve_majorized=solve_majorized,
        residual=residual,
        vectorize=lambda values: np.concatenate(
            [
                np.asarray(values["w"], dtype=float),
                np.asarray(values["x"], dtype=float),
                np.asarray(values["theta"], dtype=float),
                np.asarray(values["alpha"], dtype=float),
            ]
        ),
        max_iter=int(dca_cfg["max_iter"]),
        tol=float(dca_cfg["tol"]),
        alpha_total=instance.alpha,
    )
    accept_tol = certificate_accept_tol(dca_cfg)
    if result.feasibility_residual > accept_tol:
        return _payload(
            instance=instance,
            certificate="bernstein",
            allocation="optimized",
            w=w0,
            x=x0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"bernstein_optimized_fallback_equal_feasible:dca_candidate_feasibility={result.feasibility_residual:.6g}",
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
        return _payload(
            instance=instance,
            certificate="bernstein",
            allocation="optimized",
            w=w0,
            x=x0,
            alpha=alpha0,
            objective=float(equal_payload["objective"]),
            runtime=result.runtime,
            status=f"bernstein_optimized_fallback_equal_objective_dominates:dca_candidate_objective={float(result.objective):.6g}",
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
    w_value = np.asarray(result.values["w"], dtype=float)
    x_value = np.asarray(result.values["x"], dtype=float)
    theta_value = np.asarray(result.values["theta"], dtype=float)
    alpha_value = np.asarray(result.values["alpha"], dtype=float)
    cert_value = -mu * w_value - x_value + 0.5 * sigma**2 * w_value**2 / np.maximum(theta_value, min_theta) - np.maximum(theta_value, min_theta) * np.log(np.maximum(alpha_value, min_alpha))
    certificate_dual = np.asarray(result.diagnostics.get("certificate_dual_values", np.full(m, np.nan)), dtype=float)
    return _payload(
        instance=instance,
        certificate="bernstein",
        allocation="optimized",
        w=w_value,
        x=x_value,
        alpha=alpha_value,
        objective=result.objective,
        runtime=result.runtime,
        status=result.status,
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
            "kkt_dual_source": str(result.diagnostics.get("kkt_dual_source", "french_bernstein_dca_last_majorized_subproblem")),
            "theory_driver_source": "bernstein_kkt_lambda_theta",
        },
    )


def _solve_cantelli_equal(instance: FrenchInstance, alpha_vec: np.ndarray) -> dict[str, object]:
    m = instance.m
    mu = instance.returns_train.mean(axis=0)
    sigma = instance.returns_train.std(axis=0, ddof=0)
    q = cantelli_quantile(alpha_vec)
    w = cp.Variable(m)
    x = cp.Variable(m)
    cert = -cp.multiply(mu, w) - x + cp.multiply(q * sigma, w)
    problem = cp.Problem(cp.Minimize(cp.sum(x)), _base_constraints(instance, w, x) + [cert <= 0.0])
    diag = solve_problem(problem)
    require_success(diag)
    wv = np.asarray(w.value, dtype=float).reshape(m)
    xv = np.asarray(x.value, dtype=float).reshape(m)
    cert_val = -mu * wv - xv + q * sigma * wv
    return _payload(
        instance=instance,
        certificate="cantelli",
        allocation="equal",
        w=wv,
        x=xv,
        alpha=alpha_vec,
        objective=float(xv.sum()),
        runtime=diag.runtime,
        status=diag.status,
        iterations=0,
        stationarity=max(0.0, float(np.max(cert_val))),
        feasibility=max(0.0, float(np.max(cert_val))),
        extra={
            **_french_cantelli_diagnostics(instance, wv, xv, alpha_vec),
            "certificate_values": cert_val.tolist(),
        },
    )


def _solve_cantelli_optimized(instance: FrenchInstance, equal_payload: dict[str, object], dca_cfg: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    m = instance.m
    mu = instance.returns_train.mean(axis=0)
    sigma = instance.returns_train.std(axis=0, ddof=0)
    min_alpha = float(dca_cfg["min_alpha"])
    z0 = np.concatenate(
        [
            np.asarray(equal_payload["weights"], dtype=float),
            np.asarray(equal_payload["x_variables"], dtype=float),
            np.asarray(equal_payload["alpha_vector"], dtype=float),
        ]
    )

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return z[:m], z[m : 2 * m], z[2 * m :]

    def objective(z: np.ndarray) -> float:
        return float(np.sum(unpack(z)[1]))

    def constraints_fun(z: np.ndarray) -> np.ndarray:
        w, x, alpha = unpack(z)
        alpha = np.clip(alpha, min_alpha, 1.0 - 1.0e-9)
        cert_margin = -(-mu * w - x + sigma * w * cantelli_quantile(alpha))
        return np.concatenate(
            [
                np.array([1.0 - abs(float(w.sum() - 1.0))]),
                np.array([float(mu @ w - instance.target_return)]),
                np.array([instance.alpha - float(alpha.sum())]),
                cert_margin,
            ]
        )

    bounds = [(0.0, instance.max_weight)] * m + [(0.0, None)] * m + [(min_alpha, instance.alpha)] * m
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints_fun}],
        options={"ftol": float(dca_cfg["tol"]), "maxiter": 800, "disp": False},
    )
    z = np.asarray(result.x if result.success else z0, dtype=float)
    w, x, alpha = unpack(z)
    if alpha.sum() > instance.alpha:
        alpha = alpha * (instance.alpha / alpha.sum())
    mu_y = -mu * w - x
    sigma_y = sigma * w
    ok, used, beta = validate_cantelli_budget(mu_y, sigma_y, alpha, float(dca_cfg["feasibility_tol"]))
    cert = mu_y + sigma_y * cantelli_quantile(alpha)
    status = "scipy_slsqp"
    if not (result.success and ok):
        w = np.asarray(equal_payload["weights"], dtype=float)
        x = np.asarray(equal_payload["x_variables"], dtype=float)
        alpha = np.asarray(equal_payload["alpha_vector"], dtype=float)
        mu_y = -mu * w - x
        sigma_y = sigma * w
        ok, used, beta = validate_cantelli_budget(mu_y, sigma_y, alpha, float(dca_cfg["feasibility_tol"]))
        cert = mu_y + sigma_y * cantelli_quantile(alpha)
        status = f"cantelli_optimized_fallback_equal_feasible:{result.message};fallback_budget_used={used:.6g}"
    feasibility = max(max(0.0, float(np.max(cert))), max(0.0, used - instance.alpha))
    return _payload(
        instance=instance,
        certificate="cantelli",
        allocation="optimized",
        w=w,
        x=x,
        alpha=alpha,
        objective=float(x.sum()),
        runtime=time.perf_counter() - start,
        status=status,
        iterations=0,
        stationarity=max(0.0, float(np.max(cert))),
        feasibility=feasibility,
        extra={
            **_french_cantelli_diagnostics(instance, w, x, alpha),
            "certificate_values": cert.tolist(),
        },
    ) | {"cantelli_budget_used": float(used), "cantelli_beta": beta.tolist()}


def solve_french_instance(instance: FrenchInstance, results_dir: str | Path, dca_cfg: dict[str, object]) -> pd.DataFrame:
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
                case="french",
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
        write_json(solution_dir / f"french_{certificate}_{allocation}.json", payload)
        rows.append(
            {
                "case": "french",
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
                "weight_sum": float(payload["weight_sum"]),
                "solver_status": str(payload["solver_status"]),
            }
        )
    return pd.DataFrame(rows)
