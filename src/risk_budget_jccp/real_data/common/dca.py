from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DcaIterationLog:
    iteration: int
    objective: float
    max_certificate_violation: float
    risk_budget_used: float
    step_norm: float
    stationarity_residual: float
    solver_status: str
    solver_runtime: float


@dataclass(frozen=True)
class DcaResult:
    values: dict[str, np.ndarray | float]
    objective: float
    status: str
    runtime: float
    iterations: int
    stationarity_residual: float
    feasibility_residual: float
    logs: list[DcaIterationLog] = field(default_factory=list)
    diagnostics: dict[str, np.ndarray | float | str] = field(default_factory=dict)


def relative_step(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.linalg.norm(current - previous) / (1.0 + np.linalg.norm(previous)))


def run_dca(
    *,
    initial: dict[str, np.ndarray],
    solve_majorized: Callable[
        [dict[str, np.ndarray], int],
        tuple[dict[str, np.ndarray | float], float, str, float]
        | tuple[dict[str, np.ndarray | float], float, str, float, dict[str, np.ndarray | float | str]],
    ],
    residual: Callable[[dict[str, np.ndarray | float], dict[str, np.ndarray]], tuple[float, float]],
    vectorize: Callable[[dict[str, np.ndarray | float]], np.ndarray],
    max_iter: int,
    tol: float,
    alpha_total: float,
) -> DcaResult:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    start = time.perf_counter()
    previous = {key: np.asarray(value, dtype=float).copy() for key, value in initial.items()}
    previous_vector = vectorize(previous)
    logs: list[DcaIterationLog] = []
    last_values: dict[str, np.ndarray | float] = previous
    last_objective = float("inf")
    status = "not_started"
    stationarity = float("inf")
    feasibility = float("inf")
    last_diagnostics: dict[str, np.ndarray | float | str] = {}

    for iteration in range(1, max_iter + 1):
        solved = solve_majorized(previous, iteration)
        if len(solved) == 4:
            values, objective, status, solver_runtime = solved
            diagnostics = {}
        else:
            values, objective, status, solver_runtime, diagnostics = solved
        current_vector = vectorize(values)
        step = relative_step(current_vector, previous_vector)
        feasibility, max_violation = residual(values, previous)
        stationarity = step + max(0.0, max_violation)
        alpha_vec = np.asarray(values.get("alpha", np.array([alpha_total])), dtype=float)
        logs.append(
            DcaIterationLog(
                iteration=iteration,
                objective=float(objective),
                max_certificate_violation=float(max_violation),
                risk_budget_used=float(alpha_vec.sum()),
                step_norm=step,
                stationarity_residual=stationarity,
                solver_status=str(status),
                solver_runtime=float(solver_runtime),
            )
        )
        last_values = values
        last_objective = float(objective)
        last_diagnostics = diagnostics
        previous = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in values.items()
            if isinstance(value, np.ndarray)
        }
        previous_vector = current_vector
        if stationarity <= tol and feasibility <= max(tol, 1.0e-8):
            break

    return DcaResult(
        values=last_values,
        objective=last_objective,
        status=status,
        runtime=float(time.perf_counter() - start),
        iterations=len(logs),
        stationarity_residual=float(stationarity),
        feasibility_residual=float(feasibility),
        logs=logs,
        diagnostics=last_diagnostics,
    )


def logs_to_dicts(logs: list[DcaIterationLog]) -> list[dict[str, float | int | str]]:
    return [
        {
            "iteration": log.iteration,
            "objective": log.objective,
            "max_certificate_violation": log.max_certificate_violation,
            "risk_budget_used": log.risk_budget_used,
            "step_norm": log.step_norm,
            "stationarity_residual": log.stationarity_residual,
            "solver_status": log.solver_status,
            "solver_runtime": log.solver_runtime,
        }
        for log in logs
    ]
