from __future__ import annotations

from dataclasses import dataclass
import time

import cvxpy as cp
import numpy as np

from risk_budget_jccp.models.synthetic_capacity import (
    SyntheticCapacityInstance,
    initialize_theta,
    solve_fixed_bernstein,
)


_MIN_POSITIVE = 1e-12
_CVXPY_SOLVER = cp.CLARABEL
_VALID_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True, slots=True)
class OptimizedBernsteinMMResult:
    x: np.ndarray
    objective: float
    alpha: np.ndarray
    theta: np.ndarray
    iterations: int
    runtime: float
    final_residual: float
    solver_status: str


def _validate_solver_inputs(
    instance: SyntheticCapacityInstance,
    alpha: float,
    eps_alpha: float,
    eps_theta: float,
    max_iter: int,
    tol: float,
) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if eps_alpha <= 0.0:
        raise ValueError("eps_alpha must be positive")
    if eps_theta <= 0.0:
        raise ValueError("eps_theta must be positive")
    if alpha <= instance.num_constraints * eps_alpha:
        raise ValueError("alpha must exceed num_constraints * eps_alpha")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")


def _original_constraint_values(
    *,
    instance: SyntheticCapacityInstance,
    x: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    return (
        instance.demand
        - instance.constraint_matrix @ x
        + (instance.sigma**2) / (2.0 * theta)
        - theta * np.log(alpha)
    )


def _step_residual(
    *,
    instance: SyntheticCapacityInstance,
    x_prev: np.ndarray,
    alpha_prev: np.ndarray,
    objective_prev: float,
    x_next: np.ndarray,
    alpha_next: np.ndarray,
    theta_next: np.ndarray,
    objective_next: float,
    alpha_total: float,
) -> float:
    original_constraints = _original_constraint_values(
        instance=instance,
        x=x_next,
        alpha=alpha_next,
        theta=theta_next,
    )
    return float(
        max(
            np.linalg.norm(x_next - x_prev) / (1.0 + np.linalg.norm(x_prev)),
            np.linalg.norm(alpha_next - alpha_prev) / (1.0 + np.linalg.norm(alpha_prev)),
            abs(objective_next - objective_prev) / (1.0 + abs(objective_prev)),
            abs(float(alpha_next.sum()) - alpha_total),
            max(0.0, float(np.max(original_constraints))),
        )
    )


def solve_coupled_bernstein_mm(
    instance: SyntheticCapacityInstance,
    alpha: float,
    *,
    eps_alpha: float = 1e-6,
    eps_theta: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> OptimizedBernsteinMMResult:
    _validate_solver_inputs(
        instance=instance,
        alpha=alpha,
        eps_alpha=eps_alpha,
        eps_theta=eps_theta,
        max_iter=max_iter,
        tol=tol,
    )

    start = time.perf_counter()

    equal_alpha = np.full(instance.num_constraints, alpha / instance.num_constraints, dtype=float)
    equal_solution = solve_fixed_bernstein(instance, equal_alpha)

    x_prev = np.asarray(equal_solution.x, dtype=float)
    alpha_prev = equal_alpha
    theta_prev = np.maximum(initialize_theta(instance, alpha_prev), eps_theta)
    objective_prev = float(equal_solution.objective)

    solver_status = str(equal_solution.solver_status)
    final_residual = np.inf
    iterations_completed = 0

    for iteration in range(1, max_iter + 1):
        x_var = cp.Variable(instance.dimension, nonneg=True)
        alpha_var = cp.Variable(instance.num_constraints)
        theta_var = cp.Variable(instance.num_constraints)

        majorized_constraints = (
            instance.demand
            - instance.constraint_matrix @ x_var
            + cp.multiply(0.5 * (instance.sigma**2), cp.inv_pos(theta_var))
            + cp.rel_entr(theta_var, alpha_var)
            - cp.multiply(1.0 + np.log(theta_prev), theta_var)
            + theta_prev
        )

        problem = cp.Problem(
            cp.Minimize(instance.cost @ x_var),
            [
                majorized_constraints <= 0.0,
                cp.sum(alpha_var) == alpha,
                alpha_var >= eps_alpha,
                theta_var >= eps_theta,
            ],
        )
        problem.solve(solver=_CVXPY_SOLVER, warm_start=True)

        solver_status = str(problem.status)
        if problem.status not in _VALID_STATUSES:
            raise RuntimeError(f"MM subproblem failed with status {solver_status}")

        x_next = np.asarray(x_var.value, dtype=float).reshape(instance.dimension)
        alpha_next = np.asarray(alpha_var.value, dtype=float).reshape(instance.num_constraints)
        theta_next = np.maximum(
            np.asarray(theta_var.value, dtype=float).reshape(instance.num_constraints),
            _MIN_POSITIVE,
        )
        objective_next = float(problem.value)

        final_residual = _step_residual(
            instance=instance,
            x_prev=x_prev,
            alpha_prev=alpha_prev,
            objective_prev=objective_prev,
            x_next=x_next,
            alpha_next=alpha_next,
            theta_next=theta_next,
            objective_next=objective_next,
            alpha_total=alpha,
        )
        iterations_completed = iteration

        x_prev = x_next
        alpha_prev = alpha_next
        theta_prev = theta_next
        objective_prev = objective_next

        if final_residual <= tol:
            break

    runtime = time.perf_counter() - start

    if final_residual > tol:
        raise RuntimeError(
            "MM solver failed to converge within "
            f"{max_iter} iterations: final_residual={final_residual:.6e}, tol={tol:.6e}"
        )

    return OptimizedBernsteinMMResult(
        x=x_prev,
        objective=objective_prev,
        alpha=alpha_prev,
        theta=theta_prev,
        iterations=iterations_completed,
        runtime=float(runtime),
        final_residual=float(final_residual),
        solver_status=solver_status,
    )
