from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.stats import norm


_MIN_POSITIVE = 1e-12
_DEFAULT_DEMAND_LEVEL = 1.0
_DEFAULT_SIGMA_SCALE = 1.0
_CVXPY_SOLVER = cp.CLARABEL
_VALID_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _read_only_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    read_only = np.array(array, copy=True)
    read_only.setflags(write=False)
    return read_only


def _read_only_matrix(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    read_only = np.array(array, copy=True)
    read_only.setflags(write=False)
    return read_only


def _validate_alpha_vec(alpha_vec: np.ndarray, num_constraints: int) -> np.ndarray:
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    if alpha_arr.shape != (num_constraints,):
        raise ValueError(f"alpha_vec must have shape ({num_constraints},)")
    if not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha_vec must be finite")
    if np.any(alpha_arr <= 0.0) or np.any(alpha_arr >= 1.0):
        raise ValueError("alpha_vec must satisfy 0 < alpha_i < 1")
    return alpha_arr


def _solve_problem(problem: cp.Problem) -> str:
    problem.solve(solver=_CVXPY_SOLVER, warm_start=True)
    status = str(problem.status)
    if problem.status not in _VALID_STATUSES:
        raise RuntimeError(f"optimization failed with status {status}")
    return status


@dataclass(frozen=True, slots=True)
class SyntheticCapacityInstance:
    dimension: int
    num_constraints: int
    heterogeneity: float
    seed: int
    cost: np.ndarray
    constraint_matrix: np.ndarray
    demand: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.num_constraints <= 0:
            raise ValueError("num_constraints must be positive")
        if self.heterogeneity < 0.0:
            raise ValueError("heterogeneity must be nonnegative")

        cost = _read_only_vector(self.cost, self.dimension, "cost")
        constraint_matrix = _read_only_matrix(
            self.constraint_matrix,
            (self.num_constraints, self.dimension),
            "constraint_matrix",
        )
        demand = _read_only_vector(self.demand, self.num_constraints, "demand")
        sigma = _read_only_vector(self.sigma, self.num_constraints, "sigma")

        if np.any(cost <= 0.0):
            raise ValueError("cost must be strictly positive")
        if np.any(constraint_matrix < 0.0):
            raise ValueError("constraint_matrix must be nonnegative")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be strictly positive")

        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "constraint_matrix", constraint_matrix)
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True, slots=True)
class FixedBernsteinCapacitySolution:
    x: np.ndarray
    objective: float
    alpha: np.ndarray
    theta: np.ndarray
    solver_status: str


def bernstein_quantile(alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_vec must be one-dimensional")
    if np.any(alpha_arr <= 0.0) or np.any(alpha_arr >= 1.0):
        raise ValueError("alpha_vec must satisfy 0 < alpha_i < 1")
    return np.sqrt(2.0 * np.log(1.0 / alpha_arr))


def initialize_theta(instance: SyntheticCapacityInstance, alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = _validate_alpha_vec(alpha_vec, instance.num_constraints)
    theta = instance.sigma / bernstein_quantile(alpha_arr)
    return np.maximum(theta, _MIN_POSITIVE)


def solve_fixed_bernstein(
    instance: SyntheticCapacityInstance,
    alpha_vec: np.ndarray,
) -> FixedBernsteinCapacitySolution:
    alpha_arr = _validate_alpha_vec(alpha_vec, instance.num_constraints)

    x = cp.Variable(instance.dimension, nonneg=True)
    rhs = instance.demand + instance.sigma * bernstein_quantile(alpha_arr)
    problem = cp.Problem(
        cp.Minimize(instance.cost @ x),
        [instance.constraint_matrix @ x >= rhs],
    )
    status = _solve_problem(problem)

    x_value = np.asarray(x.value, dtype=float).reshape(instance.dimension)
    return FixedBernsteinCapacitySolution(
        x=x_value,
        objective=float(problem.value),
        alpha=np.asarray(alpha_arr, dtype=float),
        theta=initialize_theta(instance, alpha_arr),
        solver_status=status,
    )


def exact_gaussian_joint_violation(
    instance: SyntheticCapacityInstance,
    x: np.ndarray,
) -> float:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.shape != (instance.dimension,):
        raise ValueError(f"x must have shape ({instance.dimension},)")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite")

    margins = (instance.constraint_matrix @ x_arr - instance.demand) / instance.sigma
    scalar_satisfaction = norm.cdf(margins)
    joint_satisfaction = float(np.prod(scalar_satisfaction, dtype=float))
    return float(np.clip(1.0 - joint_satisfaction, 0.0, 1.0))


def make_capacity_instance(
    dimension: int,
    num_constraints: int,
    heterogeneity: float,
    seed: int,
    *,
    demand_level: float = _DEFAULT_DEMAND_LEVEL,
    sigma_scale: float = _DEFAULT_SIGMA_SCALE,
) -> SyntheticCapacityInstance:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if num_constraints <= 0:
        raise ValueError("num_constraints must be positive")
    if heterogeneity < 0.0:
        raise ValueError("heterogeneity must be nonnegative")
    if demand_level <= 0.0:
        raise ValueError("demand_level must be positive")
    if sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be positive")

    rng = np.random.default_rng(seed)

    constraint_matrix = rng.uniform(0.25, 1.25, size=(num_constraints, dimension))
    constraint_matrix = constraint_matrix / constraint_matrix.mean(axis=1, keepdims=True)

    cost = rng.uniform(0.75, 1.25, size=dimension)
    cost = cost / cost.mean()

    demand = np.full(num_constraints, demand_level, dtype=float)

    sigma = rng.lognormal(mean=0.0, sigma=heterogeneity, size=num_constraints)
    sigma = sigma / sigma.mean()
    sigma = sigma_scale * sigma

    return SyntheticCapacityInstance(
        dimension=dimension,
        num_constraints=num_constraints,
        heterogeneity=heterogeneity,
        seed=seed,
        cost=cost,
        constraint_matrix=constraint_matrix,
        demand=demand,
        sigma=sigma,
    )
