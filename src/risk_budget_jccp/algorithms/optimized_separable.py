from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_TINY = np.finfo(float).tiny
_LOG_MAX_FLOAT = math.log(np.finfo(float).max)
_OPEN_LOWER = 1e-15
_ONE_MINUS = np.nextafter(1.0, 0.0)
_BERNSTEIN_CRITICAL_ALPHA = math.exp(-0.5)
_BERNSTEIN_LOWER_UPPER = np.nextafter(_BERNSTEIN_CRITICAL_ALPHA, 0.0)
_BERNSTEIN_UPPER_LOWER = np.nextafter(_BERNSTEIN_CRITICAL_ALPHA, 1.0)
_BERNSTEIN_THRESHOLD_FACTOR = math.sqrt(math.e)
_CANTELLI_CRITICAL_ALPHA = 0.75
_CANTELLI_LOWER_UPPER = np.nextafter(_CANTELLI_CRITICAL_ALPHA, 0.0)
_CANTELLI_UPPER_LOWER = np.nextafter(_CANTELLI_CRITICAL_ALPHA, 1.0)
_CANTELLI_THRESHOLD_FACTOR = 8.0 * math.sqrt(3.0) / 9.0
_BISECTION_EPS = 1e-12


def _validate_inputs(weights: np.ndarray, alpha: float) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weights_arr.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(weights_arr <= 0.0):
        raise ValueError("weights must be strictly positive")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    return weights_arr


def _bisect_root(
    func: callable,
    lower: float,
    upper: float,
    iterations: int = 120,
) -> float:
    f_lower = func(lower)
    f_upper = func(upper)
    if abs(f_lower) <= _BISECTION_EPS:
        return lower
    if abs(f_upper) <= _BISECTION_EPS:
        return upper
    if np.signbit(f_lower) == np.signbit(f_upper):
        raise ValueError("bisection interval does not bracket a root")

    left = lower
    right = upper
    for _ in range(iterations):
        midpoint = 0.5 * (left + right)
        f_mid = func(midpoint)
        if abs(f_mid) <= _BISECTION_EPS:
            return midpoint
        if np.signbit(f_lower) != np.signbit(f_mid):
            right = midpoint
        else:
            left = midpoint
            f_lower = f_mid
    return 0.5 * (left + right)


def _bernstein_stationarity_rhs(weight: float, allocation: float) -> float:
    log_inverse_allocation = -math.log(allocation)
    log_rhs = (
        math.log(weight)
        - math.log(allocation)
        - 0.5 * math.log(2.0 * log_inverse_allocation)
    )
    if log_rhs >= _LOG_MAX_FLOAT:
        return math.inf
    return math.exp(log_rhs)


def _bernstein_threshold(weight: float) -> float:
    return weight * _BERNSTEIN_THRESHOLD_FACTOR


def _solve_bernstein_lower_coordinate(weight: float, dual: float) -> float:
    threshold = _bernstein_threshold(weight)
    if dual <= threshold * (1.0 + _BISECTION_EPS):
        return _BERNSTEIN_CRITICAL_ALPHA

    return _bisect_root(
        lambda allocation: _bernstein_stationarity_rhs(weight, allocation) - dual,
        _TINY,
        _BERNSTEIN_LOWER_UPPER,
    )


def _solve_bernstein_upper_coordinate(weight: float, dual: float) -> float:
    threshold = _bernstein_threshold(weight)
    if dual <= threshold * (1.0 + _BISECTION_EPS):
        return _BERNSTEIN_CRITICAL_ALPHA

    return _bisect_root(
        lambda allocation: _bernstein_stationarity_rhs(weight, allocation) - dual,
        _BERNSTEIN_UPPER_LOWER,
        _ONE_MINUS,
    )


def solve_separable_bernstein(weights: np.ndarray, alpha: float) -> np.ndarray:
    weights_arr = _validate_inputs(weights, alpha)
    if weights_arr.size == 1:
        return np.array([alpha], dtype=float)

    heaviest_index = int(np.argmax(weights_arr))
    dual_lower = _bernstein_threshold(float(weights_arr[heaviest_index]))

    def lower_branch_allocations(dual: float) -> np.ndarray:
        return np.array(
            [_solve_bernstein_lower_coordinate(weight, dual) for weight in weights_arr],
            dtype=float,
        )

    lower_allocations = lower_branch_allocations(dual_lower)
    if float(lower_allocations.sum()) >= alpha:
        upper = max(1.0, dual_lower)
        while float(lower_branch_allocations(upper).sum()) > alpha:
            upper *= 2.0
        dual_star = _bisect_root(
            lambda dual: float(lower_branch_allocations(dual).sum()) - alpha,
            dual_lower,
            upper,
        )
        return lower_branch_allocations(dual_star)

    def mixed_branch_allocations(dual: float) -> np.ndarray:
        allocations = lower_branch_allocations(dual)
        allocations[heaviest_index] = _solve_bernstein_upper_coordinate(
            weights_arr[heaviest_index],
            dual,
        )
        return allocations

    upper = max(1.0, dual_lower)
    while float(mixed_branch_allocations(upper).sum()) < alpha:
        upper *= 2.0

    dual_star = _bisect_root(
        lambda dual: float(mixed_branch_allocations(dual).sum()) - alpha,
        dual_lower,
        upper,
    )
    return mixed_branch_allocations(dual_star)


def _cantelli_stationarity_rhs(weight: float, allocation: float) -> float:
    ratio = (1.0 - allocation) / allocation
    return weight / (2.0 * allocation * allocation * math.sqrt(ratio))


def _cantelli_threshold(weight: float) -> float:
    return weight * _CANTELLI_THRESHOLD_FACTOR


def _solve_cantelli_lower_coordinate(weight: float, dual: float) -> float:
    threshold = _cantelli_threshold(weight)
    if dual <= threshold * (1.0 + _BISECTION_EPS):
        return _CANTELLI_CRITICAL_ALPHA

    return _bisect_root(
        lambda allocation: _cantelli_stationarity_rhs(weight, allocation) - dual,
        _OPEN_LOWER,
        _CANTELLI_LOWER_UPPER,
    )


def _solve_cantelli_upper_coordinate(weight: float, dual: float) -> float:
    threshold = _cantelli_threshold(weight)
    if dual <= threshold * (1.0 + _BISECTION_EPS):
        return _CANTELLI_CRITICAL_ALPHA

    return _bisect_root(
        lambda allocation: _cantelli_stationarity_rhs(weight, allocation) - dual,
        _CANTELLI_UPPER_LOWER,
        _ONE_MINUS,
    )


def solve_separable_cantelli(weights: np.ndarray, alpha: float) -> np.ndarray:
    weights_arr = _validate_inputs(weights, alpha)
    if weights_arr.size == 1:
        return np.array([alpha], dtype=float)

    heaviest_index = int(np.argmax(weights_arr))
    dual_lower = _cantelli_threshold(float(weights_arr[heaviest_index]))

    def lower_branch_allocations(dual: float) -> np.ndarray:
        return np.array(
            [_solve_cantelli_lower_coordinate(weight, dual) for weight in weights_arr],
            dtype=float,
        )

    lower_allocations = lower_branch_allocations(dual_lower)
    if float(lower_allocations.sum()) >= alpha:
        upper = max(1.0, dual_lower)
        while float(lower_branch_allocations(upper).sum()) > alpha:
            upper *= 2.0
        dual_star = _bisect_root(
            lambda dual: float(lower_branch_allocations(dual).sum()) - alpha,
            dual_lower,
            upper,
        )
        return lower_branch_allocations(dual_star)

    def mixed_branch_allocations(dual: float) -> np.ndarray:
        allocations = lower_branch_allocations(dual)
        allocations[heaviest_index] = _solve_cantelli_upper_coordinate(
            weights_arr[heaviest_index],
            dual,
        )
        return allocations

    upper = max(1.0, dual_lower)
    while float(mixed_branch_allocations(upper).sum()) < alpha:
        upper *= 2.0

    dual_star = _bisect_root(
        lambda dual: float(mixed_branch_allocations(dual).sum()) - alpha,
        dual_lower,
        upper,
    )
    return mixed_branch_allocations(dual_star)


def _normal_cvar_marginal(allocation: float) -> float:
    threshold = float(norm.ppf(1.0 - allocation))
    density = float(norm.pdf(threshold))
    return (density - allocation * threshold) / (allocation * allocation)


def solve_separable_normal_cvar(weights: np.ndarray, alpha: float) -> np.ndarray:
    """Solve the Gaussian CVaR resource-allocation problem by KKT bisection."""
    weights_arr = _validate_inputs(weights, alpha)
    if weights_arr.size == 1:
        return np.array([alpha], dtype=float)

    lower_allocation = max(_OPEN_LOWER, np.finfo(float).eps)

    def coordinate(weight: float, dual: float) -> float:
        if weight * _normal_cvar_marginal(alpha) >= dual:
            return alpha
        return brentq(
            lambda allocation: weight * _normal_cvar_marginal(allocation) - dual,
            lower_allocation,
            alpha,
            xtol=1e-14,
            rtol=1e-13,
            maxiter=120,
        )

    def allocations(dual: float) -> np.ndarray:
        return np.array([coordinate(float(weight), dual) for weight in weights_arr])

    dual_lower = min(float(weight) * _normal_cvar_marginal(alpha) for weight in weights_arr)
    dual_upper = max(1.0, dual_lower)
    while float(allocations(dual_upper).sum()) > alpha:
        dual_upper *= 2.0

    dual_star = brentq(
        lambda dual: float(allocations(dual).sum()) - alpha,
        dual_lower,
        dual_upper,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=120,
    )
    result = allocations(dual_star)
    result *= alpha / float(result.sum())
    return result
