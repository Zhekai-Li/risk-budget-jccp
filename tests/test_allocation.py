import warnings

import numpy as np
import pytest

from risk_budget_jccp.algorithms.equal_allocation import equal_allocation
from risk_budget_jccp.algorithms.optimized_separable import (
    solve_separable_bernstein,
    solve_separable_cantelli,
    solve_separable_normal_cvar,
)
from risk_budget_jccp.models.synthetic_service import make_service_instance, service_objective


def _coarse_bruteforce_bernstein(
    weights: np.ndarray,
    alpha: float,
    grid_size: int = 151,
) -> float:
    lower = 1e-4
    best = np.inf
    for alpha_0 in np.linspace(lower, alpha - 2.0 * lower, grid_size):
        alpha_1_max = alpha - alpha_0 - lower
        if alpha_1_max <= lower:
            continue
        alpha_1_values = np.linspace(lower, alpha_1_max, grid_size)
        alpha_2_values = alpha - alpha_0 - alpha_1_values
        feasible = alpha_2_values > lower
        if not np.any(feasible):
            continue
        for alpha_1, alpha_2 in zip(alpha_1_values[feasible], alpha_2_values[feasible], strict=False):
            alpha_vec = np.array([alpha_0, alpha_1, alpha_2])
            best = min(best, service_objective(weights, alpha_vec, surrogate="bernstein"))
    return float(best)


def test_equal_allocation_sums_to_alpha() -> None:
    alpha_vec = equal_allocation(m=5, alpha=0.05)
    assert np.isclose(alpha_vec.sum(), 0.05)


def test_equal_allocation_is_uniform() -> None:
    alpha_vec = equal_allocation(m=4, alpha=0.2)
    assert np.allclose(alpha_vec, np.full(4, 0.05))


@pytest.mark.parametrize(
    ("m", "alpha"),
    [
        (0, 0.05),
        (5, 0.0),
        (5, -0.01),
        (5, 1.01),
    ],
)
def test_equal_allocation_validates_inputs(m: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        equal_allocation(m=m, alpha=alpha)


def test_optimized_bernstein_matches_equal_when_weights_identical() -> None:
    weights = np.ones(6)
    alpha_star = solve_separable_bernstein(weights=weights, alpha=0.05)
    assert np.allclose(alpha_star, equal_allocation(6, 0.05), atol=1e-6)
    assert np.isclose(alpha_star.sum(), 0.05, atol=1e-10)


def test_optimized_bernstein_satisfies_budget_and_beats_equal_objective() -> None:
    instance = make_service_instance(m=8, heterogeneity=0.6, seed=4)
    alpha_eq = equal_allocation(8, 0.05)
    alpha_star = solve_separable_bernstein(weights=instance.weights, alpha=0.05)

    assert np.isclose(alpha_star.sum(), 0.05, atol=1e-8)

    equal_obj = service_objective(instance.weights, alpha_eq, surrogate="bernstein")
    optimized_obj = service_objective(instance.weights, alpha_star, surrogate="bernstein")
    assert optimized_obj <= equal_obj + 1e-8


def test_optimized_bernstein_matches_coarse_bruteforce_on_asymmetric_case() -> None:
    weights = np.array([0.4, 1.1, 2.3])
    alpha_star = solve_separable_bernstein(weights=weights, alpha=0.35)

    assert np.isclose(alpha_star.sum(), 0.35, atol=1e-8)

    optimized_obj = service_objective(weights, alpha_star, surrogate="bernstein")
    coarse_best = _coarse_bruteforce_bernstein(weights, 0.35)
    assert optimized_obj <= coarse_best + 5e-2


def test_optimized_bernstein_single_coordinate_high_alpha_matches_trivial_optimum() -> None:
    weights = np.array([2.0])
    alpha_star = solve_separable_bernstein(weights=weights, alpha=0.95)
    trivial_alpha = np.array([0.95])

    assert np.isclose(alpha_star.sum(), 0.95, atol=1e-10)
    assert np.allclose(alpha_star, trivial_alpha, atol=1e-10)

    optimized_obj = service_objective(weights, alpha_star, surrogate="bernstein")
    trivial_obj = service_objective(weights, trivial_alpha, surrogate="bernstein")
    assert np.isclose(optimized_obj, trivial_obj, atol=1e-10)


def test_optimized_bernstein_high_alpha_satisfies_budget_and_beats_equal_objective() -> None:
    weights = np.array([0.5, 1.5, 6.0])
    alpha_eq = equal_allocation(3, 0.95)
    alpha_star = solve_separable_bernstein(weights=weights, alpha=0.95)

    assert np.isclose(alpha_star.sum(), 0.95, atol=1e-8)

    equal_obj = service_objective(weights, alpha_eq, surrogate="bernstein")
    optimized_obj = service_objective(weights, alpha_star, surrogate="bernstein")
    assert optimized_obj <= equal_obj + 1e-8


def test_optimized_bernstein_avoids_overflow_warning_for_large_weight_skew() -> None:
    weights = np.array([1e6, 1.0, 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alpha_star = solve_separable_bernstein(weights=weights, alpha=0.05)

    assert np.all(np.isfinite(alpha_star))
    assert np.isclose(alpha_star.sum(), 0.05, atol=1e-8)
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_optimized_cantelli_dominates_equal_objective() -> None:
    instance = make_service_instance(m=8, heterogeneity=1.0, seed=4)
    alpha_eq = equal_allocation(8, 0.05)
    alpha_ora = solve_separable_cantelli(weights=instance.weights, alpha=0.05)
    equal_obj = service_objective(instance.weights, alpha_eq, surrogate="cantelli")
    optimized_obj = service_objective(instance.weights, alpha_ora, surrogate="cantelli")
    assert optimized_obj <= equal_obj + 1e-8


def test_optimized_cantelli_satisfies_budget_on_asymmetric_high_alpha_case() -> None:
    weights = np.array([0.3, 1.0, 4.0])
    alpha_star = solve_separable_cantelli(weights=weights, alpha=0.95)
    assert np.isclose(alpha_star.sum(), 0.95, atol=1e-8)


def test_optimized_cantelli_handles_upper_branch_high_alpha_case() -> None:
    weights = np.array([1.0, 100.0])
    alpha_eq = equal_allocation(2, 0.95)
    alpha_star = solve_separable_cantelli(weights=weights, alpha=0.95)

    assert np.isclose(alpha_star.sum(), 0.95, atol=1e-8)
    assert alpha_star[1] > 0.75

    equal_obj = service_objective(weights, alpha_eq, surrogate="cantelli")
    optimized_obj = service_objective(weights, alpha_star, surrogate="cantelli")
    assert optimized_obj <= equal_obj + 1e-8


def test_optimized_normal_cvar_matches_equal_when_weights_identical() -> None:
    weights = np.ones(6)
    alpha_star = solve_separable_normal_cvar(weights=weights, alpha=0.05)
    assert np.allclose(alpha_star, equal_allocation(6, 0.05), atol=1e-8)


def test_optimized_normal_cvar_satisfies_budget_and_beats_equal_objective() -> None:
    weights = np.array([0.4, 1.1, 2.3])
    alpha_equal = equal_allocation(3, 0.05)
    alpha_star = solve_separable_normal_cvar(weights=weights, alpha=0.05)
    assert np.isclose(alpha_star.sum(), 0.05, atol=1e-10)
    assert service_objective(weights, alpha_star, "cvar") <= service_objective(
        weights, alpha_equal, "cvar"
    )
