import numpy as np
import pytest
from scipy.stats import norm
import time

from risk_budget_jccp.algorithms.equal_allocation import equal_allocation
import risk_budget_jccp.algorithms.optimized_bernstein_mm as optimized_bernstein_mm
from risk_budget_jccp.algorithms.optimized_bernstein_mm import (
    OptimizedBernsteinMMResult,
    solve_coupled_bernstein_mm,
)
from risk_budget_jccp.models.synthetic_capacity import (
    SyntheticCapacityInstance,
    exact_gaussian_joint_violation,
    make_capacity_instance,
    solve_fixed_bernstein,
)


def test_make_capacity_instance_is_reproducible_and_has_expected_shapes() -> None:
    lhs = make_capacity_instance(
        dimension=3,
        num_constraints=5,
        heterogeneity=0.0,
        seed=7,
    )
    rhs = make_capacity_instance(
        dimension=3,
        num_constraints=5,
        heterogeneity=0.0,
        seed=7,
    )

    assert isinstance(lhs, SyntheticCapacityInstance)
    assert lhs.constraint_matrix.shape == (5, 3)
    assert lhs.cost.shape == (3,)
    assert lhs.demand.shape == (5,)
    assert lhs.sigma.shape == (5,)
    assert np.allclose(lhs.constraint_matrix, rhs.constraint_matrix)
    assert np.allclose(lhs.cost, rhs.cost)
    assert np.allclose(lhs.demand, rhs.demand)
    assert np.allclose(lhs.sigma, rhs.sigma)
    assert np.allclose(lhs.sigma, np.full(5, lhs.sigma[0]))
    assert lhs.constraint_matrix.flags.writeable is False
    assert lhs.cost.flags.writeable is False
    assert lhs.demand.flags.writeable is False
    assert lhs.sigma.flags.writeable is False


def test_exact_gaussian_joint_violation_matches_independent_formula() -> None:
    instance = SyntheticCapacityInstance(
        dimension=2,
        num_constraints=2,
        heterogeneity=0.0,
        seed=0,
        cost=np.array([1.0, 1.0]),
        constraint_matrix=np.eye(2),
        demand=np.zeros(2),
        sigma=np.ones(2),
    )
    x = np.array([0.0, 1.0])

    violation = exact_gaussian_joint_violation(instance, x)

    expected = 1.0 - norm.cdf(0.0) * norm.cdf(1.0)
    assert np.isclose(violation, expected, atol=1e-10)


def test_optimized_coupled_bernstein_returns_diagnostics_and_beats_equal() -> None:
    instance = make_capacity_instance(
        dimension=3,
        num_constraints=5,
        heterogeneity=1.0,
        seed=11,
    )
    alpha_total = 0.05
    alpha_equal = equal_allocation(m=instance.num_constraints, alpha=alpha_total)
    equal_solution = solve_fixed_bernstein(instance, alpha_equal)

    result = solve_coupled_bernstein_mm(
        instance,
        alpha=alpha_total,
        max_iter=25,
        tol=1e-5,
    )

    assert isinstance(result, OptimizedBernsteinMMResult)
    assert result.x.shape == (instance.dimension,)
    assert result.theta.shape == (instance.num_constraints,)
    assert result.alpha.shape == (instance.num_constraints,)
    assert np.isclose(result.alpha.sum(), alpha_total, atol=1e-8)
    assert np.all(result.alpha > 0.0)
    assert result.objective <= equal_solution.objective + 1e-6
    assert result.iterations >= 1
    assert np.isfinite(result.runtime)
    assert np.isfinite(result.final_residual)
    assert result.final_residual >= 0.0
    assert result.final_residual <= 1e-5
    assert result.solver_status in {"optimal", "optimal_inaccurate"}
    assert exact_gaussian_joint_violation(instance, result.x) <= alpha_total + 1e-8


def test_optimized_coupled_bernstein_raises_on_non_convergence() -> None:
    instance = make_capacity_instance(
        dimension=3,
        num_constraints=5,
        heterogeneity=1.0,
        seed=11,
    )

    with pytest.raises(RuntimeError, match="failed to converge"):
        solve_coupled_bernstein_mm(
            instance,
            alpha=0.05,
            max_iter=1,
            tol=1e-12,
        )


def test_optimized_coupled_bernstein_runtime_includes_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = make_capacity_instance(
        dimension=2,
        num_constraints=2,
        heterogeneity=0.2,
        seed=3,
    )
    delay_seconds = 0.2
    original_solve_fixed_bernstein = optimized_bernstein_mm.solve_fixed_bernstein

    def delayed_solve_fixed_bernstein(*args, **kwargs):
        time.sleep(delay_seconds)
        return original_solve_fixed_bernstein(*args, **kwargs)

    monkeypatch.setattr(
        optimized_bernstein_mm,
        "solve_fixed_bernstein",
        delayed_solve_fixed_bernstein,
    )

    result = solve_coupled_bernstein_mm(
        instance,
        alpha=0.05,
        max_iter=20,
        tol=1e-4,
    )

    assert result.runtime >= delay_seconds * 0.9
