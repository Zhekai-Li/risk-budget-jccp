import numpy as np
import pytest

from risk_budget_jccp import __version__
from risk_budget_jccp.models.synthetic_service import (
    SyntheticServiceInstance,
    exact_gaussian_joint_violation,
    make_service_instance,
    normal_cvar_quantile,
)


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_make_service_instance_is_reproducible() -> None:
    lhs = make_service_instance(m=5, heterogeneity=1.0, seed=3)
    rhs = make_service_instance(m=5, heterogeneity=1.0, seed=3)
    assert np.allclose(lhs.weights, rhs.weights)


def test_weights_are_normalized_to_dimension() -> None:
    instance = make_service_instance(m=10, heterogeneity=0.5, seed=1)
    assert np.isclose(instance.weights.mean(), 1.0, atol=1e-8)


@pytest.mark.parametrize("m", [0, -1])
def test_make_service_instance_rejects_nonpositive_dimension(m: int) -> None:
    with pytest.raises(ValueError, match="m must be positive"):
        make_service_instance(m=m, heterogeneity=0.5, seed=1)


def test_weights_are_explicitly_read_only() -> None:
    instance = make_service_instance(m=5, heterogeneity=1.0, seed=3)

    assert isinstance(instance, SyntheticServiceInstance)
    assert instance.weights.flags.writeable is False

    with pytest.raises(ValueError):
        instance.weights[0] = 0.0


def test_normal_cvar_joint_violation_is_below_total_budget() -> None:
    alpha_vec = np.full(10, 0.005)
    assert np.all(normal_cvar_quantile(alpha_vec) > 0.0)
    assert 0.0 < exact_gaussian_joint_violation(alpha_vec, "cvar") < 0.05
