from __future__ import annotations

import numpy as np

from risk_budget_jccp.real_data.common.certificates import (
    bernstein_quantile,
    cantelli_quantile,
    empirical_cvar,
    validate_cantelli_budget,
)


def test_cvar_certificate_feasibility_for_tail_quantile() -> None:
    losses = np.array([-2.0, -1.0, 0.0, 1.0])
    assert empirical_cvar(losses, 0.25) == 1.0


def test_bernstein_theta_feasibility_expression() -> None:
    alpha = np.array([0.05])
    theta = np.array([1.0])
    mu = np.array([-4.0])
    sigma = np.array([1.0])
    cert = mu + 0.5 * sigma**2 / theta - theta * np.log(alpha)
    assert cert[0] < 0.0
    assert bernstein_quantile(alpha)[0] > 0.0


def test_cantelli_budget_validation() -> None:
    ok, used, beta = validate_cantelli_budget(
        mu=np.array([-10.0, -12.0]),
        sigma=np.array([1.0, 1.0]),
        alpha_vec=np.array([0.02, 0.02]),
        tol=1.0e-9,
    )
    assert ok
    assert used < 0.04
    assert np.all(beta > 0.0)
    assert cantelli_quantile(np.array([0.05]))[0] > 0.0
