from __future__ import annotations

import numpy as np


def equal_allocation(m: int, alpha: float, *, min_alpha: float = 0.0) -> np.ndarray:
    if m <= 0:
        raise ValueError("m must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    allocation = np.full(m, alpha / m, dtype=float)
    if min_alpha > 0.0 and np.any(allocation < min_alpha):
        raise ValueError("alpha/m is below min_alpha")
    return allocation


def cantelli_quantile(alpha: np.ndarray | float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if np.any((a <= 0.0) | (a >= 1.0)):
        raise ValueError("Cantelli alpha values must satisfy 0 < alpha < 1")
    return np.sqrt((1.0 - a) / a)


def bernstein_quantile(alpha: np.ndarray | float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if np.any((a <= 0.0) | (a >= 1.0)):
        raise ValueError("Bernstein alpha values must satisfy 0 < alpha < 1")
    return np.sqrt(2.0 * np.log(1.0 / a))


def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(losses, dtype=float))[::-1]
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty vector")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    tail_mass = alpha * values.size
    full = int(np.floor(tail_mass))
    frac = tail_mass - full
    if full <= 0:
        return float(values[0])
    total = float(values[:full].sum())
    if full < values.size:
        total += float(frac * values[full])
    return total / max(tail_mass, 1.0e-12)


def validate_cantelli_budget(mu: np.ndarray, sigma: np.ndarray, alpha_vec: np.ndarray, tol: float) -> tuple[bool, float, np.ndarray]:
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    if mu_arr.shape != sigma_arr.shape or mu_arr.shape != alpha_arr.shape:
        raise ValueError("mu, sigma, and alpha_vec must have matching shape")
    beta = np.ones_like(mu_arr, dtype=float)
    zero_risk_mask = (sigma_arr <= float(tol)) & (mu_arr <= float(tol))
    beta[zero_risk_mask] = 0.0
    safe_mask = mu_arr < -float(tol)
    denom = sigma_arr[safe_mask] ** 2 + np.maximum(-mu_arr[safe_mask], 0.0) ** 2
    beta[safe_mask] = np.divide(
        sigma_arr[safe_mask] ** 2,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0.0,
    )
    used = float(beta.sum())
    return bool(used <= float(alpha_arr.sum()) + tol), used, beta
