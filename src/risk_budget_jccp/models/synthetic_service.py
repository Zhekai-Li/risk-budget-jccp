from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class SyntheticServiceInstance:
    m: int
    heterogeneity: float
    seed: int
    weights: np.ndarray


def _validate_alpha_vec(alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alpha_vec, dtype=float)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_vec must be one-dimensional")
    if np.any(alpha_arr <= 0.0) or np.any(alpha_arr > 1.0):
        raise ValueError("alpha_vec must satisfy 0 < alpha_i <= 1")
    return alpha_arr


def bernstein_quantile(alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = _validate_alpha_vec(alpha_vec)
    return np.sqrt(2.0 * np.log(1.0 / alpha_arr))


def cantelli_quantile(alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = _validate_alpha_vec(alpha_vec)
    return np.sqrt((1.0 - alpha_arr) / alpha_arr)


def normal_cvar_quantile(alpha_vec: np.ndarray) -> np.ndarray:
    alpha_arr = _validate_alpha_vec(alpha_vec)
    threshold = norm.ppf(1.0 - alpha_arr)
    return norm.pdf(threshold) / alpha_arr


def exact_gaussian_joint_violation(alpha_vec: np.ndarray, surrogate: str) -> float:
    alpha_arr = _validate_alpha_vec(alpha_vec)
    if surrogate == "bernstein":
        quantiles = bernstein_quantile(alpha_arr)
    elif surrogate == "cantelli":
        quantiles = cantelli_quantile(alpha_arr)
    elif surrogate == "cvar":
        quantiles = normal_cvar_quantile(alpha_arr)
    else:
        raise ValueError("surrogate must be one of {'bernstein', 'cantelli', 'cvar'}")
    scalar_violation = norm.sf(quantiles)
    return float(-np.expm1(np.log1p(-scalar_violation).sum()))


def service_objective(weights: np.ndarray, alpha_vec: np.ndarray, surrogate: str) -> float:
    weights_arr = np.asarray(weights, dtype=float)
    alpha_arr = _validate_alpha_vec(alpha_vec)
    if weights_arr.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weights_arr.shape != alpha_arr.shape:
        raise ValueError("weights and alpha_vec must have the same shape")

    if surrogate == "bernstein":
        quantiles = bernstein_quantile(alpha_arr)
    elif surrogate == "cantelli":
        quantiles = cantelli_quantile(alpha_arr)
    elif surrogate == "cvar":
        quantiles = normal_cvar_quantile(alpha_arr)
    else:
        raise ValueError("surrogate must be one of {'bernstein', 'cantelli', 'cvar'}")

    return float(np.dot(weights_arr, quantiles))


def make_service_instance(m: int, heterogeneity: float, seed: int) -> SyntheticServiceInstance:
    if m <= 0:
        raise ValueError("m must be positive")

    rng = np.random.default_rng(seed)
    component_sigma = heterogeneity / np.sqrt(2.0)
    cost_factor = rng.lognormal(mean=0.0, sigma=component_sigma, size=m)
    uncertainty_factor = rng.lognormal(mean=0.0, sigma=component_sigma, size=m)
    weights = cost_factor * uncertainty_factor
    weights = weights / weights.mean()
    weights = np.array(weights, copy=True)
    weights.setflags(write=False)
    return SyntheticServiceInstance(
        m=m,
        heterogeneity=heterogeneity,
        seed=seed,
        weights=weights,
    )
