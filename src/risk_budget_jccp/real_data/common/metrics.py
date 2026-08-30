from __future__ import annotations

import math

import numpy as np


DEFAULT_RELATIVE_VIOLATION_TOL = 1.0e-6
DEFAULT_ABSOLUTE_VIOLATION_TOL = 1.0e-12


def normalized_entropy(alpha_vec: np.ndarray) -> float:
    alpha = np.asarray(alpha_vec, dtype=float)
    if alpha.ndim != 1 or alpha.size == 0:
        raise ValueError("alpha_vec must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(alpha)) or np.any(alpha < 0.0):
        raise ValueError("alpha_vec must be finite and nonnegative")
    total = float(alpha.sum())
    if total <= 0.0:
        raise ValueError("alpha_vec must have positive total mass")
    if alpha.size == 1:
        return 1.0
    p = alpha[alpha > 0.0] / total
    return -float(np.sum(p * np.log(p))) / math.log(alpha.size)


def max_budget_share(alpha_vec: np.ndarray) -> float:
    alpha = np.asarray(alpha_vec, dtype=float)
    total = float(alpha.sum())
    if total <= 0.0:
        raise ValueError("alpha_vec must have positive total mass")
    return float(np.max(alpha) / total)


def relative_improvement(equal_objective: float, optimized_objective: float) -> float:
    if not np.isfinite(equal_objective) or not np.isfinite(optimized_objective):
        raise ValueError("objectives must be finite")
    denom = abs(float(equal_objective))
    if denom <= 0.0:
        return 0.0
    return float((equal_objective - optimized_objective) / denom)


def violation_metrics(
    y_eval: np.ndarray,
    *,
    reference_scale: np.ndarray | float | None = None,
    relative_tol: float = DEFAULT_RELATIVE_VIOLATION_TOL,
    absolute_tol: float = DEFAULT_ABSOLUTE_VIOLATION_TOL,
) -> dict[str, float | int | list[int]]:
    """Compute empirical violations above a scale-aware numerical tolerance.

    ``y_eval`` has one column per scalar constraint.  A positive value is
    counted only when it exceeds ``absolute_tol + relative_tol * scale_i``.
    The caller supplies ``scale_i`` from the physical quantity defining that
    constraint (for example, demand or a margin buffer).  This prevents a
    solver residual expressed in MW, units of demand, or portfolio return from
    being interpreted as a probabilistic violation.
    """
    y = np.asarray(y_eval, dtype=float)
    if y.ndim != 2:
        raise ValueError("y_eval must have shape (n_scenarios, n_constraints)")
    if relative_tol < 0.0 or absolute_tol < 0.0:
        raise ValueError("violation tolerances must be nonnegative")
    if reference_scale is None:
        scale = np.max(np.abs(y), axis=0)
    else:
        scale = np.asarray(reference_scale, dtype=float)
        if scale.ndim == 0:
            scale = np.full(y.shape[1], float(scale), dtype=float)
        if scale.shape != (y.shape[1],):
            raise ValueError("reference_scale must be scalar or have one value per scalar constraint")
    if np.any(~np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("reference_scale must be finite and nonnegative")
    tolerance = absolute_tol + relative_tol * scale
    mask = y > tolerance[None, :]
    counts = mask.sum(axis=1).astype(int)
    return {
        "empirical_joint_violation": float(np.mean(counts > 0)),
        "average_scalar_violations": float(np.mean(counts)),
        "max_scalar_violations": int(counts.max(initial=0)),
        "violation_counts": counts.tolist(),
        "scalar_violation_rates": mask.mean(axis=0).tolist(),
        "violation_tolerance": tolerance.tolist(),
        "max_violation_tolerance": float(np.max(tolerance, initial=absolute_tol)),
    }


def alpha_metrics(alpha_vec: np.ndarray) -> dict[str, float]:
    return {
        "max_budget_share": max_budget_share(alpha_vec),
        "normalized_entropy": normalized_entropy(alpha_vec),
    }
