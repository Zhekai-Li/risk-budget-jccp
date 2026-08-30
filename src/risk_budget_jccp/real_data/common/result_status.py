from __future__ import annotations


DEFAULT_CERTIFICATE_ACCEPT_TOL = 1.0e-4
DEFAULT_CERTIFICATE_STRICT_TOL = 1.0e-6


def certificate_accept_tol(dca_cfg: dict[str, object] | None = None) -> float:
    if dca_cfg is None:
        return DEFAULT_CERTIFICATE_ACCEPT_TOL
    return float(dca_cfg.get("certificate_accept_tol", DEFAULT_CERTIFICATE_ACCEPT_TOL))


def certificate_strict_tol(dca_cfg: dict[str, object] | None = None) -> float:
    if dca_cfg is None:
        return DEFAULT_CERTIFICATE_STRICT_TOL
    return float(dca_cfg.get("certificate_strict_tol", dca_cfg.get("feasibility_tol", DEFAULT_CERTIFICATE_STRICT_TOL)))


def classify_result(
    *,
    allocation: str,
    valid_certificate: bool,
    valid_optimization: bool,
    fallback_used: bool,
    solver_status: str,
) -> str:
    status = str(solver_status)
    if status.startswith("failed") or not valid_certificate:
        return "infeasible" if "infeasible" in status else "failed"
    if "unsupported" in status or "diagnostic" in status:
        return "diagnostic_only"
    if fallback_used:
        return "fallback_equal"
    if allocation == "optimized":
        return "success" if valid_optimization else "diagnostic_only"
    return "success"


def algorithm_class(case: str, certificate: str, allocation: str, solver_status: str) -> str:
    status = str(solver_status)
    if status.startswith("failed"):
        return "failed_infeasible" if "infeasible" in status else "failed"
    if "unsupported" in status or "diagnostic" in status:
        return "diagnostic_only"
    if allocation == "equal":
        return "fixed_equal_certificate"
    if "fallback" in status:
        return "paper_dca_attempt_fallback" if certificate in {"cvar", "bernstein"} else "diagnostic_fallback"
    if case == "power" and certificate in {"cvar", "bernstein"}:
        return "paper_dca"
    if case in {"m5", "french"} and certificate in {"cvar", "cantelli"}:
        return "paper_separable_equivalent"
    if certificate == "bernstein":
        return "paper_dca"
    return "diagnostic_only"


def status_fields(
    *,
    case: str,
    certificate: str,
    allocation: str,
    solver_status: str,
    valid_certificate: bool,
    valid_optimization: bool,
    fallback_used: bool,
    feasibility_residual: float,
    calibration_joint_violation: float,
    alpha_total: float,
    dca_cfg: dict[str, object] | None = None,
) -> dict[str, object]:
    accept_tol = certificate_accept_tol(dca_cfg)
    strict_tol = certificate_strict_tol(dca_cfg)
    finite_sample_cvar = certificate == "cvar"
    certificate_accepted = bool(valid_certificate and float(feasibility_residual) <= accept_tol)
    calibration_jvp_within_budget = bool(float(calibration_joint_violation) <= float(alpha_total) + accept_tol)
    if not certificate_accepted:
        calibration_jvp_contract = "not_applicable_certificate_not_accepted"
    elif finite_sample_cvar and calibration_jvp_within_budget:
        calibration_jvp_contract = "finite_sample_cvar_bound_observed"
    elif finite_sample_cvar:
        calibration_jvp_contract = "finite_sample_cvar_bound_violated"
    else:
        calibration_jvp_contract = "not_implied_by_moment_certificate"
    return {
        "algorithm_class": algorithm_class(case, certificate, allocation, solver_status),
        "certificate_accept_tol": accept_tol,
        "certificate_strict_tol": strict_tol,
        "passes_certificate_acceptance": certificate_accepted,
        "passes_certificate_strict": bool(valid_certificate and float(feasibility_residual) <= strict_tol),
        "certificate_acceptance_status": "accepted" if certificate_accepted else "not_accepted",
        "calibration_jvp_bound_applies": finite_sample_cvar,
        "calibration_jvp_within_budget": calibration_jvp_within_budget,
        "calibration_jvp_contract": calibration_jvp_contract,
        "result_status": classify_result(
            allocation=allocation,
            valid_certificate=valid_certificate,
            valid_optimization=valid_optimization,
            fallback_used=fallback_used,
            solver_status=solver_status,
        ),
    }
