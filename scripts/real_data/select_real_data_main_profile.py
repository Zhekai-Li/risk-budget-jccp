from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.common.latex import write_latex_table
from risk_budget_jccp.real_data.common.paths import PROCESSED_ROOT, RESULTS_ROOT, runtime_results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sensitivity profiles and recommend main real-data profiles.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    return parser.parse_args()


def _summary_path(profile: str, case: str) -> Path:
    return RESULTS_ROOT / "sensitivity" / profile / "runs" / case / "tables" / f"{case}_summary.csv"


def _processed_path(profile: str, case: str, filename: str) -> Path:
    return PROCESSED_ROOT / "sensitivity" / profile / case / filename


def _row_value(frame: pd.DataFrame, certificate: str, allocation: str, column: str, default: float = np.nan) -> float:
    row = frame.loc[(frame["certificate"] == certificate) & (frame["allocation"] == allocation)]
    if row.empty or column not in row.columns:
        return float(default)
    value = row.iloc[0][column]
    try:
        return float(value)
    except Exception:
        return float(default)


def _row_status(frame: pd.DataFrame, certificate: str, allocation: str) -> str:
    row = frame.loc[(frame["certificate"] == certificate) & (frame["allocation"] == allocation)]
    if row.empty or "result_status" not in row.columns:
        return "missing"
    return str(row.iloc[0]["result_status"])


def _audit_m5(profile: str) -> dict[str, object] | None:
    path = _summary_path(profile, "m5")
    if not path.is_file():
        return None
    summary = pd.read_csv(path)
    split_path = _processed_path(profile, "m5", "split_metadata.csv")
    split = pd.read_csv(split_path).iloc[0].to_dict() if split_path.is_file() else {}
    cvar_calib = _row_value(summary, "cvar", "optimized", "calibration_joint_violation")
    cvar_improvement = _row_value(summary, "cvar", "optimized", "relative_improvement")
    cvar_status = _row_status(summary, "cvar", "optimized")
    cantelli_improvement = _row_value(summary, "cantelli", "optimized", "relative_improvement", 0.0)
    eligible = (
        bool(split.get("selection_uses_heldout", False)) is False
        and cvar_status == "success"
        and cvar_calib <= 0.05
        and float(split.get("n_heldout_days", 365)) <= 180
    )
    score = cvar_improvement + 0.25 * max(cantelli_improvement, 0.0)
    return {
        "case": "m5",
        "profile": profile,
        "selection_uses_heldout": bool(split.get("selection_uses_heldout", False)),
        "calibration_only_promotion_rule": True,
        "n_heldout_days": int(split.get("n_heldout_days", -1)),
        "cvar_optimized_status": cvar_status,
        "cvar_calibration_joint_violation": cvar_calib,
        "cvar_heldout_joint_violation_reported_not_used": _row_value(summary, "cvar", "optimized", "heldout_joint_violation"),
        "cvar_relative_improvement": cvar_improvement,
        "cantelli_relative_improvement": cantelli_improvement,
        "promotion_eligible": bool(eligible),
        "promotion_score": float(score if eligible else -np.inf),
        "promotion_reason": "near-term stable calibration-valid CVaR profile" if eligible else "failed predeclared M5 promotion rule",
    }


def _audit_power(profile: str) -> dict[str, object] | None:
    path = _summary_path(profile, "power")
    if not path.is_file():
        return None
    summary = pd.read_csv(path)
    audit_path = _processed_path(profile, "power", "constraint_design_audit.csv")
    audit = pd.read_csv(audit_path).iloc[0].to_dict() if audit_path.is_file() else {}
    cvar_improvement = _row_value(summary, "cvar", "optimized", "relative_improvement", 0.0)
    bernstein_improvement = _row_value(summary, "bernstein", "optimized", "relative_improvement", 0.0)
    cvar_status = _row_status(summary, "cvar", "optimized")
    bernstein_status = _row_status(summary, "bernstein", "optimized")
    baseline = float(audit.get("calibration_baseline_joint_violation", np.nan))
    band_low = float(audit.get("calibration_joint_violation_band_low", 0.05))
    band_high = float(audit.get("calibration_joint_violation_band_high", 0.30))
    nominal_bad = int(audit.get("nominal_exceeds_selected_risk_limit", 999))
    optimized_success = (cvar_status == "success" and cvar_improvement > 0.0) or (
        bernstein_status == "success" and bernstein_improvement > 0.0
    )
    eligible = nominal_bad == 0 and band_low <= baseline <= band_high and optimized_success
    score = max(cvar_improvement, bernstein_improvement) + 0.05 * float(cvar_status == "success")
    return {
        "case": "power",
        "profile": profile,
        "selection_uses_heldout": bool(audit.get("selection_uses_heldout", False)),
        "calibration_only_promotion_rule": True,
        "nominal_exceeds_selected_risk_limit": nominal_bad,
        "calibration_baseline_joint_violation": baseline,
        "calibration_joint_violation_band_low": band_low,
        "calibration_joint_violation_band_high": band_high,
        "cvar_optimized_status": cvar_status,
        "bernstein_optimized_status": bernstein_status,
        "cvar_relative_improvement": cvar_improvement,
        "bernstein_relative_improvement": bernstein_improvement,
        "cvar_heldout_joint_violation_reported_not_used": _row_value(summary, "cvar", "optimized", "heldout_joint_violation"),
        "bernstein_heldout_joint_violation_reported_not_used": _row_value(summary, "bernstein", "optimized", "heldout_joint_violation"),
        "promotion_eligible": bool(eligible),
        "promotion_score": float(score if eligible else -np.inf),
        "promotion_reason": "balanced calibration-valid Power profile with accepted optimized allocation"
        if eligible
        else "failed predeclared Power promotion rule",
    }


def _audit_french(profile: str) -> dict[str, object] | None:
    path = _summary_path(profile, "french")
    if not path.is_file():
        return None
    summary = pd.read_csv(path)
    split_path = _processed_path(profile, "french", "split_metadata.csv")
    split = pd.read_csv(split_path).iloc[0].to_dict() if split_path.is_file() else {}
    cvar_status = _row_status(summary, "cvar", "optimized")
    bernstein_status = _row_status(summary, "bernstein", "optimized")
    cantelli_status = _row_status(summary, "cantelli", "optimized")
    return {
        "case": "french",
        "profile": profile,
        "selection_uses_heldout": bool(split.get("selection_uses_heldout", False)),
        "calibration_only_promotion_rule": False,
        "train_start_date": split.get("train_start_date", ""),
        "train_end_date": split.get("train_end_date", ""),
        "test_start_date": split.get("test_start_date", ""),
        "test_end_date": split.get("test_end_date", ""),
        "cvar_optimized_status": cvar_status,
        "bernstein_optimized_status": bernstein_status,
        "cantelli_optimized_status": cantelli_status,
        "cvar_calibration_joint_violation": _row_value(summary, "cvar", "optimized", "calibration_joint_violation"),
        "cvar_heldout_joint_violation_reported_not_used": _row_value(summary, "cvar", "optimized", "heldout_joint_violation"),
        "bernstein_heldout_joint_violation_reported_not_used": _row_value(summary, "bernstein", "optimized", "heldout_joint_violation"),
        "cvar_relative_improvement": _row_value(summary, "cvar", "optimized", "relative_improvement"),
        "bernstein_relative_improvement": _row_value(summary, "bernstein", "optimized", "relative_improvement"),
        "cantelli_relative_improvement": _row_value(summary, "cantelli", "optimized", "relative_improvement", 0.0),
        "promotion_eligible": False,
        "promotion_score": -np.inf,
        "promotion_reason": "French sensitivity is a robustness audit only; keep the predeclared main profile",
    }


def main() -> None:
    global RESULTS_ROOT
    args = parse_args()
    RESULTS_ROOT = runtime_results_root(args.output_root)
    rows: list[dict[str, object]] = []
    sensitivity_root = RESULTS_ROOT / "sensitivity"
    for profile_dir in sorted(path for path in sensitivity_root.glob("*") if path.is_dir() and path.name != "summary"):
        profile = profile_dir.name
        for builder in (_audit_m5, _audit_power, _audit_french):
            row = builder(profile)
            if row is not None:
                rows.append(row)
    if not rows:
        raise RuntimeError("no sensitivity profiles found; run run_real_data_sensitivity.py first")
    audit = pd.DataFrame(rows)
    recommendations = []
    for case, group in audit.groupby("case", sort=False):
        eligible = group.loc[group["promotion_eligible"].astype(bool)].copy()
        if eligible.empty:
            recommendations.append({"case": case, "recommended_main_profile": "current_main", "recommendation": "keep_current_main"})
        else:
            best = eligible.sort_values(["promotion_score", "profile"], ascending=[False, True]).iloc[0]
            recommendations.append(
                {
                    "case": case,
                    "recommended_main_profile": best["profile"],
                    "recommendation": "candidate_can_be_promoted_after_manual_review",
                }
            )
    rec = pd.DataFrame(recommendations)
    out = RESULTS_ROOT / "summary"
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "profile_selection_audit.csv", index=False)
    rec.to_csv(out / "profile_recommendations.csv", index=False)
    write_latex_table(audit.replace([np.inf, -np.inf], np.nan), table_dir / "tab_profile_selection_audit.tex")
    write_latex_table(rec, table_dir / "tab_profile_recommendations.tex")


if __name__ == "__main__":
    main()
