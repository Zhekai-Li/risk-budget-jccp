from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.common.latex import write_latex_table
from risk_budget_jccp.real_data.common.paths import runtime_results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all real-data experiments.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-if-missing", action="store_true", default=True)
    parser.add_argument("--require-prepared-data", action="store_true")
    return parser.parse_args()


def _run(script: str, config: str, output_root: str, dry_run: bool, require_prepared_data: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "real_data" / script),
        "--config",
        config,
        "--output-root",
        output_root,
    ]
    if dry_run:
        command.append("--dry-run")
    if require_prepared_data:
        command.append("--require-prepared-data")
    result = subprocess.run(command, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _compact_status(row: pd.Series) -> str:
    result_status = str(row["result_status"])
    certificate_status = str(row["certificate_acceptance_status"])
    contract = str(row["calibration_jvp_contract"])
    if result_status in {"failed", "infeasible"}:
        return f"{certificate_status}; {result_status}"
    if result_status == "fallback_equal":
        return f"fallback to equal; {contract}"
    if contract == "finite_sample_cvar_bound_observed":
        return "certificate accepted; same-sample CVaR JVP bound verified"
    if contract == "not_implied_by_moment_certificate":
        return "certificate accepted; empirical calibration JVP not implied"
    return f"{certificate_status}; {contract}"


def _write_quality_outputs(summary: pd.DataFrame, out_dir: Path, table_dir: Path) -> None:
    audit_columns = [
        "case",
        "certificate",
        "allocation",
        "result_status",
        "solver_status",
        "fallback_used",
        "sum_alpha",
        "certificate_residual_max",
        "budget_residual",
        "feasibility_residual",
        "certificate_accept_tol",
        "calibration_max_violation_tolerance",
        "calibration_joint_violation",
        "calibration_jvp_bound_applies",
        "calibration_jvp_within_budget",
        "calibration_jvp_contract",
        "heldout_joint_violation",
    ]
    audit = summary.reindex(columns=audit_columns).copy()
    audit.to_csv(out_dir / "result_quality_audit.csv", index=False)
    write_latex_table(audit, table_dir / "tab_result_quality_audit.tex")

    compact = pd.DataFrame(
        {
            "Case": summary["case"].str.upper(),
            "Cert.": summary["certificate"].str.replace("cvar", "CVaR"),
            "Alloc.": summary["allocation"],
            "Obj.": summary["objective"].map(lambda value: f"{float(value):.3f}"),
            "Gain (pct.)": (100.0 * summary["relative_improvement"]).map(lambda value: f"{float(value):.1f}"),
            "Calib. JVP": summary["calibration_joint_violation"].map(lambda value: f"{float(value):.3f}"),
            "Held-out JVP": summary["heldout_joint_violation"].map(lambda value: f"{float(value):.3f}"),
            "Certificate / calibration status": summary.apply(_compact_status, axis=1),
        }
    )
    compact.to_csv(out_dir / "real_data_summary_compact.csv", index=False)
    write_latex_table(compact, table_dir / "tab_real_data_summary_compact.tex")


def main() -> None:
    args = parse_args()
    for script in ("run_m5.py", "run_power.py", "run_french.py"):
        _run(script, args.config, args.output_root, args.dry_run, args.require_prepared_data)
    if args.dry_run:
        return
    results_root = runtime_results_root(args.output_root)
    frames = []
    for case in ("m5", "power", "french"):
        path = results_root / "runs" / case / "tables" / f"{case}_summary.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("no case summaries were produced")
    summary = pd.concat(frames, ignore_index=True)
    out_dir = results_root / "summary"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "real_data_summary.csv", index=False)
    write_latex_table(summary, table_dir / "tab_real_data_summary.tex")
    _write_quality_outputs(summary, out_dir, table_dir)
    for split in ("in_sample", "out_of_sample"):
        split_frames = []
        for case in ("m5", "power", "french"):
            path = results_root / split / case / "tables" / f"{case}_summary.csv"
            if path.is_file():
                split_frames.append(pd.read_csv(path))
        if not split_frames:
            continue
        split_summary = pd.concat(split_frames, ignore_index=True)
        split_summary.to_csv(out_dir / f"real_data_summary_{split}.csv", index=False)
        write_latex_table(split_summary, table_dir / f"tab_real_data_summary_{split}.tex")


if __name__ == "__main__":
    main()
