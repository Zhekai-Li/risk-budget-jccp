from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_budget_jccp.real_data.common.latex import write_latex_table


def summary_for_evaluation_split(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"in_sample", "out_of_sample"}:
        raise ValueError("split must be 'in_sample' or 'out_of_sample'")
    frame = summary.copy()
    frame["evaluation_split"] = split
    if split == "in_sample":
        replacements = {
            "heldout_joint_violation": "calibration_joint_violation",
            "heldout_emergency_joint_violation": "calibration_emergency_joint_violation",
            "heldout_cont_overload_rate": "calibration_cont_overload_rate",
            "heldout_average_scalar_violations": "calibration_average_scalar_violations",
            "heldout_max_scalar_violations": "calibration_max_scalar_violations",
            "empirical_joint_violation": "calibration_joint_violation",
            "average_scalar_violations": "calibration_average_scalar_violations",
            "max_scalar_violations": "calibration_max_scalar_violations",
        }
        for target, source in replacements.items():
            if target in frame.columns and source in frame.columns:
                frame[target] = frame[source]
    return frame


def write_split_summary(
    summary: pd.DataFrame,
    *,
    case: str,
    split: str,
    tables_dir: str | Path,
) -> pd.DataFrame:
    frame = summary_for_evaluation_split(summary, split)
    table_root = Path(tables_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_root / f"{case}_summary.csv", index=False)
    write_latex_table(frame, table_root / f"tab_{case}_summary.tex")
    return frame


def evaluation_label(split: str) -> str:
    return "In-sample" if split == "in_sample" else "Out-of-sample"
