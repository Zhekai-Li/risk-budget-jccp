from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_budget_jccp.real_data.common.evaluation import summary_for_evaluation_split, write_split_summary


def test_summary_for_evaluation_split_uses_calibration_metrics() -> None:
    summary = pd.DataFrame(
        {
            "case": ["x"],
            "certificate": ["cvar"],
            "allocation": ["equal"],
            "calibration_joint_violation": [0.11],
            "heldout_joint_violation": [0.22],
            "calibration_average_scalar_violations": [1.0],
            "heldout_average_scalar_violations": [2.0],
            "calibration_max_scalar_violations": [3],
            "heldout_max_scalar_violations": [4],
            "empirical_joint_violation": [0.22],
            "average_scalar_violations": [2.0],
            "max_scalar_violations": [4],
        }
    )
    in_sample = summary_for_evaluation_split(summary, "in_sample")
    out_of_sample = summary_for_evaluation_split(summary, "out_of_sample")
    assert in_sample.loc[0, "evaluation_split"] == "in_sample"
    assert in_sample.loc[0, "heldout_joint_violation"] == 0.11
    assert in_sample.loc[0, "empirical_joint_violation"] == 0.11
    assert in_sample.loc[0, "average_scalar_violations"] == 1.0
    assert in_sample.loc[0, "max_scalar_violations"] == 3
    assert out_of_sample.loc[0, "evaluation_split"] == "out_of_sample"
    assert out_of_sample.loc[0, "heldout_joint_violation"] == 0.22


def test_write_split_summary_outputs_csv_and_tex(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "case": ["x"],
            "certificate": ["cvar"],
            "allocation": ["equal"],
            "calibration_joint_violation": [0.0],
            "heldout_joint_violation": [0.1],
        }
    )
    written = write_split_summary(summary, case="x", split="out_of_sample", tables_dir=tmp_path)
    assert written.loc[0, "evaluation_split"] == "out_of_sample"
    assert (tmp_path / "x_summary.csv").is_file()
    assert (tmp_path / "tab_x_summary.tex").is_file()
