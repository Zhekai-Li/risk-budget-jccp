from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from risk_budget_jccp.reporting.validation import (
    ALLOCATIONS,
    CASES,
    CERTIFICATES,
    compare_csv,
    validate_split_summary,
)


def _summary_rows() -> list[dict[str, object]]:
    return [
        {
            "case": case,
            "certificate": certificate,
            "allocation": allocation,
            "result_status": "success",
            "objective": 1.0,
            "relative_improvement": 0.0,
            "calibration_joint_violation": 0.01,
            "heldout_joint_violation": 0.02,
            "fallback_used": False,
            "certificate_acceptance_status": "accepted",
        }
        for case in CASES
        for certificate in CERTIFICATES
        for allocation in ALLOCATIONS
    ]


def test_validate_split_summary_accepts_complete_grid(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    pd.DataFrame(_summary_rows()).to_csv(path, index=False)
    validate_split_summary(path)


def test_validate_split_summary_rejects_duplicate_configuration(tmp_path: Path) -> None:
    rows = _summary_rows()
    rows[-1] = rows[0].copy()
    path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="18 unique"):
        validate_split_summary(path)


def test_validate_split_summary_rejects_status_certificate_conflict(tmp_path: Path) -> None:
    rows = _summary_rows()
    rows[0]["certificate_acceptance_status"] = "rejected"
    path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rejected certificates"):
        validate_split_summary(path)


def test_compare_csv_compares_serialized_allocations_numerically(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {"seed": [7], "alpha_allocation": ["[0.010000000001,0.039999999999]"]}
    ).to_csv(candidate, index=False)
    pd.DataFrame({"seed": [7], "alpha_allocation": ["[0.01,0.04]"]}).to_csv(
        reference, index=False
    )

    compare_csv(candidate, reference)
