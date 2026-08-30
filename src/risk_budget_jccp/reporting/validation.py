from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_budget_jccp.evaluation.metrics import normalized_entropy, relative_improvement_percent
from risk_budget_jccp.models.synthetic_service import (
    exact_gaussian_joint_violation,
    make_service_instance,
    service_objective,
)


CASES = ("m5", "power", "french")
CERTIFICATES = ("bernstein", "cvar", "cantelli")
ALLOCATIONS = ("equal", "optimized")
PAPER_FIGURES = (
    "aggregate_improvement_vs_violation.pdf",
    "aggregate_objective_improvement.pdf",
    "budget_driver_overlay.pdf",
    "calibration_heldout_shift.pdf",
    "cross_joint_violation.pdf",
    "safety_factor_vs_budget.pdf",
)
SUMMARY_REQUIRED_COLUMNS = (
    "case",
    "certificate",
    "allocation",
    "result_status",
    "objective",
    "relative_improvement",
    "calibration_joint_violation",
    "heldout_joint_violation",
)
NONDETERMINISTIC_COLUMN_TOKENS = (
    "runtime",
    "elapsed",
    "timestamp",
    "time_seconds",
    "log_time",
    "iterations",
    "final_residual",
)
DIAGNOSTIC_TEXT_COLUMNS = {
    "failure_reason",
    "fallback_reason",
    "optimization_note",
    "solver_message",
    "solver_status",
}
SYNTHETIC_REPLICATIONS = 30
SYNTHETIC_VALUE_COLUMNS = (
    "equal_objective",
    "optimized_objective",
    "improvement_percent",
    "max_share",
    "entropy",
    "exact_joint_violation",
)


@dataclass(frozen=True)
class ValidationReport:
    checked_files: tuple[Path, ...]
    compared_files: tuple[Path, ...]


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty required artifact: {path}")
    return path


def validate_split_summary(path: Path) -> None:
    frame = pd.read_csv(_require_file(path))
    missing = sorted(set(SUMMARY_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    keys = frame.loc[:, ["case", "certificate", "allocation"]]
    if len(frame) != 18 or len(keys.drop_duplicates()) != 18:
        raise ValueError(f"{path} must contain exactly 18 unique case/certificate/allocation rows")
    expected = {(case, certificate, allocation) for case in CASES for certificate in CERTIFICATES for allocation in ALLOCATIONS}
    actual = set(keys.itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{path} has an incomplete experiment grid")
    # Cross-case summaries intentionally contain case-specific columns, which
    # are NaN outside the case that defines them. Universal reported metrics
    # must always be finite, including failed/fallback rows.
    universal_numeric = frame.loc[
        :,
        ["objective", "relative_improvement", "calibration_joint_violation", "heldout_joint_violation"],
    ]
    if not np.isfinite(universal_numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite required numeric values")
    allowed_statuses = {"success", "fallback_equal", "failed", "infeasible", "no_solution"}
    if not set(frame["result_status"].astype(str)).issubset(allowed_statuses):
        raise ValueError(f"{path} contains an unknown result_status")
    if "fallback_used" in frame:
        fallback = frame["fallback_used"].astype(str).str.lower().isin({"true", "1"})
        inconsistent = fallback & ~frame["result_status"].isin(["fallback_equal"])
        if bool(inconsistent.any()):
            raise ValueError(f"{path} has fallback_used rows without fallback_equal status")
    if "certificate_acceptance_status" in frame:
        successful = frame["result_status"].isin(["success", "fallback_equal"])
        rejected = frame["certificate_acceptance_status"].astype(str).str.lower().isin({"failed", "rejected"})
        if bool((successful & rejected).any()):
            raise ValueError(f"{path} marks successful results with rejected certificates")


def validate_synthetic_grid(path: Path) -> None:
    frame = pd.read_csv(_require_file(path))
    if frame.empty:
        raise ValueError(f"synthetic result is empty: {path}")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"synthetic result contains invalid numeric values: {path}")
    if "method" in frame and frame["method"].nunique() < 2:
        raise ValueError(f"synthetic result must compare at least two methods: {path}")


def validate_synthetic_summary_pair(
    summary_path: Path,
    raw_path: Path,
    *,
    group_columns: tuple[str, ...],
    replications: int = SYNTHETIC_REPLICATIONS,
) -> None:
    summary = pd.read_csv(_require_file(summary_path)).sort_values(list(group_columns)).reset_index(drop=True)
    raw = pd.read_csv(_require_file(raw_path))
    required_raw = set(group_columns) | set(SYNTHETIC_VALUE_COLUMNS) | {
        "m",
        "heterogeneity",
        "seed",
        "alpha_allocation",
    }
    missing = sorted(required_raw - set(raw.columns))
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {missing}")
    for row in raw.itertuples(index=False):
        allocation = np.asarray(json.loads(row.alpha_allocation), dtype=float)
        if allocation.shape != (int(row.m),):
            raise ValueError(f"{raw_path} contains an allocation with the wrong dimension")
        if not np.isclose(allocation.sum(), 0.05, atol=1e-10):
            raise ValueError(f"{raw_path} contains an allocation with the wrong budget sum")
        instance = make_service_instance(
            m=int(row.m), heterogeneity=float(row.heterogeneity), seed=int(row.seed)
        )
        equal = np.full(int(row.m), 0.05 / int(row.m), dtype=float)
        expected_equal = service_objective(instance.weights, equal, str(row.method))
        expected_optimized = service_objective(instance.weights, allocation, str(row.method))
        checks = {
            "equal_objective": expected_equal,
            "optimized_objective": expected_optimized,
            "improvement_percent": relative_improvement_percent(
                equal_value=expected_equal, optimized_value=expected_optimized
            ),
            "max_share": float(allocation.max() / allocation.sum()),
            "entropy": normalized_entropy(allocation),
            "exact_joint_violation": exact_gaussian_joint_violation(allocation, str(row.method)),
        }
        for column, expected_value in checks.items():
            if not np.isclose(float(getattr(row, column)), expected_value, atol=1e-9, rtol=1e-9):
                raise ValueError(f"{raw_path} column {column} is inconsistent with seed/allocation")

    grouped = raw.groupby(list(group_columns), sort=True)
    for seeds in grouped["seed"].apply(lambda values: sorted(values.astype(int).tolist())):
        if len(set(seeds)) != replications:
            raise ValueError(f"{raw_path} must contain {replications} distinct seeds per setting")
        if seeds != list(range(seeds[0], seeds[0] + replications)):
            raise ValueError(f"{raw_path} seeds must be consecutive in each setting")

    expected = (
        grouped[list(SYNTHETIC_VALUE_COLUMNS)]
        .mean()
        .reset_index()
        .sort_values(list(group_columns))
        .reset_index(drop=True)
    )
    if len(summary) != len(expected):
        raise ValueError(f"{summary_path} has the wrong number of aggregate rows")
    for column in expected.columns:
        if column in group_columns:
            if not summary[column].equals(expected[column]):
                raise ValueError(f"{summary_path} grouping column {column} does not match raw data")
        elif not np.allclose(summary[column], expected[column], atol=1e-10, rtol=1e-10):
            raise ValueError(f"{summary_path} column {column} does not match raw means")
    expected_std = grouped["improvement_percent"].std(ddof=0).to_numpy()
    if not np.allclose(summary["improvement_percent_std"], expected_std, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{summary_path} standard deviations do not match raw data")
    if not (summary["replications"].astype(int) == replications).all():
        raise ValueError(f"{summary_path} must report {replications} replications")


def validate_canonical_results(input_dir: Path) -> tuple[Path, ...]:
    input_dir = input_dir.resolve()
    service = input_dir / "synthetic_service.csv"
    service_raw = input_dir / "synthetic_service_raw.csv"
    cross = input_dir / "synthetic_cross_domain.csv"
    cross_raw = input_dir / "synthetic_cross_domain_raw.csv"
    coupled = input_dir / "synthetic_coupled_capacity.csv"
    validate_synthetic_summary_pair(
        service,
        service_raw,
        group_columns=("method", "m", "heterogeneity"),
    )
    validate_synthetic_summary_pair(
        cross,
        cross_raw,
        group_columns=("domain", "label", "method", "m", "heterogeneity"),
    )
    coupled_frame = pd.read_csv(_require_file(coupled))
    required = {
        "heterogeneity",
        "equal_objective",
        "optimized_objective",
        "improvement_percent",
        "max_share",
        "exact_joint_violation",
        "iterations",
        "runtime",
        "final_residual",
    }
    missing = sorted(required - set(coupled_frame.columns))
    if missing:
        raise ValueError(f"{coupled} is missing required columns: {missing}")
    if len(coupled_frame) != 4 or set(coupled_frame["heterogeneity"]) != {0.0, 0.5, 1.0, 1.5}:
        raise ValueError(f"{coupled} must contain the four fixed-seed heterogeneity settings")
    numeric = coupled_frame[list(required)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{coupled} contains non-finite values")
    if not (coupled_frame["optimized_objective"] <= coupled_frame["equal_objective"] + 1e-8).all():
        raise ValueError(f"{coupled} contains a non-improving optimized solution")
    return service, service_raw, cross, cross_raw, coupled


def _comparable_columns(candidate: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    shared = []
    for column in candidate.columns.intersection(reference.columns):
        if column not in DIAGNOSTIC_TEXT_COLUMNS and not any(
            token in column.lower() for token in NONDETERMINISTIC_COLUMN_TOKENS
        ):
            shared.append(column)
    return shared


def compare_csv(candidate_path: Path, reference_path: Path, *, rtol: float = 1e-4, atol: float = 1e-6) -> None:
    candidate = pd.read_csv(_require_file(candidate_path))
    reference = pd.read_csv(_require_file(reference_path))
    columns = _comparable_columns(candidate, reference)
    if candidate.shape[0] != reference.shape[0] or not columns:
        raise ValueError(f"candidate/reference CSV shape mismatch: {candidate_path} vs {reference_path}")
    for column in columns:
        left = candidate[column]
        right = reference[column]
        if column == "alpha_allocation":
            for row_index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
                left_allocation = np.asarray(json.loads(str(left_value)), dtype=float)
                right_allocation = np.asarray(json.loads(str(right_value)), dtype=float)
                if left_allocation.shape != right_allocation.shape or not np.allclose(
                    left_allocation,
                    right_allocation,
                    rtol=rtol,
                    atol=atol,
                ):
                    raise ValueError(
                        f"allocation mismatch in row {row_index}: "
                        f"{candidate_path} vs {reference_path}"
                    )
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=rtol, atol=atol, equal_nan=True):
                raise ValueError(f"numeric mismatch in {column}: {candidate_path} vs {reference_path}")
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            raise ValueError(f"value mismatch in {column}: {candidate_path} vs {reference_path}")


def validate_reproduction(candidate_root: Path, reference_root: Path, *, compare: bool = True) -> ValidationReport:
    candidate_root = candidate_root.resolve()
    reference_root = reference_root.resolve()
    checked: list[Path] = []
    compared: list[Path] = []
    summary_dir = candidate_root / "real_data" / "summary"
    for split in ("in_sample", "out_of_sample"):
        path = summary_dir / f"real_data_summary_{split}.csv"
        validate_split_summary(path)
        checked.append(path)
        if compare:
            reference = reference_root / "real_data" / "summary" / path.name
            compare_csv(path, reference)
            compared.append(path)
    synthetic_dir = candidate_root / "synthetic"
    checked.extend(validate_canonical_results(synthetic_dir))
    for name in (
        "synthetic_service.csv",
        "synthetic_service_raw.csv",
        "synthetic_cross_domain.csv",
        "synthetic_cross_domain_raw.csv",
        "synthetic_coupled_capacity.csv",
    ):
        path = synthetic_dir / name
        if compare:
            reference = reference_root / "tables" / name
            compare_csv(path, reference)
            compared.append(path)
    for name in PAPER_FIGURES:
        checked.append(_require_file(candidate_root / "paper" / name))
    checked.append(_require_file(summary_dir / "profile_selection_audit.csv"))
    return ValidationReport(tuple(checked), tuple(compared))
