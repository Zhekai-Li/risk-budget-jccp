from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "synthetic" / "run_synthetic.py"
CONFIG_PATH = REPO_ROOT / "configs" / "synthetic" / "synthetic_service.yaml"
EXPECTED_COLUMNS = {
    "method",
    "m",
    "heterogeneity",
    "equal_objective",
    "optimized_objective",
    "improvement_percent",
    "max_share",
    "entropy",
    "exact_joint_violation",
    "improvement_percent_std",
    "replications",
}


def _run_synthetic(*, output_csv: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--config",
        str(CONFIG_PATH),
        "--output",
        str(output_csv),
    ]
    if extra_args is not None:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_run_synthetic_produces_expected_rows(tmp_path: Path) -> None:
    output_csv = tmp_path / "synthetic_results.csv"
    completed = _run_synthetic(
        output_csv=output_csv,
        extra_args=["--m-values", "5", "--heterogeneity-values", "0.0"],
    )

    assert completed.returncode == 0, completed.stderr
    assert output_csv.exists()

    results = pd.read_csv(output_csv)
    assert set(results.columns) == EXPECTED_COLUMNS
    assert len(results) == 3
    assert set(results["method"]) == {"bernstein", "cantelli", "cvar"}
    assert (results["m"] == 5).all()
    assert (results["heterogeneity"] == 0.0).all()
    assert (results["optimized_objective"] <= results["equal_objective"]).all()
    assert (results["replications"] == 30).all()
    raw = pd.read_csv(output_csv.with_name("synthetic_results_raw.csv"))
    assert len(raw) == 90
    assert set(raw["seed"]) == set(range(7, 37))


def test_run_synthetic_is_reproducible_for_same_inputs(tmp_path: Path) -> None:
    first_output = tmp_path / "synthetic_results_first.csv"
    second_output = tmp_path / "synthetic_results_second.csv"
    extra_args = ["--m-values", "5", "--heterogeneity-values", "0.0"]

    first_completed = _run_synthetic(output_csv=first_output, extra_args=extra_args)
    second_completed = _run_synthetic(output_csv=second_output, extra_args=extra_args)

    assert first_completed.returncode == 0, first_completed.stderr
    assert second_completed.returncode == 0, second_completed.stderr

    first_results = pd.read_csv(first_output)
    second_results = pd.read_csv(second_output)
    pd.testing.assert_frame_equal(first_results, second_results)


def test_run_synthetic_rejects_empty_m_value_override(tmp_path: Path) -> None:
    output_csv = tmp_path / "synthetic_results.csv"
    completed = _run_synthetic(
        output_csv=output_csv,
        extra_args=["--m-values", "--heterogeneity-values", "0.0"],
    )

    assert completed.returncode != 0
    assert "expected at least one argument" in completed.stderr
