from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "synthetic" / "run_synthetic_coupled_capacity.py"
EXPECTED_COLUMNS = [
    "heterogeneity",
    "equal_objective",
    "optimized_objective",
    "improvement_percent",
    "max_share",
    "exact_joint_violation",
    "iterations",
    "runtime",
    "final_residual",
]


def _write_config(path: Path) -> None:
    config = {
        "alpha": 0.05,
        "seed": 7,
        "dimension": 3,
        "num_constraints": 5,
        "heterogeneity_values": [0.8],
        "solver": {
            "max_iter": 20,
            "tol": 1e-5,
            "eps_alpha": 1e-6,
            "eps_theta": 1e-6,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _run_benchmark(*, config_path: Path, output_csv: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--output",
            str(output_csv),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_run_synthetic_coupled_capacity_writes_expected_csv(tmp_path: Path) -> None:
    config_path = tmp_path / "synthetic_coupled_capacity.yaml"
    output_csv = tmp_path / "synthetic_coupled_capacity.csv"
    _write_config(config_path)

    completed = _run_benchmark(config_path=config_path, output_csv=output_csv)

    assert completed.returncode == 0, completed.stderr
    assert output_csv.exists()

    results = pd.read_csv(output_csv)
    assert list(results.columns) == EXPECTED_COLUMNS
    assert len(results) == 1
    row = results.iloc[0]
    assert row["heterogeneity"] == 0.8
    assert row["optimized_objective"] <= row["equal_objective"] + 1e-8
    assert row["improvement_percent"] >= 0.0
    assert 0.0 < row["max_share"] <= 1.0
    assert 0.0 <= row["exact_joint_violation"] <= 0.05 + 1e-8
    assert row["iterations"] >= 1
    assert row["runtime"] >= 0.0
    assert row["final_residual"] >= 0.0


def test_run_synthetic_coupled_capacity_is_reproducible_except_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "synthetic_coupled_capacity.yaml"
    first_output = tmp_path / "synthetic_coupled_capacity_first.csv"
    second_output = tmp_path / "synthetic_coupled_capacity_second.csv"
    _write_config(config_path)

    first_completed = _run_benchmark(config_path=config_path, output_csv=first_output)
    second_completed = _run_benchmark(config_path=config_path, output_csv=second_output)

    assert first_completed.returncode == 0, first_completed.stderr
    assert second_completed.returncode == 0, second_completed.stderr

    first_results = pd.read_csv(first_output)
    second_results = pd.read_csv(second_output)
    pd.testing.assert_frame_equal(
        first_results.drop(columns=["runtime"]),
        second_results.drop(columns=["runtime"]),
    )


def test_run_synthetic_coupled_capacity_fails_when_solver_does_not_converge(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "synthetic_coupled_capacity.yaml"
    output_csv = tmp_path / "synthetic_coupled_capacity.csv"
    _write_config(config_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["solver"]["max_iter"] = 1
    config["solver"]["tol"] = 1e-12
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = _run_benchmark(config_path=config_path, output_csv=output_csv)

    assert completed.returncode != 0
    assert "failed to converge" in completed.stderr
    assert not output_csv.exists()
