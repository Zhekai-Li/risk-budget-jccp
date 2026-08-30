from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "make_tables.py"


def _write_fixture_tables(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "method": "bernstein",
                "m": 20,
                "heterogeneity": 0.5,
                "equal_objective": 10.0,
                "optimized_objective": 9.0,
                "improvement_percent": 10.0,
                "max_share": 0.2,
                "entropy": 0.9,
            },
            {
                "method": "cantelli",
                "m": 20,
                "heterogeneity": 0.5,
                "equal_objective": 12.0,
                "optimized_objective": 10.5,
                "improvement_percent": 12.5,
                "max_share": 0.3,
                "entropy": 0.8,
            },
            {
                "method": "cvar",
                "m": 20,
                "heterogeneity": 0.5,
                "equal_objective": 11.0,
                "optimized_objective": 10.0,
                "improvement_percent": 9.1,
                "max_share": 0.25,
                "entropy": 0.85,
            },
        ]
    ).to_csv(input_dir / "synthetic_service.csv", index=False)

    pd.DataFrame(
        [
            {
                "domain": "service_capacity",
                "label": "Service capacity",
                "method": "bernstein",
                "m": 20,
                "heterogeneity": 0.5,
                "equal_objective": 10.0,
                "optimized_objective": 9.0,
                "improvement_percent": 10.0,
                "max_share": 0.2,
                "entropy": 0.9,
                "exact_joint_violation": 0.01,
                "improvement_percent_std": 0.5,
                "replications": 30,
            }
        ]
    ).to_csv(input_dir / "synthetic_cross_domain.csv", index=False)

    pd.DataFrame(
        [
            {
                "heterogeneity": 0.5,
                "equal_objective": 3.5,
                "optimized_objective": 3.2,
                "improvement_percent": 8.6,
                "max_share": 0.4,
                "exact_joint_violation": 0.01,
                "iterations": 5,
                "runtime": 0.2,
                "final_residual": 1e-6,
            }
        ]
    ).to_csv(input_dir / "synthetic_coupled_capacity.csv", index=False)



def test_make_tables_writes_expected_latex_fragments(tmp_path: Path) -> None:
    input_dir = tmp_path / "tables"
    output_dir = tmp_path / "tables_latex"
    _write_fixture_tables(input_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    expected_outputs = {
        "service_bernstein.tex",
        "service_cantelli.tex",
        "service_cvar.tex",
        "synthetic_cross_domain.tex",
        "coupled_capacity.tex",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_outputs


def test_make_tables_requires_all_input_csvs(tmp_path: Path) -> None:
    input_dir = tmp_path / "tables"
    output_dir = tmp_path / "tables_latex"
    input_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode != 0
    assert "missing required input" in completed.stderr.lower()
