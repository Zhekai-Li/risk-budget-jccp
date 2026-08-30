from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "validate_results.py"
SYNTHETIC_FILES = (
    "synthetic_service.csv",
    "synthetic_service_raw.csv",
    "synthetic_cross_domain.csv",
    "synthetic_cross_domain_raw.csv",
    "synthetic_coupled_capacity.csv",
)


def _copy_canonical_results(destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in SYNTHETIC_FILES:
        shutil.copy2(REPO_ROOT / "results" / "tables" / name, destination / name)


def test_validate_results_accepts_canonical_synthetic_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "tables"
    _copy_canonical_results(input_dir)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input-dir", str(input_dir)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "canonical result validation passed" in completed.stdout.lower()


def test_validate_results_rejects_tampered_raw_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "tables"
    _copy_canonical_results(input_dir)
    raw_path = input_dir / "synthetic_service_raw.csv"
    frame = pd.read_csv(raw_path)
    frame.loc[0, "optimized_objective"] += 1.0
    frame.to_csv(raw_path, index=False)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input-dir", str(input_dir)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "inconsistent with seed/allocation" in completed.stderr
