from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TESTS = (
    "tests/test_run_synthetic.py",
    "tests/test_run_synthetic_cross_domain.py",
    "tests/test_run_synthetic_coupled_capacity.py",
    "tests/real_data/test_m5_pipeline_tiny.py",
    "tests/real_data/test_power_pipeline_tiny.py",
    "tests/real_data/test_french_pipeline_tiny.py",
    "tests/test_make_paper_figures.py",
    "tests/test_make_tables.py",
    "tests/test_validate_final_results.py",
    "tests/test_result_validation.py",
    "tests/test_experiment_configs.py",
)


def main() -> None:
    command = [sys.executable, "-m", "pytest", "-q", *SMOKE_TESTS]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("offline smoke pipeline passed")


if __name__ == "__main__":
    main()
