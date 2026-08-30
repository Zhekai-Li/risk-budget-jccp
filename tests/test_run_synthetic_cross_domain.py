from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "synthetic" / "run_synthetic_cross_domain.py"


def test_cross_domain_runner_writes_raw_and_summary_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "cross_domain.yaml"
    output_path = tmp_path / "summary.csv"
    raw_output_path = tmp_path / "raw.csv"
    config_path.write_text(
        yaml.safe_dump(
            {
                "alpha": 0.05,
                "base_seed": 11,
                "replications": 2,
                "domains": [
                    {
                        "domain": "test_case",
                        "label": "Test case",
                        "method": "bernstein",
                        "m": 4,
                        "heterogeneity": 0.5,
                        "seed_offset": 20,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--raw-output",
            str(raw_output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    summary = pd.read_csv(output_path)
    raw = pd.read_csv(raw_output_path)
    assert len(summary) == 1
    assert len(raw) == 2
    assert summary.loc[0, "replications"] == 2
    assert set(raw["seed"]) == {31, 32}
    assert raw["alpha_allocation"].str.startswith("[").all()
    assert (raw["optimized_objective"] <= raw["equal_objective"]).all()
