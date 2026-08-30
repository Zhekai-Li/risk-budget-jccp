from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_requires_prepared_data_fails_without_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "m5_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "alpha": 0.05,
                "profile": "tiny",
                "profiles": {
                    "tiny": {
                        "m5": {
                            "n_series": 2,
                            "max_train_days": 10,
                            "max_test_days": 2,
                            "min_active_days": 1,
                            "selection_metric": "revenue_mean",
                            "cost_proxy": "median_sell_price",
                        },
                        "power": {
                            "n_snapshots": 1,
                            "max_branches": 1,
                            "max_train_scenarios": 2,
                            "max_test_scenarios": 2,
                            "use_external_wind_solar_api": False,
                            "fallback_to_rts_builtin": True,
                        },
                        "french": {
                            "max_assets": 2,
                            "max_train_scenarios": 2,
                            "max_test_scenarios": 2,
                            "max_weight": 0.6,
                            "target_return_fraction": 0.5,
                            "start_date": "2020-01-01",
                            "train_end_date": "2020-01-02",
                            "test_start_date": "2020-01-03",
                            "test_end_date": "2020-01-04",
                        },
                        "dca": {
                            "max_iter": 1,
                            "tol": 1.0e-4,
                            "proximal_weight": 1.0e-4,
                            "min_alpha": 1.0e-8,
                            "feasibility_tol": 1.0e-6,
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "real_data" / "run_m5.py"),
            "--config",
            str(config_path),
            "--require-prepared-data",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "prepare_m5_data.py" in result.stderr
