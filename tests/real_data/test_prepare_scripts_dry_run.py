from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_scripts_dry_run() -> None:
    for script in ("prepare_m5_data.py", "prepare_power_data.py", "prepare_french_data.py"):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "real_data" / script),
                "--config",
                str(REPO_ROOT / "configs" / "real_data" / "main.yaml"),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
