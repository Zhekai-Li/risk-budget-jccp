from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare all real-data inputs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--case", choices=("m5", "power", "french"), action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.case or ["m5", "power", "french"]
    for case in cases:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "real_data" / f"prepare_{case}_data.py"),
            "--config",
            args.config,
            "--output-root",
            args.output_root,
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        if args.force_download:
            command.append("--force-download")
        result = subprocess.run(command, cwd=REPO_ROOT, text=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
