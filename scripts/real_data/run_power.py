from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.cases.power.build_instance import build_instance
from risk_budget_jccp.real_data.cases.power.report import build_power_report_assets
from risk_budget_jccp.real_data.cases.power.solve import solve_power_instance
from risk_budget_jccp.real_data.common.config import case_config, load_config, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, format_status
from risk_budget_jccp.real_data.common.logging_utils import configure_case_logger
from risk_budget_jccp.real_data.common.paths import ensure_case_dirs, publish_split_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RTS-GMLC power real-data JCCP experiment.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-if-missing", action="store_true", default=True)
    parser.add_argument("--require-prepared-data", action="store_true")
    return parser.parse_args()


def _prepare_data(config_path: str, output_root: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "real_data" / "prepare_power_data.py"),
            "--config",
            config_path,
            "--output-root",
            output_root,
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_common_config(config)
    cfg = case_config(config, "power")
    paths = ensure_case_dirs("power", args.output_root)
    logger = configure_case_logger("power", paths["logs"])
    logger.info("validated Power config and real-data output paths")
    if args.dry_run:
        return
    status = check_case_data("power", cfg, raw_dir=paths["raw"], processed_dir=paths["processed"])
    if not status.valid:
        if args.require_prepared_data:
            raise RuntimeError(
                "Power processed data are not ready. Run: "
                "python scripts/real_data/prepare_power_data.py "
                f"--config {args.config}\n{format_status(status)}"
            )
        _prepare_data(args.config, args.output_root)
        status = check_case_data("power", cfg, raw_dir=paths["raw"], processed_dir=paths["processed"])
        if not status.valid:
            raise RuntimeError(f"Power data preparation did not produce valid processed data:\n{format_status(status)}")
    instance = build_instance(paths["processed"], alpha=cfg["alpha"])
    summary = solve_power_instance(instance, paths["results"], cfg["dca"])
    summary.to_csv(paths["tables"] / "power_summary.csv", index=False)
    build_power_report_assets(instance, summary, paths["results"])
    publish_split_outputs(paths["results"], "power", args.output_root)
    logger.info("Power real-data experiment complete: %s", paths["results"])


if __name__ == "__main__":
    main()
