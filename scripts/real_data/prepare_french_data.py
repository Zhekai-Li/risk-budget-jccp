from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.cases.french.download import download_french
from risk_budget_jccp.real_data.cases.french.preprocess import preprocess_french
from risk_budget_jccp.real_data.common.config import case_config, load_config, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, format_status, write_manifest
from risk_budget_jccp.real_data.common.logging_utils import configure_case_logger
from risk_budget_jccp.real_data.common.paths import ensure_case_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare Kenneth French real-data inputs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_common_config(config)
    cfg = case_config(config, "french")
    paths = ensure_case_dirs("french", args.output_root)
    logger = configure_case_logger("french.prepare", paths["logs"])
    status = check_case_data("french", cfg, raw_dir=paths["raw"], processed_dir=paths["processed"])
    logger.info("\n%s", format_status(status))
    if args.dry_run:
        return
    if status.valid and not args.force and not args.force_download:
        logger.info("French processed data already valid; skipping prepare.")
        return
    if args.force_download:
        for path in paths["raw"].glob("49_Industry_Portfolios_Daily*"):
            if path.is_file():
                path.unlink()
    raw_path = download_french(paths["raw"])
    preprocess_french(
        raw_path,
        paths["processed"],
        start_date=cfg["start_date"],
        train_end_date=cfg["train_end_date"],
        test_start_date=cfg["test_start_date"],
        test_end_date=cfg["test_end_date"],
        max_assets=cfg["max_assets"],
        max_train_scenarios=cfg["max_train_scenarios"],
        max_test_scenarios=cfg["max_test_scenarios"],
        heldout_policy=str(cfg.get("heldout_policy", "first_after_split")),
    )
    write_manifest(
        case="french",
        config=cfg,
        raw_inputs=[raw_path],
        processed_dir=paths["processed"],
        provenance_path=paths["raw"] / "provenance.json",
    )
    logger.info("French data prepared: %s", paths["processed"])


if __name__ == "__main__":
    main()
