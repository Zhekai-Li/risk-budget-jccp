from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.cases.m5.download import download_m5
from risk_budget_jccp.real_data.cases.m5.preprocess import preprocess_m5
from risk_budget_jccp.real_data.common.config import case_config, load_config, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, format_status, write_manifest
from risk_budget_jccp.real_data.common.logging_utils import configure_case_logger
from risk_budget_jccp.real_data.common.paths import ensure_case_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare M5 real-data inputs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run preprocessing even if processed data are valid.")
    parser.add_argument("--force-download", action="store_true", help="Remove the cached M5 zip and download again.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_common_config(config)
    cfg = case_config(config, "m5")
    paths = ensure_case_dirs("m5", args.output_root)
    logger = configure_case_logger("m5.prepare", paths["logs"])
    status = check_case_data("m5", cfg, raw_dir=paths["raw"], processed_dir=paths["processed"])
    logger.info("\n%s", format_status(status))
    if args.dry_run:
        return
    if status.valid and not args.force and not args.force_download:
        logger.info("M5 processed data already valid; skipping prepare.")
        return
    zip_path = paths["raw"] / "m5-forecasting-accuracy.zip"
    if args.force_download and zip_path.exists():
        zip_path.unlink()
    raw_zip = download_m5(paths["raw"])
    preprocess_m5(
        raw_zip,
        paths["processed"],
        n_series=cfg["n_series"],
        max_train_days=cfg["max_train_days"],
        max_test_days=cfg["max_test_days"],
        min_active_days=cfg["min_active_days"],
        selection_policy=str(cfg.get("selection_policy", "stratified_revenue_cv_category_store")),
        max_category_share=float(cfg.get("max_category_share", 0.55)),
        revenue_quantile_bins=int(cfg.get("revenue_quantile_bins", 3)),
        cv_quantile_bins=int(cfg.get("cv_quantile_bins", 3)),
        stability_filter=bool(cfg.get("stability_filter", True)),
        min_calibration_mean_ratio=float(cfg.get("min_calibration_mean_ratio", 0.25)),
        max_calibration_mean_ratio=float(cfg.get("max_calibration_mean_ratio", 4.0)),
        max_calibration_zero_rate_shift=float(cfg.get("max_calibration_zero_rate_shift", 0.60)),
    )
    write_manifest(
        case="m5",
        config=cfg,
        raw_inputs=[raw_zip],
        processed_dir=paths["processed"],
        provenance_path=paths["raw"] / "provenance.json",
    )
    logger.info("M5 data prepared: %s", paths["processed"])


if __name__ == "__main__":
    main()
