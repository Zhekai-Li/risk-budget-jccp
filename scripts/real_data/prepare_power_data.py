from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.cases.power.download import download_power
from risk_budget_jccp.real_data.cases.power.preprocess import preprocess_power
from risk_budget_jccp.real_data.common.config import case_config, load_config, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, format_status, write_manifest
from risk_budget_jccp.real_data.common.logging_utils import configure_case_logger
from risk_budget_jccp.real_data.common.paths import ensure_case_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare RTS-GMLC power real-data inputs.")
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
    cfg = case_config(config, "power")
    paths = ensure_case_dirs("power", args.output_root)
    logger = configure_case_logger("power.prepare", paths["logs"])
    status = check_case_data("power", cfg, raw_dir=paths["raw"], processed_dir=paths["processed"])
    logger.info("\n%s", format_status(status))
    if args.dry_run:
        return
    if status.valid and not args.force and not args.force_download:
        logger.info("Power processed data already valid; skipping prepare.")
        return
    repo_dir = paths["raw"] / "RTS-GMLC"
    if args.force_download and repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo = download_power(
        paths["raw"],
        use_external_api=bool(cfg["use_external_wind_solar_api"]),
        fallback_to_builtin=bool(cfg["fallback_to_rts_builtin"]),
    )
    preprocess_power(
        repo,
        paths["processed"],
        alpha=cfg["alpha"],
        n_snapshots=cfg["n_snapshots"],
        max_branches=cfg["max_branches"],
        max_train_scenarios=cfg["max_train_scenarios"],
        max_test_scenarios=cfg["max_test_scenarios"],
        primary_rating=str(cfg.get("primary_rating", "lte")),
        snapshot_selection_policy=str(cfg.get("snapshot_selection_policy", "forecast_representative_diverse")),
        branch_selection_policy=str(cfg.get("branch_selection_policy", "stratified_tier_volatility")),
        risk_tier_policy=str(cfg.get("risk_tier_policy", "next_available_rating")),
        risk_limit_gamma_grid=cfg.get("risk_limit_gamma_grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
        target_calibration_joint_violation=float(cfg.get("target_calibration_joint_violation", 0.15)),
        calibration_joint_violation_band=cfg.get("calibration_joint_violation_band", [0.08, 0.35]),
        random_seed=int(cfg.get("random_seed", config.get("random_seed", 20260525))),
        tier_margin=float(cfg.get("tier_margin", 0.02)),
        near_cont_threshold=float(cfg.get("near_cont_threshold", 0.80)),
        exclude_above_ste=bool(cfg.get("exclude_above_ste", True)),
    )
    write_manifest(
        case="power",
        config=cfg,
        raw_inputs=[repo],
        processed_dir=paths["processed"],
        provenance_path=paths["raw"] / "provenance.json",
    )
    logger.info("Power data prepared: %s", paths["processed"])


if __name__ == "__main__":
    main()
