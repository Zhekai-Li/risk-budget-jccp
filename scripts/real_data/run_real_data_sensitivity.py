from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.common.config import DEFAULT_PROFILE, _deep_update, case_config, load_config, load_yaml, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, write_manifest
from risk_budget_jccp.real_data.common.paths import PROCESSED_ROOT, RAW_ROOT, RESULTS_ROOT, runtime_results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated real-data sensitivity profiles.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "sensitivity.yaml"))
    parser.add_argument("--require-prepared-data", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute sensitivity processed data and results.")
    parser.add_argument("--profile", action="append", help="Run only the named sensitivity profile; may be repeated.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    return parser.parse_args()


def _profile_config(entry: dict[str, object]) -> dict[str, object]:
    profile_name = str(entry.get("profile", "mac_m4"))
    raw = {
        "alpha": 0.05,
        "profile": profile_name,
        "random_seed": 20260525,
        "profiles": deepcopy(DEFAULT_PROFILE["profiles"]),
    }
    overrides = entry.get("overrides", {})
    if isinstance(overrides, dict):
        raw["profiles"][profile_name] = _deep_update(raw["profiles"][profile_name], overrides)
    merged = _deep_update(deepcopy(DEFAULT_PROFILE), raw)
    return load_config_from_mapping(merged, profile_name)


def load_config_from_mapping(config: dict[str, object], profile_name: str) -> dict[str, object]:
    if profile_name not in config["profiles"]:
        raise ValueError(f"unknown profile {profile_name!r}")
    profile = deepcopy(config["profiles"][profile_name])
    profile["alpha"] = float(config.get("alpha", 0.05))
    profile["profile"] = profile_name
    profile["random_seed"] = int(config.get("random_seed", profile.get("random_seed", 20260525)))
    return profile


def _move_split_outputs(run_root: Path, profile_root: Path, case: str) -> None:
    for split in ("in_sample", "out_of_sample"):
        src = run_root / split
        if not src.exists():
            continue
        dst = profile_root / split / case
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _run_m5(profile_name: str, cfg: dict[str, object], *, force: bool, require_prepared_data: bool) -> pd.DataFrame:
    from risk_budget_jccp.real_data.cases.m5.build_instance import build_instance
    from risk_budget_jccp.real_data.cases.m5.download import download_m5
    from risk_budget_jccp.real_data.cases.m5.preprocess import preprocess_m5
    from risk_budget_jccp.real_data.cases.m5.report import build_m5_report_assets
    from risk_budget_jccp.real_data.cases.m5.solve import solve_m5_instance

    case_cfg = case_config(cfg, "m5")
    raw_dir = RAW_ROOT / "m5"
    processed = PROCESSED_ROOT / "sensitivity" / profile_name / "m5"
    profile_root = RESULTS_ROOT / "sensitivity" / profile_name
    run_root = profile_root / "runs" / "m5"
    if force and processed.exists():
        shutil.rmtree(processed)
    if force and run_root.exists():
        shutil.rmtree(run_root)
    status = check_case_data("m5", case_cfg, raw_dir=raw_dir, processed_dir=processed)
    if force or not status.valid:
        raw_zip = raw_dir / "m5-forecasting-accuracy.zip"
        if require_prepared_data and not raw_zip.is_file():
            raise RuntimeError("M5 raw data are missing; run prepare_m5_data.py for the main profile first.")
        raw_zip = download_m5(raw_dir)
        preprocess_m5(
            raw_zip,
            processed,
            n_series=case_cfg["n_series"],
            max_train_days=case_cfg["max_train_days"],
            max_test_days=case_cfg["max_test_days"],
            min_active_days=case_cfg["min_active_days"],
            selection_policy=str(case_cfg.get("selection_policy", "stratified_revenue_cv_category_store")),
            max_category_share=float(case_cfg.get("max_category_share", 0.55)),
            revenue_quantile_bins=int(case_cfg.get("revenue_quantile_bins", 3)),
            cv_quantile_bins=int(case_cfg.get("cv_quantile_bins", 3)),
            stability_filter=bool(case_cfg.get("stability_filter", True)),
            min_calibration_mean_ratio=float(case_cfg.get("min_calibration_mean_ratio", 0.25)),
            max_calibration_mean_ratio=float(case_cfg.get("max_calibration_mean_ratio", 4.0)),
            max_calibration_zero_rate_shift=float(case_cfg.get("max_calibration_zero_rate_shift", 0.60)),
        )
        write_manifest(case="m5", config=case_cfg, raw_inputs=[raw_zip], processed_dir=processed)
    instance = build_instance(processed, alpha=case_cfg["alpha"])
    run_root.mkdir(parents=True, exist_ok=True)
    summary = solve_m5_instance(instance, run_root, case_cfg["dca"])
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_root / "tables" / "m5_summary.csv", index=False)
    build_m5_report_assets(instance, summary, run_root, processed_dir=processed)
    _move_split_outputs(run_root, profile_root, "m5")
    return summary.assign(profile=profile_name)


def _run_power(profile_name: str, cfg: dict[str, object], *, force: bool, require_prepared_data: bool) -> pd.DataFrame:
    from risk_budget_jccp.real_data.cases.power.build_instance import build_instance
    from risk_budget_jccp.real_data.cases.power.download import download_power
    from risk_budget_jccp.real_data.cases.power.preprocess import preprocess_power
    from risk_budget_jccp.real_data.cases.power.report import build_power_report_assets
    from risk_budget_jccp.real_data.cases.power.solve import solve_power_instance

    case_cfg = case_config(cfg, "power")
    raw_dir = RAW_ROOT / "power"
    processed = PROCESSED_ROOT / "sensitivity" / profile_name / "power"
    profile_root = RESULTS_ROOT / "sensitivity" / profile_name
    run_root = profile_root / "runs" / "power"
    if force and processed.exists():
        shutil.rmtree(processed)
    if force and run_root.exists():
        shutil.rmtree(run_root)
    status = check_case_data("power", case_cfg, raw_dir=raw_dir, processed_dir=processed)
    if force or not status.valid:
        repo = raw_dir / "RTS-GMLC"
        if require_prepared_data and not (repo / "RTS_Data").is_dir():
            raise RuntimeError("RTS-GMLC raw data are missing; run prepare_power_data.py for the main profile first.")
        repo = download_power(
            raw_dir,
            use_external_api=bool(case_cfg["use_external_wind_solar_api"]),
            fallback_to_builtin=bool(case_cfg["fallback_to_rts_builtin"]),
        )
        preprocess_power(
            repo,
            processed,
            alpha=case_cfg["alpha"],
            n_snapshots=case_cfg["n_snapshots"],
            max_branches=case_cfg["max_branches"],
            max_train_scenarios=case_cfg["max_train_scenarios"],
            max_test_scenarios=case_cfg["max_test_scenarios"],
            primary_rating=str(case_cfg.get("primary_rating", "lte")),
            snapshot_selection_policy=str(case_cfg.get("snapshot_selection_policy", "forecast_representative_diverse")),
            branch_selection_policy=str(case_cfg.get("branch_selection_policy", "stratified_tier_volatility")),
            risk_tier_policy=str(case_cfg.get("risk_tier_policy", "next_available_rating")),
            risk_limit_gamma_grid=case_cfg.get("risk_limit_gamma_grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
            target_calibration_joint_violation=float(case_cfg.get("target_calibration_joint_violation", 0.15)),
            calibration_joint_violation_band=case_cfg.get("calibration_joint_violation_band", [0.08, 0.35]),
            random_seed=int(case_cfg.get("random_seed", cfg.get("random_seed", 20260525))),
            tier_margin=float(case_cfg.get("tier_margin", 0.02)),
            near_cont_threshold=float(case_cfg.get("near_cont_threshold", 0.80)),
            exclude_above_ste=bool(case_cfg.get("exclude_above_ste", True)),
        )
        write_manifest(case="power", config=case_cfg, raw_inputs=[repo], processed_dir=processed)
    instance = build_instance(processed, alpha=case_cfg["alpha"])
    run_root.mkdir(parents=True, exist_ok=True)
    summary = solve_power_instance(instance, run_root, case_cfg["dca"])
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_root / "tables" / "power_summary.csv", index=False)
    build_power_report_assets(instance, summary, run_root, processed_dir=processed)
    _move_split_outputs(run_root, profile_root, "power")
    return summary.assign(profile=profile_name)


def _run_french(profile_name: str, cfg: dict[str, object], *, force: bool, require_prepared_data: bool) -> pd.DataFrame:
    from risk_budget_jccp.real_data.cases.french.build_instance import build_instance
    from risk_budget_jccp.real_data.cases.french.download import download_french
    from risk_budget_jccp.real_data.cases.french.preprocess import preprocess_french
    from risk_budget_jccp.real_data.cases.french.report import build_french_report_assets
    from risk_budget_jccp.real_data.cases.french.solve import solve_french_instance

    case_cfg = case_config(cfg, "french")
    raw_dir = RAW_ROOT / "french"
    processed = PROCESSED_ROOT / "sensitivity" / profile_name / "french"
    profile_root = RESULTS_ROOT / "sensitivity" / profile_name
    run_root = profile_root / "runs" / "french"
    if force and processed.exists():
        shutil.rmtree(processed)
    if force and run_root.exists():
        shutil.rmtree(run_root)
    status = check_case_data("french", case_cfg, raw_dir=raw_dir, processed_dir=processed)
    if force or not status.valid:
        if require_prepared_data and not any(raw_dir.glob("49_Industry_Portfolios_Daily*")):
            raise RuntimeError("French raw data are missing; run prepare_french_data.py for the main profile first.")
        raw_path = download_french(raw_dir)
        preprocess_french(
            raw_path,
            processed,
            start_date=case_cfg["start_date"],
            train_end_date=case_cfg["train_end_date"],
            test_start_date=case_cfg["test_start_date"],
            test_end_date=case_cfg["test_end_date"],
            max_assets=case_cfg["max_assets"],
            max_train_scenarios=case_cfg["max_train_scenarios"],
            max_test_scenarios=case_cfg["max_test_scenarios"],
            heldout_policy=str(case_cfg.get("heldout_policy", "first_after_split")),
        )
        write_manifest(
            case="french",
            config=case_cfg,
            raw_inputs=[raw_path],
            processed_dir=processed,
            provenance_path=raw_dir / "provenance.json",
        )
    instance = build_instance(
        processed,
        alpha=case_cfg["alpha"],
        max_weight=case_cfg["max_weight"],
        target_return_fraction=case_cfg["target_return_fraction"],
    )
    run_root.mkdir(parents=True, exist_ok=True)
    summary = solve_french_instance(instance, run_root, case_cfg["dca"])
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_root / "tables" / "french_summary.csv", index=False)
    build_french_report_assets(instance, summary, run_root, processed_dir=processed)
    _move_split_outputs(run_root, profile_root, "french")
    return summary.assign(profile=profile_name)


def main() -> None:
    global RESULTS_ROOT
    args = parse_args()
    RESULTS_ROOT = runtime_results_root(args.output_root)
    spec = load_yaml(args.config)
    requested = set(args.profile or [])
    frames: list[pd.DataFrame] = []
    for entry in spec.get("sensitivity_profiles", []):
        profile_name = str(entry["name"])
        if requested and profile_name not in requested:
            continue
        case = str(entry["case"])
        cfg = _profile_config(entry)
        validate_common_config(cfg)
        if case == "m5":
            frame = _run_m5(profile_name, cfg, force=args.force, require_prepared_data=args.require_prepared_data)
        elif case == "power":
            frame = _run_power(profile_name, cfg, force=args.force, require_prepared_data=args.require_prepared_data)
        elif case == "french":
            frame = _run_french(profile_name, cfg, force=args.force, require_prepared_data=args.require_prepared_data)
        else:
            raise ValueError(f"sensitivity runner does not support case={case!r}")
        frame.insert(0, "sensitivity_profile", profile_name)
        frames.append(frame)
    if frames:
        out = RESULTS_ROOT / "sensitivity" / "summary"
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out / "sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
