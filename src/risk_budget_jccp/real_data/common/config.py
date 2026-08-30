from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROFILE: dict[str, Any] = {
    "alpha": 0.05,
    "profile": "mac_m4",
    "random_seed": 20260525,
    "profiles": {
        "mac_m4": {
            "m5": {
                "n_series": 150,
                "max_train_days": 730,
                "max_test_days": 365,
                "min_active_days": 120,
                "selection_metric": "revenue_mean",
                "selection_policy": "stratified_revenue_cv_category_store",
                "max_category_share": 0.55,
                "revenue_quantile_bins": 3,
                "cv_quantile_bins": 3,
                "stability_filter": True,
                "min_calibration_mean_ratio": 0.25,
                "max_calibration_mean_ratio": 4.0,
                "max_calibration_zero_rate_shift": 0.60,
                "cost_proxy": "median_sell_price",
            },
            "power": {
                "n_snapshots": 3,
                "max_branches": 3,
                "max_train_scenarios": 100,
                "max_test_scenarios": 300,
                "use_external_wind_solar_api": True,
                "fallback_to_rts_builtin": True,
                "primary_rating": "ste",
                "sensitivity_ratings": ["ste"],
                "report_continuous_overload": True,
                "allow_nominal_cont_overload": True,
                "flag_nominal_lte_overload": True,
                "snapshot_selection_policy": "forecast_representative_diverse",
                "branch_selection_policy": "stratified_tier_volatility",
                "risk_tier_policy": "next_available_rating",
                "risk_limit_gamma_grid": [0.0, 0.25, 0.5, 0.75, 0.875, 1.0],
                "target_calibration_joint_violation": 0.15,
                "calibration_joint_violation_band": [0.05, 0.30],
                "random_seed": 20260525,
                "tier_margin": 0.02,
                "near_cont_threshold": 0.80,
                "exclude_above_ste": True,
            },
            "french": {
                "max_assets": 49,
                "max_train_scenarios": 1500,
                "max_test_scenarios": 1000,
                "max_weight": 0.12,
                "target_return_fraction": 0.60,
                "start_date": "2010-01-01",
                "train_end_date": "2018-12-31",
                "test_start_date": "2019-01-01",
                "test_end_date": "2024-12-31",
                "asset_selection_policy": "standard_49_industries",
                "heldout_policy": "first_after_split",
            },
            "dca": {
                "max_iter": 25,
                "tol": 1.0e-4,
                "proximal_weight": 1.0e-4,
                "min_alpha": 1.0e-8,
                "feasibility_tol": 1.0e-6,
                "certificate_accept_tol": 1.0e-4,
                "certificate_strict_tol": 1.0e-6,
            },
        }
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_PROFILE)
    if path is not None:
        config = _deep_update(config, load_yaml(path))
    profile_name = str(config.get("profile", "mac_m4"))
    if profile_name not in config["profiles"]:
        raise ValueError(f"unknown profile {profile_name!r}")
    profile = deepcopy(config["profiles"][profile_name])
    for key in ("alpha", "profile"):
        profile[key] = config[key]
    profile["random_seed"] = int(config.get("random_seed", profile.get("random_seed", 20260525)))
    for key, value in config.items():
        if key not in {"profiles", "profile"}:
            if isinstance(value, dict) and isinstance(profile.get(key), dict):
                profile[key] = _deep_update(profile[key], value)
            elif key != "alpha":
                profile[key] = deepcopy(value)
    return profile


def case_config(config: dict[str, Any], case: str) -> dict[str, Any]:
    if case not in config:
        raise ValueError(f"config is missing case section {case!r}")
    result = deepcopy(config[case])
    result["alpha"] = float(config.get("alpha", 0.05))
    result["dca"] = deepcopy(config.get("dca", DEFAULT_PROFILE["profiles"]["mac_m4"]["dca"]))
    return result


def validate_common_config(config: dict[str, Any]) -> None:
    alpha = float(config.get("alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    dca = config.get("dca", {})
    for name in ("max_iter",):
        if int(dca.get(name, 0)) <= 0:
            raise ValueError(f"dca.{name} must be positive")
    for name in ("tol", "proximal_weight", "min_alpha", "feasibility_tol"):
        if float(dca.get(name, 0.0)) <= 0.0:
            raise ValueError(f"dca.{name} must be positive")
    for name in ("certificate_accept_tol", "certificate_strict_tol"):
        if float(dca.get(name, DEFAULT_PROFILE["profiles"]["mac_m4"]["dca"][name])) <= 0.0:
            raise ValueError(f"dca.{name} must be positive")
