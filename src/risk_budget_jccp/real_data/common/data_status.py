from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.common.logging_utils import read_json, write_json
from risk_budget_jccp.real_data.common.paths import PROCESSED_ROOT, RAW_ROOT, case_processed_dir, case_raw_dir


SCHEMA_VERSION = 1

REQUIRED_PROCESSED_FILES = {
    "m5": ("demand_train.csv", "demand_test.csv", "series_metadata.csv"),
    "french": ("returns_train.csv", "returns_test.csv", "industry_metadata.csv"),
    "power": ("power_instance.npz", "line_metadata.csv", "generator_metadata.csv"),
}


@dataclass(frozen=True)
class DataStatus:
    case: str
    raw_available: bool
    processed_available: bool
    manifest_available: bool
    valid: bool
    stale: bool
    message: str
    manifest: dict[str, Any] | None
    missing_files: tuple[str, ...]


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_path(case: str, processed_dir: str | Path | None = None) -> Path:
    root = Path(processed_dir) if processed_dir is not None else case_processed_dir(case)
    return root / "manifest.json"


def required_processed_paths(case: str, processed_dir: str | Path | None = None) -> list[Path]:
    if case not in REQUIRED_PROCESSED_FILES:
        raise ValueError(f"unknown real-data case {case!r}")
    root = Path(processed_dir) if processed_dir is not None else case_processed_dir(case)
    return [root / filename for filename in REQUIRED_PROCESSED_FILES[case]]


def raw_available(case: str, raw_dir: str | Path | None = None) -> bool:
    root = Path(raw_dir) if raw_dir is not None else case_raw_dir(case)
    if case == "m5":
        return (root / "m5-forecasting-accuracy.zip").is_file()
    if case == "french":
        return any(root.glob("49_Industry_Portfolios_Daily*"))
    if case == "power":
        return (root / "RTS-GMLC" / "RTS_Data").is_dir()
    raise ValueError(f"unknown real-data case {case!r}")


def _stats_for_case(case: str, processed_dir: Path) -> dict[str, int]:
    if case == "m5":
        train = pd.read_csv(processed_dir / "demand_train.csv", index_col=0)
        test = pd.read_csv(processed_dir / "demand_test.csv", index_col=0)
        return {
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "n_constraints": int(train.shape[1]),
        }
    if case == "french":
        train = pd.read_csv(processed_dir / "returns_train.csv", index_col=0)
        test = pd.read_csv(processed_dir / "returns_test.csv", index_col=0)
        return {
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "n_constraints": int(train.shape[1]),
        }
    if case == "power":
        arrays = np.load(processed_dir / "power_instance.npz", allow_pickle=True)
        train = arrays["flow_residual_train"]
        test = arrays["flow_residual_test"]
        return {
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "n_constraints": int(train.shape[1]),
        }
    raise ValueError(f"unknown real-data case {case!r}")


def write_manifest(
    *,
    case: str,
    config: dict[str, Any],
    raw_inputs: list[str | Path],
    processed_dir: str | Path | None = None,
    provenance_path: str | Path | None = None,
) -> Path:
    root = Path(processed_dir) if processed_dir is not None else case_processed_dir(case)
    missing = [str(path) for path in required_processed_paths(case, root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot write {case} manifest; missing processed files: {missing}")
    stats = _stats_for_case(case, root)
    outputs = [str(path) for path in required_processed_paths(case, root)]
    manifest = {
        "case": case,
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "raw_inputs": [str(Path(path)) for path in raw_inputs],
        "processed_outputs": outputs,
        "provenance_path": None if provenance_path is None else str(Path(provenance_path)),
        **stats,
    }
    return write_json(root / "manifest.json", manifest)


def check_case_data(
    case: str,
    config: dict[str, Any] | None = None,
    *,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> DataStatus:
    processed_root = Path(processed_dir) if processed_dir is not None else case_processed_dir(case)
    missing_paths = [path for path in required_processed_paths(case, processed_root) if not path.is_file()]
    manifest_file = manifest_path(case, processed_root)
    manifest = read_json(manifest_file) if manifest_file.is_file() else None
    processed_available = not missing_paths
    stale = False
    valid = False
    message_parts: list[str] = []

    if not processed_available:
        message_parts.append("processed files missing")
    if manifest is None:
        message_parts.append("manifest missing")
    else:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            stale = True
            message_parts.append("manifest schema version is stale")
        if manifest.get("status") != "valid":
            message_parts.append(f"manifest status is {manifest.get('status')!r}")
        if config is not None and manifest.get("config_hash") != config_hash(config):
            stale = True
            message_parts.append("config hash differs from manifest")
        try:
            stats = _stats_for_case(case, processed_root)
            for key, value in stats.items():
                if int(manifest.get(key, -1)) != int(value):
                    stale = True
                    message_parts.append(f"manifest {key} differs from processed data")
        except Exception as exc:
            message_parts.append(f"processed schema check failed: {exc}")

    valid = processed_available and manifest is not None and not stale and not message_parts
    if valid:
        message = "processed data valid"
    else:
        message = "; ".join(message_parts) if message_parts else "processed data invalid"

    return DataStatus(
        case=case,
        raw_available=raw_available(case, raw_dir),
        processed_available=processed_available,
        manifest_available=manifest is not None,
        valid=valid,
        stale=stale,
        message=message,
        manifest=manifest,
        missing_files=tuple(str(path) for path in missing_paths),
    )


def format_status(status: DataStatus) -> str:
    manifest = status.manifest or {}
    lines = [
        f"{status.case}:",
        f"  raw: {'available' if status.raw_available else 'missing'}",
        f"  processed: {'valid' if status.valid else ('stale' if status.stale else 'missing/invalid')}",
        f"  manifest: {'available' if status.manifest_available else 'missing'}",
        f"  message: {status.message}",
    ]
    for key in ("n_train", "n_test", "n_constraints"):
        if key in manifest:
            lines.append(f"  {key}: {manifest[key]}")
    if status.missing_files:
        lines.append("  missing_files:")
        lines.extend(f"    - {path}" for path in status.missing_files)
    if not status.valid:
        lines.append(f"  next action: python scripts/real_data/prepare_{status.case}_data.py --config configs/real_data/main.yaml")
    return "\n".join(lines)


def all_case_statuses(config_by_case: dict[str, dict[str, Any]]) -> list[DataStatus]:
    return [check_case_data(case, cfg) for case, cfg in config_by_case.items()]
