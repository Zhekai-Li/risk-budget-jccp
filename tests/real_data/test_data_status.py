from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.common.data_status import check_case_data, write_manifest


def test_data_status_missing_manifest_is_invalid(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    status = check_case_data("m5", {"alpha": 0.05}, raw_dir=raw, processed_dir=processed)
    assert not status.valid
    assert not status.manifest_available
    assert status.missing_files


def test_data_status_valid_m5_manifest(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    (raw / "m5-forecasting-accuracy.zip").write_bytes(b"placeholder")
    pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).to_csv(processed / "demand_train.csv")
    pd.DataFrame([[2, 3]], columns=["A", "B"]).to_csv(processed / "demand_test.csv")
    pd.DataFrame({"id": ["A", "B"], "median_price": [1.0, 2.0]}).to_csv(
        processed / "series_metadata.csv",
        index=False,
    )
    cfg = {"alpha": 0.05, "n_series": 2}
    write_manifest(case="m5", config=cfg, raw_inputs=[raw / "m5-forecasting-accuracy.zip"], processed_dir=processed)
    status = check_case_data("m5", cfg, raw_dir=raw, processed_dir=processed)
    assert status.valid
    assert status.manifest is not None
    assert status.manifest["n_train"] == 2
    assert status.manifest["n_constraints"] == 2


def test_data_status_stale_config_hash(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    np.savez(
        processed / "power_instance.npz",
        flow_residual_train=np.ones((3, 2)),
        flow_residual_test=np.ones((2, 2)),
    )
    pd.DataFrame({"branch_id": ["L", "L"], "thermal_limit": [1.0, 1.0]}).to_csv(
        processed / "line_metadata.csv",
        index=False,
    )
    pd.DataFrame({"generator_id": ["G"], "cost": [1.0], "pmin": [0.0], "pmax": [10.0]}).to_csv(
        processed / "generator_metadata.csv",
        index=False,
    )
    cfg = {"alpha": 0.05}
    write_manifest(case="power", config=cfg, raw_inputs=[raw / "RTS-GMLC"], processed_dir=processed)
    status = check_case_data("power", {"alpha": 0.1}, raw_dir=raw, processed_dir=processed)
    assert not status.valid
    assert status.stale
