from __future__ import annotations

from io import StringIO
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.common.paths import case_processed_dir


MISSING_SENTINELS = {-99.99, -999.0, -999.99}


def _extract_csv_text(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"no CSV file found inside {path}")
            with archive.open(csv_names[0]) as handle:
                return handle.read().decode("latin1", errors="replace")
    return path.read_text(encoding="latin1", errors="replace")


def _parse_french_daily(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith(",") and "," in line and not line.lower().startswith(",annual"):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError("could not locate French CSV header row")
    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        first = line.split(",", 1)[0].strip()
        if not (len(first) == 8 and first.isdigit()):
            break
        data_lines.append(line)
    frame = pd.read_csv(StringIO("\n".join(data_lines)))
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col].astype(str), format="%Y%m%d")
    frame = frame.set_index(date_col).sort_index(kind="stable")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    for sentinel in MISSING_SENTINELS:
        frame = frame.mask(np.isclose(frame, sentinel))
    return frame / 100.0


def preprocess_french(
    raw_path: str | Path,
    processed_dir: str | Path | None = None,
    *,
    start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    max_assets: int,
    max_train_scenarios: int,
    max_test_scenarios: int,
    heldout_policy: str = "first_after_split",
) -> Path:
    output = Path(processed_dir) if processed_dir is not None else case_processed_dir("french")
    output.mkdir(parents=True, exist_ok=True)
    returns = _parse_french_daily(_extract_csv_text(Path(raw_path)))
    returns = returns.loc[pd.Timestamp(start_date) : pd.Timestamp(test_end_date)]
    returns = returns.iloc[:, : int(max_assets)]
    train = returns.loc[: pd.Timestamp(train_end_date)].tail(int(max_train_scenarios))
    test_window = returns.loc[pd.Timestamp(test_start_date) : pd.Timestamp(test_end_date)]
    if str(heldout_policy) == "first_after_split":
        test = test_window.head(int(max_test_scenarios))
    elif str(heldout_policy) == "latest_period":
        test = test_window.tail(int(max_test_scenarios))
    else:
        raise ValueError(f"unknown French heldout_policy={heldout_policy!r}")
    usable_columns = train.columns[~(train.isna().any(axis=0) | test.isna().any(axis=0))]
    train = train.loc[:, usable_columns]
    test = test.loc[:, usable_columns]
    if train.empty or test.empty:
        raise ValueError("French preprocessing produced empty calibration or held-out data")
    downside = returns.loc[train.index, train.columns].clip(upper=0.0)
    heldout_downside = test.clip(upper=0.0)
    train_vol = train.std(axis=0, ddof=0)
    heldout_vol = test.std(axis=0, ddof=0)
    metadata = pd.DataFrame(
        {
            "industry": train.columns,
            "mean_return": train.mean(axis=0).to_numpy(dtype=float),
            "volatility": train_vol.to_numpy(dtype=float),
            "downside_cvar_95": downside.quantile(0.05, axis=0).abs().to_numpy(dtype=float),
            "heldout_mean_return": test.mean(axis=0).to_numpy(dtype=float),
            "heldout_volatility": heldout_vol.to_numpy(dtype=float),
            "heldout_downside_cvar_95": heldout_downside.quantile(0.05, axis=0).abs().to_numpy(dtype=float),
            "heldout_to_calibration_volatility_ratio": np.divide(
                heldout_vol,
                train_vol,
                out=np.full(len(train.columns), np.nan, dtype=float),
                where=train_vol.to_numpy(dtype=float) > 0.0,
            ),
            "asset_selection_policy": "standard_49_industries" if int(max_assets) >= 49 else "first_n_industries",
            "heldout_policy": str(heldout_policy),
            "selection_uses_heldout": False,
        }
    )
    train.to_csv(output / "returns_train.csv")
    test.to_csv(output / "returns_test.csv")
    metadata.to_csv(output / "industry_metadata.csv", index=False)
    regime = metadata.loc[
        :,
        [
            "industry",
            "mean_return",
            "heldout_mean_return",
            "volatility",
            "heldout_volatility",
            "downside_cvar_95",
            "heldout_downside_cvar_95",
            "heldout_to_calibration_volatility_ratio",
        ],
    ].copy()
    regime.to_csv(output / "regime_shift.csv", index=False)
    split_metadata = {
        "case": "french",
        "split_policy": "time_ordered_calibration_then_heldout",
        "selection_uses_heldout": False,
        "start_date": str(pd.Timestamp(start_date).date()),
        "train_start_date": str(train.index.min().date()),
        "train_end_date": str(train.index.max().date()),
        "test_start_date": str(test.index.min().date()),
        "test_end_date": str(test.index.max().date()),
        "n_calibration_scenarios": int(train.shape[0]),
        "n_heldout_scenarios": int(test.shape[0]),
        "n_assets": int(train.shape[1]),
        "asset_selection_policy": "standard_49_industries" if int(max_assets) >= 49 else "first_n_industries",
        "heldout_policy": str(heldout_policy),
    }
    (output / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    pd.DataFrame([split_metadata]).to_csv(output / "split_metadata.csv", index=False)
    return output
