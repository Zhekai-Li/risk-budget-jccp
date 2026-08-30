from pathlib import Path
import re

import numpy as np
import pandas as pd

from risk_budget_jccp.models.dc_dispatch import PowerDispatchInstance


SOURCE_DATA_FILES = {
    "bus": ("SourceData", "bus.csv"),
    "branch": ("SourceData", "branch.csv"),
    "gen": ("SourceData", "gen.csv"),
}

REGIONAL_TIMESERIES_FILES = {
    "load_day_ahead": ("timeseries_data_files", "Load", "DAY_AHEAD_regional_Load.csv"),
    "load_real_time": ("timeseries_data_files", "Load", "REAL_TIME_regional_Load.csv"),
    "wind_day_ahead": ("timeseries_data_files", "WIND", "DAY_AHEAD_wind.csv"),
    "wind_real_time": ("timeseries_data_files", "WIND", "REAL_TIME_wind.csv"),
}


def _normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_rts_data_root(root: str | Path) -> Path:
    candidate = Path(root)
    if candidate.name == "RTS_Data":
        return candidate

    nested_candidate = candidate / "RTS_Data"
    if nested_candidate.is_dir():
        return nested_candidate
    return candidate


def _require_existing_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required RTS-GMLC file not found: {path}")
    return path


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match is not None:
            return match
    raise KeyError(f"missing required column; tried {candidates}")


def _coerce_string_ids(values: pd.Series) -> tuple[str, ...]:
    return tuple(str(value) for value in values.tolist())


def _first_finite_positive_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> np.ndarray:
    selected = None
    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        matches = [
            column
            for column in frame.columns
            if _normalize_column_name(column) == normalized_candidate
        ]
        if not matches:
            continue
        series = pd.to_numeric(frame[matches[0]], errors="coerce")
        if selected is None:
            selected = series
        else:
            selected = selected.where(selected > 0.0, series)

    if selected is None:
        raise KeyError(f"missing required rating column; tried {candidates}")

    values = selected.to_numpy(dtype=float, copy=True)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("branch ratings must be finite and strictly positive")
    return values


def _build_timestamp_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    direct_time_candidates = ("Time", "Timestamp", "Datetime", "DateTime")
    normalized = {_normalize_column_name(column): column for column in frame.columns}

    for candidate in direct_time_candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match is None:
            continue
        timestamp = pd.to_datetime(frame[match], utc=False)
        return pd.DatetimeIndex(timestamp, name="Time")

    year_match = normalized.get("year")
    month_match = normalized.get("month")
    day_match = normalized.get("day")
    period_match = normalized.get("period")
    if year_match and month_match and day_match and period_match:
        date = pd.to_datetime(
            {
                "year": frame[year_match],
                "month": frame[month_match],
                "day": frame[day_match],
            },
            utc=False,
        )
        per_day = (
            frame.groupby([year_match, month_match, day_match], sort=False)[period_match]
            .count()
            .max()
        )
        if per_day <= 0:
            raise ValueError("period-based RTS-GMLC timeseries must have at least one row per day")
        minutes_per_period = int(round(24 * 60 / per_day))
        offset = pd.to_timedelta(frame[period_match].to_numpy(dtype=int, copy=False) - 1, unit="m")
        offset = offset * minutes_per_period
        return pd.DatetimeIndex(date + offset, name="Time")

    raise KeyError("could not infer timestamp columns for RTS-GMLC timeseries")


def _read_regional_timeseries_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    timestamp_index = _build_timestamp_index(frame)

    time_columns = {
        column
        for column in frame.columns
        if _normalize_column_name(column) in {"time", "timestamp", "datetime", "year", "month", "day", "period"}
    }
    value_columns = [column for column in frame.columns if column not in time_columns]
    if not value_columns:
        raise ValueError(f"no regional value columns found in {path}")

    regional = frame.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
    if regional.isna().any().any():
        raise ValueError(f"non-numeric regional values found in {path}")

    regional.index = timestamp_index
    regional = regional.sort_index(kind="stable")
    return regional


def _infer_regular_cadence(index: pd.DatetimeIndex, label: str) -> pd.Timedelta:
    if index.has_duplicates:
        raise ValueError(f"{label} timestamps must be unique")
    if len(index) < 2:
        raise ValueError(f"{label} timeseries must contain at least two timestamps")

    deltas = index.to_series().diff().dropna().unique()
    if len(deltas) != 1:
        raise ValueError(f"{label} timestamps must have a regular cadence")

    cadence = pd.Timedelta(deltas[0])
    if cadence <= pd.Timedelta(0):
        raise ValueError(f"{label} timestamps must increase strictly")
    return cadence


def _validate_matching_columns(
    forecast_columns: pd.Index,
    actual_columns: pd.Index,
) -> pd.Index:
    forecast_set = set(forecast_columns)
    actual_set = set(actual_columns)
    if forecast_set != actual_set:
        missing_from_actual = sorted(forecast_set - actual_set)
        extra_in_actual = sorted(actual_set - forecast_set)
        raise ValueError(
            "forecast and actual timeseries must have identical regional columns; "
            f"missing from actual={missing_from_actual}, extra in actual={extra_in_actual}"
        )
    return forecast_columns


def _aggregate_actual_to_forecast_cadence(
    actual: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
    forecast_cadence: pd.Timedelta,
    actual_cadence: pd.Timedelta,
) -> pd.DataFrame:
    if actual_cadence > forecast_cadence:
        raise ValueError("actual timeseries cadence cannot be coarser than forecast cadence")
    if forecast_cadence.value % actual_cadence.value != 0:
        raise ValueError("actual cadence must divide forecast cadence exactly")

    expected_samples = forecast_cadence.value // actual_cadence.value
    resample_kwargs = {
        "rule": forecast_cadence,
        "origin": forecast_index[0],
        "label": "left",
        "closed": "left",
    }
    # RTS-GMLC real-time load/wind series are aggregated to the day-ahead cadence by
    # taking the mean over each forecast interval, so errors remain on forecast timestamps.
    aggregated = actual.resample(**resample_kwargs).mean().reindex(forecast_index)
    counts = actual.resample(**resample_kwargs).count().reindex(forecast_index)

    if aggregated.isna().any().any() or not counts.eq(expected_samples).all().all():
        raise ValueError(
            "actual timeseries do not fully cover forecast timestamps after cadence reconciliation"
        )
    return aggregated


def _align_forecast_and_actual(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_sorted = forecast.sort_index(kind="stable")
    actual_sorted = actual.sort_index(kind="stable")
    ordered_columns = _validate_matching_columns(forecast_sorted.columns, actual_sorted.columns)
    actual_sorted = actual_sorted.loc[:, ordered_columns]

    forecast_cadence = _infer_regular_cadence(forecast_sorted.index, "forecast")
    actual_cadence = _infer_regular_cadence(actual_sorted.index, "actual")

    if actual_cadence == forecast_cadence:
        if not actual_sorted.index.equals(forecast_sorted.index):
            raise ValueError("forecast and actual timeseries must have identical timestamps")
        return forecast_sorted.copy(), actual_sorted.copy()

    actual_aligned = _aggregate_actual_to_forecast_cadence(
        actual_sorted,
        forecast_index=forecast_sorted.index,
        forecast_cadence=forecast_cadence,
        actual_cadence=actual_cadence,
    )
    return forecast_sorted.copy(), actual_aligned.loc[:, ordered_columns].copy()


def load_rts_gmlc_source_data(root: str | Path) -> dict[str, pd.DataFrame]:
    rts_root = _resolve_rts_data_root(root)
    return {
        name: pd.read_csv(_require_existing_file(rts_root.joinpath(*parts)))
        for name, parts in SOURCE_DATA_FILES.items()
    }


def load_rts_gmlc_regional_timeseries(root: str | Path) -> dict[str, pd.DataFrame]:
    rts_root = _resolve_rts_data_root(root)
    return {
        name: _read_regional_timeseries_csv(_require_existing_file(rts_root.joinpath(*parts)))
        for name, parts in REGIONAL_TIMESERIES_FILES.items()
    }


def build_chronological_error_train_test(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must satisfy 0 < train_fraction < 1")

    forecast_aligned, actual_aligned = _align_forecast_and_actual(forecast, actual)
    errors = actual_aligned - forecast_aligned
    split_index = int(np.floor(len(errors) * train_fraction))
    if split_index <= 0 or split_index >= len(errors):
        raise ValueError("train_fraction must yield non-empty train and test splits")

    error_train = errors.iloc[:split_index].to_numpy(dtype=float, copy=True)
    error_test = errors.iloc[split_index:].to_numpy(dtype=float, copy=True)
    return error_train, error_test


def _extract_network_arrays(
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    gen: pd.DataFrame,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    bus_id_column = _find_column(bus, ("Bus ID", "Bus"))
    branch_from_column = _find_column(branch, ("From Bus", "From", "F_BUS"))
    branch_to_column = _find_column(branch, ("To Bus", "To", "T_BUS"))
    branch_x_column = _find_column(branch, ("X", "BR_X", "Reactance"))
    gen_bus_column = _find_column(gen, ("Bus ID", "Bus", "GEN_BUS"))
    gen_pmin_column = _find_column(gen, ("PMin MW", "Pmin MW", "PMIN"))
    gen_pmax_column = _find_column(gen, ("PMax MW", "Pmax MW", "PMAX"))

    branch_id_column = None
    for candidate in ("UID", "Line", "Branch ID"):
        try:
            branch_id_column = _find_column(branch, (candidate,))
            break
        except KeyError:
            continue
    if branch_id_column is None:
        branch_ids = tuple(str(index) for index in branch.index)
    else:
        branch_ids = _coerce_string_ids(branch[branch_id_column])

    generator_id_column = None
    for candidate in ("GEN UID", "Gen UID", "UID", "Generator ID"):
        try:
            generator_id_column = _find_column(gen, (candidate,))
            break
        except KeyError:
            continue
    if generator_id_column is None:
        generator_ids = tuple(str(index) for index in gen.index)
    else:
        generator_ids = _coerce_string_ids(gen[generator_id_column])

    bus_ids = _coerce_string_ids(bus[bus_id_column])
    bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    branch_from_ids = _coerce_string_ids(branch[branch_from_column])
    branch_to_ids = _coerce_string_ids(branch[branch_to_column])
    gen_bus_ids = _coerce_string_ids(gen[gen_bus_column])

    try:
        branch_from_bus = np.array([bus_index[bus_id] for bus_id in branch_from_ids], dtype=int)
        branch_to_bus = np.array([bus_index[bus_id] for bus_id in branch_to_ids], dtype=int)
        generator_bus = np.array([bus_index[bus_id] for bus_id in gen_bus_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"network references unknown bus id: {exc.args[0]}") from exc

    branch_reactance = pd.to_numeric(branch[branch_x_column], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if np.any(~np.isfinite(branch_reactance)) or np.any(branch_reactance == 0.0):
        raise ValueError("branch reactance values must be finite and nonzero")

    generator_pmin_mw = pd.to_numeric(gen[gen_pmin_column], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    generator_pmax_mw = pd.to_numeric(gen[gen_pmax_column], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if np.any(~np.isfinite(generator_pmin_mw)) or np.any(~np.isfinite(generator_pmax_mw)):
        raise ValueError("generator bounds must be finite")

    return {
        "bus_ids": bus_ids,
        "branch_ids": branch_ids,
        "generator_ids": generator_ids,
        "branch_from_bus": branch_from_bus,
        "branch_to_bus": branch_to_bus,
        "branch_reactance": branch_reactance,
        "branch_flow_limit_mw": _first_finite_positive_column(
            branch,
            ("Cont Rating", "LTE Rating", "RATE_A", "Rate A"),
        ),
        "generator_bus": generator_bus,
        "generator_pmin_mw": generator_pmin_mw,
        "generator_pmax_mw": generator_pmax_mw,
    }


def build_dc_ptdf(
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    slack_bus: str | int,
) -> np.ndarray:
    bus_id_column = _find_column(bus, ("Bus ID", "Bus"))
    branch_from_column = _find_column(branch, ("From Bus", "From", "F_BUS"))
    branch_to_column = _find_column(branch, ("To Bus", "To", "T_BUS"))
    branch_x_column = _find_column(branch, ("X", "BR_X", "Reactance"))

    bus_ids = _coerce_string_ids(bus[bus_id_column])
    slack_bus_id = str(slack_bus)
    if slack_bus_id not in bus_ids:
        raise ValueError(f"slack bus {slack_bus} is not present in bus data")

    bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    from_index = np.array(
        [bus_index[str(bus_id)] for bus_id in branch[branch_from_column]],
        dtype=int,
    )
    to_index = np.array(
        [bus_index[str(bus_id)] for bus_id in branch[branch_to_column]],
        dtype=int,
    )
    reactance = pd.to_numeric(branch[branch_x_column], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if np.any(~np.isfinite(reactance)) or np.any(reactance == 0.0):
        raise ValueError("branch reactance values must be finite and nonzero")

    branch_count = len(branch)
    bus_count = len(bus)
    incidence = np.zeros((branch_count, bus_count), dtype=float)
    incidence[np.arange(branch_count), from_index] = 1.0
    incidence[np.arange(branch_count), to_index] = -1.0

    susceptance = 1.0 / reactance
    weighted_incidence = susceptance[:, None] * incidence
    bbus = incidence.T @ weighted_incidence

    slack_index = bus_index[slack_bus_id]
    keep = np.arange(bus_count) != slack_index
    reduced_bbus = bbus[np.ix_(keep, keep)]
    reduced_inverse = np.linalg.inv(reduced_bbus)

    ptdf = np.zeros((branch_count, bus_count), dtype=float)
    non_slack_positions = np.flatnonzero(keep)
    for reduced_column, bus_column in enumerate(non_slack_positions):
        injection = np.zeros(bus_count - 1, dtype=float)
        injection[reduced_column] = 1.0
        theta_reduced = reduced_inverse @ injection
        theta = np.zeros(bus_count, dtype=float)
        theta[keep] = theta_reduced
        ptdf[:, bus_column] = weighted_incidence @ theta

    return ptdf


def build_rts_gmlc_power_dispatch_instance(
    root: str | Path,
    slack_bus: str | int,
    train_fraction: float,
) -> PowerDispatchInstance:
    source = load_rts_gmlc_source_data(root)
    timeseries = load_rts_gmlc_regional_timeseries(root)
    network = _extract_network_arrays(source["bus"], source["branch"], source["gen"])
    ptdf = build_dc_ptdf(source["bus"], source["branch"], slack_bus=slack_bus)

    load_forecast, load_actual = _align_forecast_and_actual(
        timeseries["load_day_ahead"],
        timeseries["load_real_time"],
    )
    wind_forecast, wind_actual = _align_forecast_and_actual(
        timeseries["wind_day_ahead"],
        timeseries["wind_real_time"],
    )

    load_error_train, load_error_test = build_chronological_error_train_test(
        load_forecast,
        load_actual,
        train_fraction=train_fraction,
    )
    wind_error_train, wind_error_test = build_chronological_error_train_test(
        wind_forecast,
        wind_actual,
        train_fraction=train_fraction,
    )

    return PowerDispatchInstance(
        bus_ids=network["bus_ids"],
        branch_ids=network["branch_ids"],
        generator_ids=network["generator_ids"],
        branch_from_bus=network["branch_from_bus"],
        branch_to_bus=network["branch_to_bus"],
        branch_reactance=network["branch_reactance"],
        branch_flow_limit_mw=network["branch_flow_limit_mw"],
        generator_bus=network["generator_bus"],
        generator_pmin_mw=network["generator_pmin_mw"],
        generator_pmax_mw=network["generator_pmax_mw"],
        ptdf=ptdf,
        load_regions=tuple(load_forecast.columns),
        wind_regions=tuple(wind_forecast.columns),
        load_timestamps=tuple(load_forecast.index.to_numpy(dtype="datetime64[ns]")),
        wind_timestamps=tuple(wind_forecast.index.to_numpy(dtype="datetime64[ns]")),
        load_forecast=load_forecast.to_numpy(dtype=float, copy=True),
        load_actual=load_actual.to_numpy(dtype=float, copy=True),
        wind_forecast=wind_forecast.to_numpy(dtype=float, copy=True),
        wind_actual=wind_actual.to_numpy(dtype=float, copy=True),
        load_error_train=load_error_train,
        load_error_test=load_error_test,
        wind_error_train=wind_error_train,
        wind_error_test=wind_error_test,
        slack_bus_id=str(slack_bus),
    )
