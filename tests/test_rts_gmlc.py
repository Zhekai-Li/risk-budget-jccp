from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_power_modules():
    try:
        from risk_budget_jccp.data.rts_gmlc import (
            _align_forecast_and_actual,
            build_chronological_error_train_test,
            build_dc_ptdf,
            build_rts_gmlc_power_dispatch_instance,
            load_rts_gmlc_regional_timeseries,
            load_rts_gmlc_source_data,
        )
        from risk_budget_jccp.models.dc_dispatch import PowerDispatchInstance
    except ImportError as exc:  # pragma: no cover - exercised in TDD red step
        pytest.fail(f"RTS-GMLC modules are not importable yet: {exc}")

    return {
        "PowerDispatchInstance": PowerDispatchInstance,
        "_align_forecast_and_actual": _align_forecast_and_actual,
        "build_chronological_error_train_test": build_chronological_error_train_test,
        "build_dc_ptdf": build_dc_ptdf,
        "build_rts_gmlc_power_dispatch_instance": build_rts_gmlc_power_dispatch_instance,
        "load_rts_gmlc_regional_timeseries": load_rts_gmlc_regional_timeseries,
        "load_rts_gmlc_source_data": load_rts_gmlc_source_data,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _expand_hourly_means_to_five_minute_frame(
    start: str,
    hourly_means: dict[str, list[float]],
) -> pd.DataFrame:
    time_index = pd.date_range(start, periods=12 * len(next(iter(hourly_means.values()))), freq="5min")
    values: dict[str, list[float]] = {}
    centered_offsets = np.arange(12, dtype=float) - 5.5
    for region, means in hourly_means.items():
        expanded: list[float] = []
        for mean in means:
            expanded.extend((mean + centered_offsets).tolist())
        values[region] = expanded

    return pd.DataFrame({"Time": time_index, **values})


def _build_minimal_rts_tree(root: Path) -> Path:
    rts_root = root / "RTS_Data"

    _write_csv(
        rts_root / "SourceData" / "bus.csv",
        pd.DataFrame(
            {
                "Bus ID": [101, 102, 103],
                "Bus Name": ["BUS-101", "BUS-102", "BUS-103"],
            }
        ),
    )
    _write_csv(
        rts_root / "SourceData" / "branch.csv",
        pd.DataFrame(
            {
                "UID": ["L1", "L2"],
                "From Bus": [101, 102],
                "To Bus": [102, 103],
                "X": [0.1, 0.2],
                "Cont Rating": [100.0, 90.0],
            }
        ),
    )
    _write_csv(
        rts_root / "SourceData" / "gen.csv",
        pd.DataFrame(
            {
                "GEN UID": ["G1", "G2"],
                "Bus ID": [101, 103],
                "PMin MW": [5.0, 10.0],
                "PMax MW": [50.0, 80.0],
            }
        ),
    )

    load_times = pd.date_range("2020-01-01 00:00", periods=4, freq="h")
    _write_csv(
        rts_root / "timeseries_data_files" / "Load" / "DAY_AHEAD_regional_Load.csv",
        pd.DataFrame(
            {
                "Time": load_times,
                "RegionA": [100.0, 102.0, 104.0, 106.0],
                "RegionB": [70.0, 72.0, 74.0, 76.0],
            }
        ),
    )
    _write_csv(
        rts_root / "timeseries_data_files" / "Load" / "REAL_TIME_regional_Load.csv",
        _expand_hourly_means_to_five_minute_frame(
            start="2020-01-01 00:00",
            hourly_means={
                "RegionA": [98.0, 101.0, 103.0, 108.0],
                "RegionB": [71.0, 70.0, 75.0, 77.0],
            },
        ),
    )

    wind_times = pd.date_range("2020-01-01 00:00", periods=4, freq="h")
    _write_csv(
        rts_root / "timeseries_data_files" / "WIND" / "DAY_AHEAD_wind.csv",
        pd.DataFrame(
            {
                "Time": wind_times,
                "WindRegion1": [30.0, 32.0, 29.0, 35.0],
            }
        ),
    )
    _write_csv(
        rts_root / "timeseries_data_files" / "WIND" / "REAL_TIME_wind.csv",
        _expand_hourly_means_to_five_minute_frame(
            start="2020-01-01 00:00",
            hourly_means={"WindRegion1": [28.0, 33.0, 31.0, 34.0]},
        ),
    )

    return rts_root


def test_build_dc_ptdf_returns_finite_branch_by_bus_matrix() -> None:
    modules = _load_power_modules()

    bus = pd.DataFrame({"Bus ID": [101, 102, 103]})
    branch = pd.DataFrame(
        {
            "UID": ["L1", "L2"],
            "From Bus": [101, 102],
            "To Bus": [102, 103],
            "X": [1.0, 1.0],
            "Cont Rating": [100.0, 100.0],
        }
    )

    ptdf = modules["build_dc_ptdf"](bus, branch, slack_bus=101)

    assert ptdf.shape == (2, 3)
    assert np.isfinite(ptdf).all()
    np.testing.assert_allclose(np.abs(ptdf @ np.array([-1.0, 1.0, 0.0])), [1.0, 0.0])
    np.testing.assert_allclose(np.abs(ptdf @ np.array([-1.0, 0.0, 1.0])), [1.0, 1.0])


def test_rts_gmlc_loader_reads_minimal_fixture_tree(tmp_path: Path) -> None:
    modules = _load_power_modules()
    rts_root = _build_minimal_rts_tree(tmp_path)

    source = modules["load_rts_gmlc_source_data"](rts_root)
    assert set(source) == {"branch", "bus", "gen"}
    assert list(source["bus"]["Bus ID"]) == [101, 102, 103]
    assert list(source["branch"]["UID"]) == ["L1", "L2"]
    assert list(source["gen"]["GEN UID"]) == ["G1", "G2"]

    timeseries = modules["load_rts_gmlc_regional_timeseries"](rts_root)
    assert set(timeseries) == {
        "load_day_ahead",
        "load_real_time",
        "wind_day_ahead",
        "wind_real_time",
    }
    assert list(timeseries["load_day_ahead"].columns) == ["RegionA", "RegionB"]
    assert list(timeseries["wind_real_time"].columns) == ["WindRegion1"]
    assert timeseries["load_day_ahead"].index.is_monotonic_increasing
    assert timeseries["wind_real_time"].index.is_monotonic_increasing

    instance = modules["build_rts_gmlc_power_dispatch_instance"](
        rts_root,
        slack_bus=101,
        train_fraction=0.5,
    )

    assert isinstance(instance, modules["PowerDispatchInstance"])
    assert instance.bus_ids == ("101", "102", "103")
    assert instance.branch_ids == ("L1", "L2")
    assert instance.generator_ids == ("G1", "G2")
    assert instance.ptdf.shape == (2, 3)
    assert instance.load_error_train.shape == (2, 2)
    assert instance.load_error_test.shape == (2, 2)
    assert instance.wind_error_train.shape == (2, 1)
    assert instance.wind_error_test.shape == (2, 1)
    np.testing.assert_allclose(
        instance.load_error_train,
        np.array([[-2.0, 1.0], [-1.0, -2.0]]),
    )
    np.testing.assert_allclose(
        instance.wind_error_train,
        np.array([[-2.0], [1.0]]),
    )


def test_build_chronological_error_train_test_uses_actual_minus_forecast() -> None:
    modules = _load_power_modules()

    timestamps = pd.to_datetime(
        [
            "2020-01-01 02:00",
            "2020-01-01 00:00",
            "2020-01-01 03:00",
            "2020-01-01 01:00",
            "2020-01-01 04:00",
        ]
    )
    forecast = pd.DataFrame(
        {
            "North": [12.0, 10.0, 13.0, 11.0, 14.0],
            "South": [22.0, 20.0, 23.0, 21.0, 24.0],
        },
        index=timestamps,
    )
    actual = pd.DataFrame(
        {
            "North": [10.0, 8.0, 10.0, 9.0, 12.0],
            "South": [21.0, 18.0, 20.0, 20.0, 22.0],
        },
        index=timestamps,
    )

    train_error, test_error = modules["build_chronological_error_train_test"](
        forecast,
        actual,
        train_fraction=0.6,
    )

    expected_error = (
        actual.sort_index(kind="stable") - forecast.sort_index(kind="stable")
    ).to_numpy(dtype=float)
    np.testing.assert_allclose(train_error, expected_error[:3])
    np.testing.assert_allclose(test_error, expected_error[3:])
    assert train_error.shape == (3, 2)
    assert test_error.shape == (2, 2)


def test_build_chronological_error_train_test_aggregates_five_minute_actual_to_hourly_mean() -> None:
    modules = _load_power_modules()

    forecast_index = pd.date_range("2020-01-01 00:00", periods=4, freq="h")
    forecast = pd.DataFrame({"North": [103.0, 203.0, 303.0, 403.0]}, index=forecast_index)

    actual = _expand_hourly_means_to_five_minute_frame(
        start="2020-01-01 00:00",
        hourly_means={"North": [105.5, 205.5, 305.5, 405.5]},
    ).set_index("Time")

    train_error, test_error = modules["build_chronological_error_train_test"](
        forecast,
        actual,
        train_fraction=0.5,
    )

    expected_error = np.full((4, 1), 2.5)
    np.testing.assert_allclose(train_error, expected_error[:2])
    np.testing.assert_allclose(test_error, expected_error[2:])


def test_align_forecast_and_actual_rejects_partial_column_overlap() -> None:
    modules = _load_power_modules()

    forecast_index = pd.date_range("2020-01-01 00:00", periods=3, freq="h")
    forecast = pd.DataFrame(
        {
            "North": [100.0, 101.0, 102.0],
            "South": [200.0, 201.0, 202.0],
        },
        index=forecast_index,
    )
    actual = pd.DataFrame(
        {
            "North": [99.0, 98.0, 97.0],
            "East": [150.0, 151.0, 152.0],
        },
        index=forecast_index,
    )

    with pytest.raises(ValueError, match="regional columns"):
        modules["_align_forecast_and_actual"](forecast, actual)


def test_align_forecast_and_actual_rejects_missing_forecast_intervals_after_resampling() -> None:
    modules = _load_power_modules()

    forecast_index = pd.date_range("2020-01-01 00:00", periods=3, freq="h")
    forecast = pd.DataFrame({"North": [100.0, 101.0, 102.0]}, index=forecast_index)

    actual = _expand_hourly_means_to_five_minute_frame(
        start="2020-01-01 00:00",
        hourly_means={"North": [99.0, 100.0]},
    ).set_index("Time")

    with pytest.raises(ValueError, match="timestamps"):
        modules["_align_forecast_and_actual"](forecast, actual)
