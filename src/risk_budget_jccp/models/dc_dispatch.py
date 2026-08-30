from dataclasses import dataclass

import numpy as np


def _readonly_float_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _readonly_int_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=int, copy=True)
    array.setflags(write=False)
    return array


def _validate_matrix_shape(values: np.ndarray, rows: int, columns: int, message: str) -> None:
    if values.ndim != 2 or values.shape != (rows, columns):
        raise ValueError(message)


def _validate_vector_length(values: np.ndarray, expected_length: int, message: str) -> None:
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class PowerDispatchInstance:
    bus_ids: tuple[str, ...]
    branch_ids: tuple[str, ...]
    generator_ids: tuple[str, ...]
    branch_from_bus: np.ndarray
    branch_to_bus: np.ndarray
    branch_reactance: np.ndarray
    branch_flow_limit_mw: np.ndarray
    generator_bus: np.ndarray
    generator_pmin_mw: np.ndarray
    generator_pmax_mw: np.ndarray
    ptdf: np.ndarray
    load_regions: tuple[str, ...]
    wind_regions: tuple[str, ...]
    load_timestamps: tuple[np.datetime64, ...]
    wind_timestamps: tuple[np.datetime64, ...]
    load_forecast: np.ndarray
    load_actual: np.ndarray
    wind_forecast: np.ndarray
    wind_actual: np.ndarray
    load_error_train: np.ndarray
    load_error_test: np.ndarray
    wind_error_train: np.ndarray
    wind_error_test: np.ndarray
    slack_bus_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus_ids", tuple(str(bus_id) for bus_id in self.bus_ids))
        object.__setattr__(self, "branch_ids", tuple(str(branch_id) for branch_id in self.branch_ids))
        object.__setattr__(
            self,
            "generator_ids",
            tuple(str(generator_id) for generator_id in self.generator_ids),
        )
        object.__setattr__(self, "load_regions", tuple(self.load_regions))
        object.__setattr__(self, "wind_regions", tuple(self.wind_regions))
        object.__setattr__(
            self,
            "load_timestamps",
            tuple(np.datetime64(timestamp) for timestamp in self.load_timestamps),
        )
        object.__setattr__(
            self,
            "wind_timestamps",
            tuple(np.datetime64(timestamp) for timestamp in self.wind_timestamps),
        )
        object.__setattr__(self, "branch_from_bus", _readonly_int_array(self.branch_from_bus))
        object.__setattr__(self, "branch_to_bus", _readonly_int_array(self.branch_to_bus))
        object.__setattr__(self, "branch_reactance", _readonly_float_array(self.branch_reactance))
        object.__setattr__(
            self,
            "branch_flow_limit_mw",
            _readonly_float_array(self.branch_flow_limit_mw),
        )
        object.__setattr__(self, "generator_bus", _readonly_int_array(self.generator_bus))
        object.__setattr__(self, "generator_pmin_mw", _readonly_float_array(self.generator_pmin_mw))
        object.__setattr__(self, "generator_pmax_mw", _readonly_float_array(self.generator_pmax_mw))
        object.__setattr__(self, "ptdf", _readonly_float_array(self.ptdf))
        object.__setattr__(self, "load_forecast", _readonly_float_array(self.load_forecast))
        object.__setattr__(self, "load_actual", _readonly_float_array(self.load_actual))
        object.__setattr__(self, "wind_forecast", _readonly_float_array(self.wind_forecast))
        object.__setattr__(self, "wind_actual", _readonly_float_array(self.wind_actual))
        object.__setattr__(self, "load_error_train", _readonly_float_array(self.load_error_train))
        object.__setattr__(self, "load_error_test", _readonly_float_array(self.load_error_test))
        object.__setattr__(self, "wind_error_train", _readonly_float_array(self.wind_error_train))
        object.__setattr__(self, "wind_error_test", _readonly_float_array(self.wind_error_test))
        object.__setattr__(self, "slack_bus_id", str(self.slack_bus_id))

        bus_count = len(self.bus_ids)
        branch_count = len(self.branch_ids)
        generator_count = len(self.generator_ids)
        load_region_count = len(self.load_regions)
        wind_region_count = len(self.wind_regions)

        _validate_vector_length(
            self.branch_from_bus,
            branch_count,
            "branch_from_bus length must match number of branches",
        )
        _validate_vector_length(
            self.branch_to_bus,
            branch_count,
            "branch_to_bus length must match number of branches",
        )
        _validate_vector_length(
            self.branch_reactance,
            branch_count,
            "branch_reactance length must match number of branches",
        )
        _validate_vector_length(
            self.branch_flow_limit_mw,
            branch_count,
            "branch_flow_limit_mw length must match number of branches",
        )
        _validate_vector_length(
            self.generator_bus,
            generator_count,
            "generator_bus length must match number of generators",
        )
        _validate_vector_length(
            self.generator_pmin_mw,
            generator_count,
            "generator_pmin_mw length must match number of generators",
        )
        _validate_vector_length(
            self.generator_pmax_mw,
            generator_count,
            "generator_pmax_mw length must match number of generators",
        )
        _validate_matrix_shape(
            self.ptdf,
            branch_count,
            bus_count,
            "ptdf must have shape (num_branches, num_buses)",
        )

        _validate_matrix_shape(
            self.load_forecast,
            len(self.load_timestamps),
            load_region_count,
            "load_forecast shape must match load timestamps and regions",
        )
        _validate_matrix_shape(
            self.load_actual,
            len(self.load_timestamps),
            load_region_count,
            "load_actual shape must match load timestamps and regions",
        )
        _validate_matrix_shape(
            self.wind_forecast,
            len(self.wind_timestamps),
            wind_region_count,
            "wind_forecast shape must match wind timestamps and regions",
        )
        _validate_matrix_shape(
            self.wind_actual,
            len(self.wind_timestamps),
            wind_region_count,
            "wind_actual shape must match wind timestamps and regions",
        )

        if self.load_error_train.ndim != 2 or self.load_error_train.shape[1] != load_region_count:
            raise ValueError("load_error_train must be a 2D array with one column per load region")
        if self.load_error_test.ndim != 2 or self.load_error_test.shape[1] != load_region_count:
            raise ValueError("load_error_test must be a 2D array with one column per load region")
        if self.wind_error_train.ndim != 2 or self.wind_error_train.shape[1] != wind_region_count:
            raise ValueError("wind_error_train must be a 2D array with one column per wind region")
        if self.wind_error_test.ndim != 2 or self.wind_error_test.shape[1] != wind_region_count:
            raise ValueError("wind_error_test must be a 2D array with one column per wind region")

        if self.slack_bus_id not in self.bus_ids:
            raise ValueError("slack_bus_id must be present in bus_ids")
        if np.any(self.branch_flow_limit_mw <= 0.0):
            raise ValueError("branch_flow_limit_mw must be strictly positive")
        if np.any(~np.isfinite(self.ptdf)):
            raise ValueError("ptdf must contain only finite values")
