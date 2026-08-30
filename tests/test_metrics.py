import numpy as np
import pandas as pd
import pytest

from risk_budget_jccp.evaluation.metrics import normalized_entropy, relative_improvement_percent
from risk_budget_jccp.utils.io import load_yaml, write_csv


def test_normalized_entropy_is_one_for_uniform_allocation() -> None:
    alpha_vec = np.full(5, 0.01)
    assert np.isclose(normalized_entropy(alpha_vec), 1.0)


@pytest.mark.parametrize(
    ("alpha_vec", "message"),
    [
        (np.ones((2, 2)), "one-dimensional"),
        (np.array([]), "non-empty"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, 0.0]), "strictly positive"),
        (np.array([1.0, -1.0]), "strictly positive"),
    ],
)
def test_normalized_entropy_rejects_invalid_inputs(alpha_vec: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_entropy(alpha_vec)


def test_normalized_entropy_returns_one_for_single_element() -> None:
    assert normalized_entropy(np.array([42.0])) == 1.0


def test_normalized_entropy_handles_large_finite_values_without_overflow() -> None:
    value = normalized_entropy(np.array([1e308, 1e308]))
    assert np.isfinite(value)
    assert np.isclose(value, 1.0)


def test_relative_improvement_percent_is_positive_for_better_solution() -> None:
    assert np.isclose(relative_improvement_percent(equal_value=10.0, optimized_value=8.0), 20.0)


@pytest.mark.parametrize(
    ("equal_value", "optimized_value", "message"),
    [
        (np.inf, 8.0, "finite"),
        (10.0, np.nan, "finite"),
        (0.0, 8.0, "equal_value must be positive"),
        (-1.0, 8.0, "equal_value must be positive"),
    ],
)
def test_relative_improvement_percent_rejects_invalid_inputs(
    equal_value: float,
    optimized_value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        relative_improvement_percent(equal_value=equal_value, optimized_value=optimized_value)


def test_load_yaml_reads_yaml_file(tmp_path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("name: demo\nitems:\n  - 1\n  - 2\n", encoding="utf-8")

    data = load_yaml(yaml_path)

    assert data == {"name": "demo", "items": [1, 2]}


def test_write_csv_creates_parent_directory_and_round_trips_content(tmp_path) -> None:
    df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    csv_path = tmp_path / "nested" / "output.csv"

    returned_path = write_csv(df, csv_path)

    assert returned_path == csv_path
    assert csv_path.exists()
    round_trip = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(round_trip, df)


def test_write_csv_rejects_non_dataframe_input(tmp_path) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        write_csv([{"name": "a"}], tmp_path / "output.csv")
