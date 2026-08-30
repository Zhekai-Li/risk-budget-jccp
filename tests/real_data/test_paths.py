from __future__ import annotations

from risk_budget_jccp.real_data.common.paths import case_results_dir, runtime_results_root


def test_case_results_dir_uses_runs_not_evaluation_split() -> None:
    assert case_results_dir("m5") == runtime_results_root() / "runs" / "m5"


def test_case_results_dir_accepts_explicit_output_root(tmp_path) -> None:
    assert case_results_dir("power", tmp_path) == tmp_path.resolve() / "real_data" / "runs" / "power"
