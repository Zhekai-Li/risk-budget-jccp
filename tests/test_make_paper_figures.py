import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "make_paper_figures.py"
from risk_budget_jccp.reporting import paper_figures as MAKE_PAPER_FIGURES

def _write_paper_input_fixture(root: Path) -> tuple[Path, Path]:
    processed_dir = root / "processed"
    runs_dir = root / "runs"
    for case in ("m5", "power", "french"):
        (processed_dir / case).mkdir(parents=True)
        (runs_dir / case / "solutions").mkdir(parents=True)

    pd.DataFrame(
        {"std_demand": [1.0, 2.0, 3.0], "median_price": [2.0, 1.5, 1.0]}
    ).to_csv(processed_dir / "m5" / "series_metadata.csv", index=False)
    pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0], "c": [3.0, 6.0]}).to_csv(
        processed_dir / "m5" / "demand_train.csv", index=False
    )
    pd.DataFrame({"a": [1.5, 2.5], "b": [3.0, 5.0], "c": [4.0, 7.0]}).to_csv(
        processed_dir / "m5" / "demand_test.csv", index=False
    )

    pd.DataFrame(
        {"flow_std": [1.0, 1.5, 2.0], "base_utilization": [0.5, 0.7, 0.9]}
    ).to_csv(processed_dir / "power" / "line_metadata.csv", index=False)
    np.savez(
        processed_dir / "power" / "power_instance.npz",
        flow_residual_train=np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0], [2.0, 4.0, 5.0]]),
        flow_residual_test=np.array([[0.0, 1.5, 3.0], [1.5, 3.0, 5.0], [3.0, 5.0, 7.0]]),
    )

    pd.DataFrame(
        {"volatility": [0.01, 0.02, 0.03], "heldout_volatility": [0.015, 0.025, 0.04]}
    ).to_csv(processed_dir / "french" / "industry_metadata.csv", index=False)

    def write_solution(case: str, name: str, payload: dict[str, object]) -> None:
        path = runs_dir / case / "solutions" / name
        path.write_text(json.dumps(payload), encoding="utf-8")

    write_solution("m5", "m5_cvar_optimized.json", {"alpha_vector": [0.01, 0.015, 0.025]})
    write_solution("m5", "m5_cantelli_optimized.json", {"alpha_vector": [0.02, 0.015, 0.015]})
    bernstein_payload = {
        "alpha_vector": [0.01, 0.015, 0.025],
        "certificate_values": [0.0, 0.0, 0.0],
        "certificate_dual_values": [1.0, 1.5, 2.0],
        "theta_variables": [1.0, 1.0, 1.0],
        "alpha_lower_bound": 1.0e-8,
        "heldout_scalar_violation_rates": [0.01, 0.02, 0.03],
    }
    write_solution("power", "power_bernstein_optimized.json", bernstein_payload)
    write_solution(
        "french",
        "french_cvar_optimized.json",
        {"alpha_vector": [0.01, 0.015, 0.025], "weights": [0.2, 0.3, 0.5]},
    )
    write_solution(
        "french",
        "french_bernstein_optimized.json",
        {**bernstein_payload, "weights": [0.25, 0.35, 0.4]},
    )
    return processed_dir, runs_dir


def test_make_paper_figures_writes_expected_pdfs(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper"
    processed_dir, runs_dir = _write_paper_input_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-dir",
            str(output_dir),
            "--summary-dir",
            str(REPO_ROOT / "results" / "real_data" / "summary"),
            "--processed-dir",
            str(processed_dir),
            "--runs-dir",
            str(runs_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "aggregate_improvement_vs_violation.pdf",
        "aggregate_objective_improvement.pdf",
        "budget_driver_overlay.pdf",
        "calibration_heldout_shift.pdf",
        "cross_joint_violation.pdf",
        "safety_factor_vs_budget.pdf",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert all(path.stat().st_size > 0 for path in output_dir.iterdir())


def test_safety_factor_builder_has_required_semantics() -> None:
    fig, ax = MAKE_PAPER_FIGURES.build_safety_factor_vs_budget()
    try:
        assert ax.get_yscale() == "log"
        assert [line.get_label() for line in ax.lines] == ["Bernstein", "Gaussian CVaR", "Cantelli"]
        assert [line.get_color() for line in ax.lines] == [
            MAKE_PAPER_FIGURES.OKABE_ITO["blue"],
            MAKE_PAPER_FIGURES.OKABE_ITO["orange"],
            MAKE_PAPER_FIGURES.OKABE_ITO["vermillion"],
        ]
        alpha = ax.lines[0].get_xdata()
        assert np.allclose(ax.lines[0].get_ydata(), MAKE_PAPER_FIGURES.bernstein_safety_factor(alpha))
        assert np.allclose(ax.lines[1].get_ydata(), MAKE_PAPER_FIGURES.gaussian_cvar_safety_factor(alpha))
        assert np.allclose(ax.lines[2].get_ydata(), MAKE_PAPER_FIGURES.cantelli_safety_factor(alpha))
    finally:
        MAKE_PAPER_FIGURES.plt.close(fig)


def test_aggregate_objective_frame_uses_ood_optimized_success_rows(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    rows = [
        {
            "case": "m5",
            "certificate": "cvar",
            "allocation": "equal",
            "result_status": "success",
            "relative_improvement": 0.0,
        },
        {
            "case": "m5",
            "certificate": "cvar",
            "allocation": "optimized",
            "result_status": "success",
            "relative_improvement": 0.05,
        },
        {
            "case": "power",
            "certificate": "bernstein",
            "allocation": "optimized",
            "result_status": "failed",
            "relative_improvement": 0.10,
        },
    ]
    pd.DataFrame(
        [
            {
                "case": "french",
                "certificate": "cvar",
                "allocation": "optimized",
                "result_status": "success",
                "relative_improvement": 0.99,
            }
        ]
    ).to_csv(summary_dir / "real_data_summary_in_sample.csv", index=False)
    pd.DataFrame(rows).to_csv(summary_dir / "real_data_summary_out_of_sample.csv", index=False)

    frame = MAKE_PAPER_FIGURES._read_aggregate_objective_frame(summary_dir)

    assert set(frame["evaluation_split"]) == {"out_of_sample"}
    assert set(frame["allocation"]) == {"optimized"}
    assert set(frame["result_status"]) == {"success"}
    assert len(frame) == 1
    assert np.allclose(frame["relative_improvement"], 0.05)


def test_improvement_vs_violation_frames_use_both_splits_and_optimized_rows(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    base_rows = [
        {
            "case": "m5",
            "certificate": "cvar",
            "allocation": "equal",
            "result_status": "success",
            "relative_improvement": 0.0,
            "calibration_joint_violation": 0.01,
            "heldout_joint_violation": 0.20,
        },
        {
            "case": "m5",
            "certificate": "cvar",
            "allocation": "optimized",
            "result_status": "success",
            "relative_improvement": 0.05,
            "calibration_joint_violation": 0.02,
            "heldout_joint_violation": 0.30,
        },
    ]
    pd.DataFrame(base_rows).to_csv(summary_dir / "real_data_summary_in_sample.csv", index=False)
    pd.DataFrame(base_rows).to_csv(summary_dir / "real_data_summary_out_of_sample.csv", index=False)

    frames = MAKE_PAPER_FIGURES._read_improvement_vs_violation_frames(summary_dir)

    assert set(frames) == {"in_sample", "out_of_sample"}
    assert set(frames["in_sample"]["allocation"]) == {"optimized"}
    assert set(frames["out_of_sample"]["allocation"]) == {"optimized"}
    assert np.allclose(frames["in_sample"]["calibration_joint_violation"], 0.02)
    assert np.allclose(frames["out_of_sample"]["heldout_joint_violation"], 0.30)


def test_improvement_vs_violation_figure_labels_every_configuration() -> None:
    frames = MAKE_PAPER_FIGURES._read_improvement_vs_violation_frames()

    fig, axes = MAKE_PAPER_FIGURES._build_aggregate_improvement_vs_violation(frames)

    expected_labels = {
        "M5-CVaR",
        "M5-Bernstein",
        "M5-Cantelli",
        "Power-CVaR",
        "Power-Bernstein",
        "Power-Cantelli",
        "French-CVaR",
        "French-Bernstein",
        "French-Cantelli",
    }
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            assert ax.get_xlabel() == "Objective gain over equal (%)"
            assert {annotation.get_text() for annotation in ax.texts} == expected_labels
            label_bboxes = []
            for annotation in ax.texts:
                assert annotation.get_bbox_patch() is None
                assert annotation.get_path_effects()
                assert annotation.arrow_patch is not None
                assert np.isclose(annotation.arrow_patch.get_linewidth(), 0.45)
                expected_line_color = MAKE_PAPER_FIGURES.matplotlib.colors.to_rgba(
                    MAKE_PAPER_FIGURES.IMPROVEMENT_VIOLATION_LABEL_LINE_COLOR,
                    alpha=0.9,
                )
                assert np.allclose(annotation.arrow_patch.get_edgecolor(), expected_line_color)
                label_bbox = MAKE_PAPER_FIGURES._annotation_text_bbox(annotation, renderer)
                label_bboxes.append(label_bbox)
                point_x, point_y = ax.transData.transform(annotation.xy)
                distance_x = max(label_bbox.x0 - point_x, 0.0, point_x - label_bbox.x1)
                distance_y = max(label_bbox.y0 - point_y, 0.0, point_y - label_bbox.y1)
                assert np.hypot(distance_x, distance_y) <= 55.0
            for label_index, label_bbox in enumerate(label_bboxes):
                assert all(
                    not label_bbox.overlaps(other_bbox)
                    for other_bbox in label_bboxes[label_index + 1 :]
                )
    finally:
        MAKE_PAPER_FIGURES.plt.close(fig)


def test_out_of_sample_summary_reader_requires_ood_file(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    expected = pd.DataFrame(
        [
            {
                "case": "m5",
                "certificate": "cvar",
                "allocation": "equal",
                "heldout_joint_violation": 0.20,
            }
        ]
    )
    expected.to_csv(summary_dir / "real_data_summary_out_of_sample.csv", index=False)

    frame = MAKE_PAPER_FIGURES._read_out_of_sample_summary(summary_dir)

    assert frame.to_dict(orient="records") == expected.to_dict(orient="records")


def test_cross_joint_violation_frames_use_both_splits_and_display_statuses(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    rows = [
        {
            "case": "power",
            "certificate": "cantelli",
            "allocation": "equal",
            "result_status": "infeasible",
            "calibration_joint_violation": 0.13,
            "heldout_joint_violation": 0.07,
        },
        {
            "case": "power",
            "certificate": "cantelli",
            "allocation": "optimized",
            "result_status": "failed",
            "calibration_joint_violation": 0.13,
            "heldout_joint_violation": 0.07,
        },
        {
            "case": "m5",
            "certificate": "bernstein",
            "allocation": "optimized",
            "result_status": "fallback_equal",
            "calibration_joint_violation": 0.72,
            "heldout_joint_violation": 0.88,
        },
    ]
    pd.DataFrame(rows).to_csv(summary_dir / "real_data_summary_in_sample.csv", index=False)
    pd.DataFrame(rows).to_csv(summary_dir / "real_data_summary_out_of_sample.csv", index=False)

    frames = MAKE_PAPER_FIGURES._read_cross_joint_violation_frames(summary_dir)

    assert set(frames) == {"in_sample", "out_of_sample"}
    assert set(frames["in_sample"]["display_status"]) == {"fallback_equal", "no_solution"}
    assert set(frames["out_of_sample"]["display_status"]) == {"fallback_equal", "no_solution"}
    assert frames["in_sample"].iloc[0]["case"] == "m5"


def test_m5_budget_overlay_uses_shared_primitive_x_axis(tmp_path: Path) -> None:
    processed_dir, runs_dir = _write_paper_input_fixture(tmp_path)
    frame = MAKE_PAPER_FIGURES._read_m5_budget_overlay_data(processed_dir, runs_dir)

    assert {"economic_demand_risk", "cvar_budget_share", "cantelli_budget_share"}.issubset(frame.columns)
    assert np.all(np.isfinite(frame["economic_demand_risk"]))
    assert np.isclose(frame["cvar_budget_share"].sum(), 1.0)
    assert np.isclose(frame["cantelli_budget_share"].sum(), 1.0)
    assert frame["cvar_budget_share"].max() > frame["cantelli_budget_share"].max()


def test_power_budget_overlay_contains_kkt_and_heldout_audit_axes(tmp_path: Path) -> None:
    processed_dir, runs_dir = _write_paper_input_fixture(tmp_path)
    frame = MAKE_PAPER_FIGURES._read_power_budget_overlay_data(processed_dir, runs_dir)

    assert {"budget_share", "bernstein_kkt_product", "heldout_rate_times_flow_std_times_utilization"}.issubset(
        frame.columns
    )
    assert np.isclose(frame["budget_share"].sum(), 1.0)
    assert frame["is_kkt_comparable"].any()
    kkt_values = frame.loc[frame["is_kkt_comparable"], "bernstein_kkt_product"]
    assert np.all(np.isfinite(kkt_values))
    assert np.all(kkt_values > 0.0)
    assert np.isfinite(frame["heldout_rate_times_flow_std_times_utilization"]).all()


def test_french_budget_overlay_uses_weight_volatility_for_budget_panels(tmp_path: Path) -> None:
    processed_dir, runs_dir = _write_paper_input_fixture(tmp_path)
    frames = MAKE_PAPER_FIGURES._read_french_budget_overlay_data(processed_dir, runs_dir)

    assert set(frames) == {"cvar", "bernstein"}
    for frame in frames.values():
        assert {"weight_volatility", "budget_share"}.issubset(frame.columns)
        assert np.isclose(frame["budget_share"].sum(), 1.0)
        assert np.isfinite(frame["weight_volatility"]).all()
    bernstein = frames["bernstein"]
    assert bernstein["is_kkt_comparable"].any()
    kkt_values = bernstein.loc[bernstein["is_kkt_comparable"], "bernstein_kkt_product"]
    assert np.all(np.isfinite(kkt_values))
    assert np.all(kkt_values > 0.0)


def test_calibration_heldout_shift_readers_return_matching_finite_vectors(tmp_path: Path) -> None:
    processed_dir, _ = _write_paper_input_fixture(tmp_path)
    for reader in (
        MAKE_PAPER_FIGURES._read_m5_shift_data,
        MAKE_PAPER_FIGURES._read_power_shift_data,
        MAKE_PAPER_FIGURES._read_french_shift_data,
    ):
        calibration, heldout = reader(processed_dir)
        assert calibration.shape == heldout.shape
        assert calibration.ndim == 1
        assert calibration.size > 0
        assert np.all(np.isfinite(calibration))
        assert np.all(np.isfinite(heldout))


def test_power_shift_data_matches_scalar_constraint_count(tmp_path: Path) -> None:
    processed_dir, _ = _write_paper_input_fixture(tmp_path)
    calibration, heldout = MAKE_PAPER_FIGURES._read_power_shift_data(processed_dir)
    power_metadata = pd.read_csv(processed_dir / "power" / "line_metadata.csv")

    assert len(calibration) == len(power_metadata)
    assert len(heldout) == len(power_metadata)


def test_safety_factor_ordering_matches_paper_statement() -> None:
    alpha = np.geomspace(MAKE_PAPER_FIGURES.ALPHA_MIN, 1.0e-1, 100)

    bernstein = MAKE_PAPER_FIGURES.bernstein_safety_factor(alpha)
    cvar = MAKE_PAPER_FIGURES.gaussian_cvar_safety_factor(alpha)
    cantelli = MAKE_PAPER_FIGURES.cantelli_safety_factor(alpha)

    assert np.all(cantelli > bernstein)
    assert np.all(np.isfinite(cvar))
    assert np.all(cvar > 0.0)


def test_safety_factor_values_are_log_safe_over_display_range() -> None:
    alpha = np.geomspace(
        MAKE_PAPER_FIGURES.ALPHA_MIN,
        MAKE_PAPER_FIGURES.ALPHA_MAX,
        100,
    )

    factors = [
        MAKE_PAPER_FIGURES.bernstein_safety_factor(alpha),
            MAKE_PAPER_FIGURES.gaussian_cvar_safety_factor(alpha),
        MAKE_PAPER_FIGURES.cantelli_safety_factor(alpha),
    ]

    assert MAKE_PAPER_FIGURES.ALPHA_MIN > 0.0
    assert MAKE_PAPER_FIGURES.ALPHA_MAX == 1.0e-1
    for values in factors:
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
