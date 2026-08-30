from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import norm

import scienceplots  # noqa: F401  # Registers the required paper styles.

plt.style.use(["science", "grid", "no-latex"])


FIG_FONT_SIZE = 12
FIG_WIDTH_IN = 5.3
FIG_HEIGHT_IN = 3.35
WIDE_FIG_WIDTH_IN = 7.8
WIDE_FIG_HEIGHT_IN = 3.45

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "font.size": FIG_FONT_SIZE,
        "axes.labelsize": FIG_FONT_SIZE,
        "axes.titlesize": FIG_FONT_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE,
        "ytick.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE,
        "legend.title_fontsize": FIG_FONT_SIZE,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.6,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "paper"
SUMMARY_DIR = REPO_ROOT / "results" / "real_data" / "summary"
PROCESSED_REAL_DATA_DIR = REPO_ROOT / "data" / "processed" / "real_data"
REAL_DATA_RUNS_DIR = REPO_ROOT / "results" / "real_data" / "runs"
ALPHA_MIN = 1.0e-4
ALPHA_MAX = 1.0e-1

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

CASE_LABELS = {"m5": "M5", "power": "Power", "french": "French"}
CERT_LABELS = {"cvar": "CVaR", "bernstein": "Bernstein", "cantelli": "Cantelli"}
IMPROVEMENT_VIOLATION_LABEL_OFFSETS = (
    (6, 0),
    (6, 8),
    (6, -8),
    (-6, 0),
    (-6, 8),
    (-6, -8),
    (6, 16),
    (6, -16),
    (-6, 16),
    (-6, -16),
    (6, 24),
    (6, -24),
    (-6, 24),
    (-6, -24),
    (16, 8),
    (16, -8),
    (-16, 8),
    (-16, -8),
)
IMPROVEMENT_VIOLATION_LABEL_LINE_COLOR = "0.72"
ALPHA = 0.05
QUALITY_LABELS = {
    "main_positive": "Reliable or near-reliable gain",
    "positive_with_drift": "Gain with drift",
    "stress_negative": "Gain with target exceedance",
    "certificate_mismatch": "Certificate mismatch",
    "over_conservative_certificate": "Over-conservative",
    "fallback_no_improvement": "No improving allocation",
    "solver_or_formulation_issue": "solver/formulation issue",
    "reference_equal": "equal reference",
    "diagnostic_only": "diagnostic",
}
QUALITY_COLORS = {
    "main_positive": OKABE_ITO["bluish_green"],
    "positive_with_drift": OKABE_ITO["orange"],
    "stress_negative": OKABE_ITO["vermillion"],
    "certificate_mismatch": OKABE_ITO["reddish_purple"],
    "over_conservative_certificate": OKABE_ITO["yellow"],
    "fallback_no_improvement": OKABE_ITO["sky_blue"],
    "solver_or_formulation_issue": OKABE_ITO["black"],
    "reference_equal": OKABE_ITO["blue"],
    "diagnostic_only": OKABE_ITO["sky_blue"],
}
STATUS_LABELS = {
    "fallback_equal": "fallback",
    "no_solution": "no solution",
}
STATUS_HATCHES = {
    "success": "",
    "fallback_equal": "////",
    "no_solution": "xxxx",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-only figure PDFs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where PDFs will be written.")
    parser.add_argument("--summary-dir", default=str(SUMMARY_DIR))
    parser.add_argument("--runs-dir", default=str(REAL_DATA_RUNS_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_REAL_DATA_DIR))
    return parser.parse_args(argv)


def bernstein_safety_factor(alpha: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 * np.log(1.0 / alpha))


def cantelli_safety_factor(alpha: np.ndarray) -> np.ndarray:
    return np.sqrt((1.0 - alpha) / alpha)


def gaussian_cvar_safety_factor(alpha: np.ndarray) -> np.ndarray:
    return norm.pdf(norm.ppf(1.0 - alpha)) / alpha


def _read_solution_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _payload_vector(payload: dict[str, object], key: str, length: int, *, default: float = np.nan) -> np.ndarray:
    values = np.asarray(payload.get(key, np.full(length, default)), dtype=float)
    if values.size != length:
        return np.full(length, default, dtype=float)
    return values


def _budget_share(payload: dict[str, object]) -> np.ndarray:
    alpha = np.asarray(payload["alpha_vector"], dtype=float)
    alpha_sum = float(np.sum(alpha))
    if alpha_sum <= 0.0:
        return np.full(alpha.shape, np.nan, dtype=float)
    return alpha / alpha_sum


def _bernstein_kkt_product(payload: dict[str, object], length: int) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(payload["alpha_vector"], dtype=float)
    certificate_values = _payload_vector(payload, "certificate_values", length, default=0.0)
    certificate_dual = _payload_vector(payload, "certificate_dual_values", length)
    theta = _payload_vector(payload, "theta_variables", length, default=0.0)
    alpha_lower_bound = float(payload.get("alpha_lower_bound", np.nan))
    if np.isfinite(alpha_lower_bound):
        is_alpha_interior = alpha > alpha_lower_bound * (1.0 + 1.0e-5)
    else:
        is_alpha_interior = alpha > 0.0
    is_kkt_comparable = (
        np.isfinite(certificate_values)
        & (np.abs(certificate_values) <= 1.0e-4)
        & is_alpha_interior
        & np.isfinite(certificate_dual)
        & (certificate_dual > 1.0e-10)
        & np.isfinite(theta)
    )
    if "theory_budget_driver" in payload:
        theory_budget_driver = _payload_vector(payload, "theory_budget_driver", length)
    else:
        theory_budget_driver = certificate_dual * theta
    kkt_product = np.full(length, np.nan, dtype=float)
    kkt_product[is_kkt_comparable] = theory_budget_driver[is_kkt_comparable]
    return kkt_product, is_kkt_comparable


def _read_m5_budget_overlay_data(
    processed_dir: Path = PROCESSED_REAL_DATA_DIR,
    runs_dir: Path = REAL_DATA_RUNS_DIR,
) -> pd.DataFrame:
    metadata_path = processed_dir / "m5" / "series_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing required input: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    cvar = _read_solution_json(runs_dir / "m5" / "solutions" / "m5_cvar_optimized.json")
    cantelli = _read_solution_json(runs_dir / "m5" / "solutions" / "m5_cantelli_optimized.json")
    frame = pd.DataFrame(
        {
            "economic_demand_risk": metadata["std_demand"].to_numpy(dtype=float)
            * metadata["median_price"].to_numpy(dtype=float),
            "cvar_budget_share": _budget_share(cvar),
            "cantelli_budget_share": _budget_share(cantelli),
        }
    )
    return frame


def _read_power_budget_overlay_data(
    processed_dir: Path = PROCESSED_REAL_DATA_DIR,
    runs_dir: Path = REAL_DATA_RUNS_DIR,
) -> pd.DataFrame:
    metadata_path = processed_dir / "power" / "line_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing required input: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    payload = _read_solution_json(runs_dir / "power" / "solutions" / "power_bernstein_optimized.json")
    length = len(metadata)
    kkt_product, is_kkt_comparable = _bernstein_kkt_product(payload, length)
    heldout_rate = _payload_vector(payload, "heldout_scalar_violation_rates", length, default=0.0)
    frame = pd.DataFrame(
        {
            "budget_share": _budget_share(payload),
            "bernstein_kkt_product": kkt_product,
            "heldout_rate_times_flow_std_times_utilization": heldout_rate
            * metadata["flow_std"].to_numpy(dtype=float)
            * metadata["base_utilization"].to_numpy(dtype=float),
            "is_kkt_comparable": is_kkt_comparable,
        }
    )
    return frame


def _read_french_budget_overlay_data(
    processed_dir: Path = PROCESSED_REAL_DATA_DIR,
    runs_dir: Path = REAL_DATA_RUNS_DIR,
) -> dict[str, pd.DataFrame]:
    metadata_path = processed_dir / "french" / "industry_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing required input: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    volatility = metadata["volatility"].to_numpy(dtype=float)
    frames: dict[str, pd.DataFrame] = {}
    for certificate in ("cvar", "bernstein"):
        payload = _read_solution_json(runs_dir / "french" / "solutions" / f"french_{certificate}_optimized.json")
        weights = np.asarray(payload["weights"], dtype=float)
        frame = pd.DataFrame(
            {
                "weight_volatility": weights * volatility,
                "budget_share": _budget_share(payload),
            }
        )
        if certificate == "bernstein":
            kkt_product, is_kkt_comparable = _bernstein_kkt_product(payload, len(metadata))
            frame["bernstein_kkt_product"] = kkt_product
            frame["is_kkt_comparable"] = is_kkt_comparable
        frames[certificate] = frame
    return frames


def _read_m5_shift_data(processed_dir: Path = PROCESSED_REAL_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    train_path = processed_dir / "m5" / "demand_train.csv"
    test_path = processed_dir / "m5" / "demand_test.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing required input: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"missing required input: {test_path}")
    train = pd.read_csv(train_path).select_dtypes(include=[np.number])
    test = pd.read_csv(test_path).select_dtypes(include=[np.number])
    common_columns = [column for column in train.columns if column in set(test.columns)]
    if not common_columns:
        raise ValueError("M5 demand shift data has no shared numeric series columns")
    return train[common_columns].mean(axis=0).to_numpy(dtype=float), test[common_columns].mean(axis=0).to_numpy(dtype=float)


def _read_power_shift_data(processed_dir: Path = PROCESSED_REAL_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    instance_path = processed_dir / "power" / "power_instance.npz"
    if not instance_path.is_file():
        raise FileNotFoundError(f"missing required input: {instance_path}")
    with np.load(instance_path) as instance:
        train = np.asarray(instance["flow_residual_train"], dtype=float)
        test = np.asarray(instance["flow_residual_test"], dtype=float)
    return train.std(axis=0, ddof=0), test.std(axis=0, ddof=0)


def _read_french_shift_data(processed_dir: Path = PROCESSED_REAL_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = processed_dir / "french" / "industry_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing required input: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {"volatility", "heldout_volatility"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"French shift data is missing columns: {sorted(missing)}")
    return metadata["volatility"].to_numpy(dtype=float), metadata["heldout_volatility"].to_numpy(dtype=float)


def _read_aggregate_objective_frame(summary_dir: Path = SUMMARY_DIR) -> pd.DataFrame:
    path = summary_dir / "real_data_summary_out_of_sample.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")
    frame = pd.read_csv(path)
    frame["evaluation_split"] = "out_of_sample"
    opt = frame.loc[(frame["allocation"] == "optimized") & (frame["result_status"] == "success")].copy()
    opt["case_order"] = opt["case"].map({"m5": 0, "power": 1, "french": 2}).fillna(99)
    opt["cert_order"] = opt["certificate"].map({"cvar": 0, "bernstein": 1, "cantelli": 2}).fillna(99)
    opt["split_order"] = opt["evaluation_split"].map({"in_sample": 0, "out_of_sample": 1}).fillna(99)
    return opt.sort_values(["case_order", "cert_order", "split_order", "case", "certificate"]).reset_index(drop=True)


def _read_out_of_sample_summary(summary_dir: Path = SUMMARY_DIR) -> pd.DataFrame:
    path = summary_dir / "real_data_summary_out_of_sample.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")
    return pd.read_csv(path)


def _normalize_status_for_display(status: object) -> str:
    status_value = str(status)
    if status_value in {"infeasible", "failed"}:
        return "no_solution"
    return status_value


def _read_cross_joint_violation_frames(summary_dir: Path = SUMMARY_DIR) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ("in_sample", "out_of_sample"):
        path = summary_dir / f"real_data_summary_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
        frame = pd.read_csv(path)
        frame["case_order"] = frame["case"].map({"m5": 0, "power": 1, "french": 2}).fillna(99)
        frame["cert_order"] = frame["certificate"].map({"cvar": 0, "bernstein": 1, "cantelli": 2}).fillna(99)
        frame["display_status"] = frame["result_status"].map(_normalize_status_for_display)
        frames[split] = frame.sort_values(["case_order", "cert_order", "allocation"]).reset_index(drop=True)
    return frames


def _quality_category(row: pd.Series) -> str:
    case = str(row.get("case", ""))
    cert = str(row.get("certificate", ""))
    allocation = str(row.get("allocation", ""))
    status = str(row.get("result_status", ""))
    improvement = float(row.get("relative_improvement", 0.0) or 0.0)
    heldout = float(row.get("heldout_joint_violation", row.get("empirical_joint_violation", np.nan)))
    calibration = float(row.get("calibration_joint_violation", np.nan))
    if cert == "cantelli" and status in {"infeasible", "failed", "fallback_equal"}:
        return "over_conservative_certificate"
    if status == "infeasible":
        return "over_conservative_certificate"
    if status == "failed":
        return "solver_or_formulation_issue"
    if case == "m5" and cert == "bernstein":
        return "certificate_mismatch"
    if allocation == "equal":
        return "reference_equal"
    if status == "fallback_equal":
        return "fallback_no_improvement"
    if case == "m5" and np.isfinite(heldout) and heldout > ALPHA and np.isfinite(calibration) and calibration <= ALPHA:
        return "stress_negative"
    if improvement > 1.0e-8:
        return "main_positive" if (not np.isfinite(heldout) or heldout <= 0.08) else "positive_with_drift"
    return "diagnostic_only"


def _read_improvement_vs_violation_frames(summary_dir: Path = SUMMARY_DIR) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ("in_sample", "out_of_sample"):
        path = summary_dir / f"real_data_summary_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
        frame = pd.read_csv(path)
        frame["evaluation_split"] = split
        frame["quality_category"] = frame.apply(_quality_category, axis=1)
        frames[split] = frame.loc[frame["allocation"] == "optimized"].copy()
    return frames


def build_safety_factor_vs_budget() -> tuple[plt.Figure, plt.Axes]:
    alpha = np.geomspace(ALPHA_MIN, ALPHA_MAX, 400)
    bernstein = bernstein_safety_factor(alpha)
    gaussian_cvar = gaussian_cvar_safety_factor(alpha)
    cantelli = cantelli_safety_factor(alpha)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.plot(alpha, bernstein, label="Bernstein", color=OKABE_ITO["blue"])
    ax.plot(alpha, gaussian_cvar, label="Gaussian CVaR", color=OKABE_ITO["orange"])
    ax.plot(alpha, cantelli, label="Cantelli", color=OKABE_ITO["vermillion"])

    ax.set_yscale("log")
    ax.set_xlabel(r"Scalar violation budget $\alpha_i$")
    ax.set_ylabel(r"Safety factor $q(\alpha_i)$")
    ax.set_xlim(0.0, ALPHA_MAX)
    all_factors = np.concatenate([bernstein, gaussian_cvar, cantelli])
    ax.set_ylim(float(all_factors.min()) * 0.9, float(all_factors.max()) * 1.1)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    return fig, ax


def _save_safety_factor_vs_budget(output_path: Path) -> None:
    fig, _ = build_safety_factor_vs_budget()
    fig.savefig(output_path)
    plt.close(fig)


def _save_aggregate_objective_improvement(output_path: Path, summary_dir: Path = SUMMARY_DIR) -> None:
    opt = _read_aggregate_objective_frame(summary_dir)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    if opt.empty:
        ax.text(0.5, 0.5, "No accepted optimized result", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        labels = [
            f"{CASE_LABELS.get(row['case'], row['case'])}\n{CERT_LABELS.get(row['certificate'], row['certificate'])}"
            for _, row in opt.iterrows()
        ]
        x = np.arange(len(opt))
        values = 100.0 * opt["relative_improvement"].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            values,
            0.62,
            color=OKABE_ITO["orange"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
        ax.set_ylabel("OOD objective gain over equal (\\%)")
        ax.set_xticks(x, labels, rotation=0, ha="center")
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            ax.set_ylim(top=max(1.0, float(np.nanmax(finite_values)) * 1.28))
        for bar, value in zip(bars, values, strict=False):
            if np.isfinite(value) and value > 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=FIG_FONT_SIZE,
                )

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.96)
    fig.savefig(output_path)
    plt.close(fig)


def _configuration_label(row: pd.Series) -> str:
    case = str(row["case"])
    certificate = str(row["certificate"])
    return f"{CASE_LABELS.get(case, case)}-{CERT_LABELS.get(certificate, certificate)}"


def _annotation_text_bbox(annotation: plt.Annotation, renderer: object) -> object:
    return Text.get_window_extent(annotation, renderer=renderer)


def _bbox_intersection_area(first: object, second: object) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _label_alignment(dx: float, dy: float) -> tuple[str, str]:
    horizontal = "left" if dx > 0.0 else "right" if dx < 0.0 else "center"
    vertical = "bottom" if dy > 0.0 else "top" if dy < 0.0 else "center"
    return horizontal, vertical


def _annotate_improvement_vs_violation(
    ax: plt.Axes,
    frame: pd.DataFrame,
    violation_column: str,
) -> None:
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    legend = ax.get_legend()
    legend_bbox = legend.get_window_extent(renderer).padded(2.0) if legend is not None else None
    data_points = np.column_stack(
        [
            100.0 * frame["relative_improvement"].to_numpy(dtype=float),
            frame[violation_column].to_numpy(dtype=float),
        ]
    )
    display_points = ax.transData.transform(data_points)
    marker_radius = figure.dpi * 4.5 / 72.0
    marker_bboxes = [
        matplotlib.transforms.Bbox.from_bounds(
            point[0] - marker_radius,
            point[1] - marker_radius,
            2.0 * marker_radius,
            2.0 * marker_radius,
        )
        for point in display_points
    ]
    neighbor_counts = [
        int(np.count_nonzero(np.linalg.norm(display_points - point, axis=1) < 90.0)) - 1
        for point in display_points
    ]
    row_order = sorted(
        range(len(frame)),
        key=lambda index: (-neighbor_counts[index], display_points[index, 1], display_points[index, 0]),
    )
    placed_bboxes: list[object] = []
    for row_index in row_order:
        row = frame.iloc[row_index]
        annotation = ax.annotate(
            _configuration_label(row),
            xy=(100.0 * float(row["relative_improvement"]), float(row[violation_column])),
            xytext=IMPROVEMENT_VIOLATION_LABEL_OFFSETS[0],
            textcoords="offset points",
            fontsize=FIG_FONT_SIZE - 4,
            color=OKABE_ITO["black"],
            arrowprops={
                "arrowstyle": "-",
                "color": IMPROVEMENT_VIOLATION_LABEL_LINE_COLOR,
                "linewidth": 0.45,
                "alpha": 0.9,
                "shrinkA": 1.5,
                "shrinkB": 4.5,
            },
            path_effects=[path_effects.withStroke(linewidth=2.25, foreground=(1.0, 1.0, 1.0, 0.78))],
            annotation_clip=True,
            zorder=4,
        )
        best_choice: tuple[tuple[float, float, float, float], tuple[int, int, str, str], object] | None = None
        for dx, dy in IMPROVEMENT_VIOLATION_LABEL_OFFSETS:
            horizontal_alignment, vertical_alignment = _label_alignment(dx, dy)
            annotation.set_position((dx, dy))
            annotation.set_horizontalalignment(horizontal_alignment)
            annotation.set_verticalalignment(vertical_alignment)
            annotation.update_positions(renderer)
            candidate_bbox = _annotation_text_bbox(annotation, renderer).padded(1.5)
            outside_area = candidate_bbox.width * candidate_bbox.height - _bbox_intersection_area(
                candidate_bbox, axes_bbox
            )
            label_overlap_area = sum(
                _bbox_intersection_area(candidate_bbox, placed_bbox) for placed_bbox in placed_bboxes
            )
            if legend_bbox is not None:
                label_overlap_area += _bbox_intersection_area(candidate_bbox, legend_bbox)
            marker_overlap_area = sum(
                _bbox_intersection_area(candidate_bbox, marker_bbox)
                for marker_index, marker_bbox in enumerate(marker_bboxes)
                if marker_index != row_index
            )
            score = (
                0.0 if outside_area <= 1.0e-6 else 1.0,
                0.0 if label_overlap_area <= 1.0e-6 else 1.0,
                100.0 * outside_area + 10.0 * label_overlap_area + marker_overlap_area,
                float(np.hypot(dx, dy)),
            )
            choice = (score, (dx, dy, horizontal_alignment, vertical_alignment), candidate_bbox)
            if best_choice is None or choice[0] < best_choice[0]:
                best_choice = choice
        assert best_choice is not None
        _, (dx, dy, horizontal_alignment, vertical_alignment), candidate_bbox = best_choice
        annotation.set_position((dx, dy))
        annotation.set_horizontalalignment(horizontal_alignment)
        annotation.set_verticalalignment(vertical_alignment)
        annotation.update_positions(renderer)
        placed_bboxes.append(candidate_bbox)


def _build_aggregate_improvement_vs_violation(
    frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if frames is None:
        frames = _read_improvement_vs_violation_frames()
    specs = [
        ("in_sample", "In-sample", "calibration_joint_violation", "Calibration joint violation"),
        ("out_of_sample", "OOD", "heldout_joint_violation", "Held-out joint violation"),
    ]
    all_x: list[float] = []
    all_y: list[float] = []
    for split, _, column, _ in specs:
        frame = frames[split]
        all_x.extend((100.0 * frame["relative_improvement"]).to_numpy(dtype=float).tolist())
        all_y.extend(frame[column].to_numpy(dtype=float).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(WIDE_FIG_WIDTH_IN, WIDE_FIG_HEIGHT_IN), sharex=True, sharey=True)
    for ax, (split, title, violation_column, ylabel) in zip(axes, specs, strict=True):
        frame = frames[split]
        legend_handles: dict[str, object] = {}
        for quality, group in frame.groupby("quality_category", sort=False):
            x_values = 100.0 * group["relative_improvement"]
            y_values = group[violation_column]
            scatter = ax.scatter(
                x_values,
                y_values,
                s=58,
                color=QUALITY_COLORS.get(str(quality), OKABE_ITO["sky_blue"]),
                edgecolor="black",
                linewidth=0.45,
                label=QUALITY_LABELS.get(str(quality), str(quality)),
                zorder=3,
            )
            legend_handles.setdefault(str(quality), scatter)
        ax.axhline(ALPHA, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.2, zorder=2)
        ax.axvline(0.0, color=OKABE_ITO["black"], linewidth=0.8, zorder=2)
        ax.set_title(title)
        ax.set_xlabel("Objective gain over equal (%)")
        ax.set_ylabel(ylabel)
        present = [quality for quality in QUALITY_LABELS if quality in legend_handles]
        if present:
            ax.legend(
                [legend_handles[quality] for quality in present],
                [QUALITY_LABELS[quality] for quality in present],
                frameon=False,
                loc="upper right",
                fontsize=FIG_FONT_SIZE - 2,
                labelspacing=0.2,
                handletextpad=0.35,
                borderaxespad=0.25,
            )

    if all_x:
        axes[0].set_xlim(min(all_x) - 1.2, max(all_x) + 1.6)
    if all_y:
        axes[0].set_ylim(min(-0.095, min(all_y) - 0.06), max(all_y) + 0.06)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.90, wspace=0.22)
    for ax, (split, _, violation_column, _) in zip(axes, specs, strict=True):
        _annotate_improvement_vs_violation(ax, frames[split], violation_column)
    return fig, axes


def _save_aggregate_improvement_vs_violation(output_path: Path, summary_dir: Path = SUMMARY_DIR) -> None:
    fig, _ = _build_aggregate_improvement_vs_violation(_read_improvement_vs_violation_frames(summary_dir))
    fig.savefig(output_path)
    plt.close(fig)


def _save_cross_joint_violation(output_path: Path, summary_dir: Path = SUMMARY_DIR) -> None:
    frames = _read_cross_joint_violation_frames(summary_dir)
    specs = [
        ("in_sample", "In-sample", "calibration_joint_violation", "Calibration JVP"),
        ("out_of_sample", "OOD", "heldout_joint_violation", "Held-out JVP"),
    ]
    indexed_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_values: list[float] = []
    base_index: pd.MultiIndex | None = None
    for split, _, value_column, _ in specs:
        ordered = frames[split]
        grouped = ordered.pivot_table(
            index=["case", "certificate"],
            columns="allocation",
            values=value_column,
            aggfunc="first",
        ).reindex(columns=["equal", "optimized"])
        statuses = ordered.pivot_table(
            index=["case", "certificate"],
            columns="allocation",
            values="display_status",
            aggfunc="first",
        ).reindex(columns=["equal", "optimized"])
        index_order = pd.MultiIndex.from_frame(ordered[["case", "certificate"]].drop_duplicates())
        grouped = grouped.reindex(index_order)
        statuses = statuses.reindex(index_order)
        if base_index is None:
            base_index = grouped.index
        else:
            grouped = grouped.reindex(base_index)
            statuses = statuses.reindex(base_index)
        indexed_frames[split] = (grouped, statuses)
        all_values.extend(grouped.to_numpy(dtype=float).ravel().tolist())
    if base_index is None:
        raise ValueError("no cross joint violation data to plot")

    labels = [f"{CASE_LABELS.get(case, case)}\n{CERT_LABELS.get(cert, cert)}" for case, cert in base_index]
    x = np.arange(len(base_index)) * 1.38
    width = 0.46
    offset = 0.29

    fig, axes = plt.subplots(2, 1, figsize=(WIDE_FIG_WIDTH_IN, 5.15), sharex=True, sharey=True)
    finite_values = np.asarray([value for value in all_values if np.isfinite(value)], dtype=float)
    y_top = max(ALPHA * 1.45, float(np.nanmax(finite_values)) * 1.20 + 1.0e-6) if finite_values.size else ALPHA * 1.45

    for ax, (split, title, _, ylabel) in zip(axes, specs, strict=True):
        grouped, statuses = indexed_frames[split]
        equal_values = grouped["equal"].to_numpy(dtype=float)
        optimized_values = grouped["optimized"].to_numpy(dtype=float)
        bars_equal = ax.bar(
            x - offset,
            equal_values,
            width,
            color=OKABE_ITO["blue"],
            edgecolor="black",
            linewidth=0.5,
            label="equal allocation",
        )
        bars_optimized = ax.bar(
            x + offset,
            optimized_values,
            width,
            color=OKABE_ITO["orange"],
            edgecolor="black",
            linewidth=0.5,
            label="optimized allocation",
        )
        for bars, allocation in ((bars_equal, "equal"), (bars_optimized, "optimized")):
            for bar, status in zip(bars, statuses[allocation].astype(str), strict=False):
                bar.set_hatch(STATUS_HATCHES.get(status, ""))

        ax.axhline(
            ALPHA,
            color=OKABE_ITO["vermillion"],
            linestyle="--",
            linewidth=1.3,
            label=r"target $\alpha=0.05$",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, y_top)
        for bars in (bars_equal, bars_optimized):
            for bar in bars:
                value = bar.get_height()
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.3f}",
                        (bar.get_x() + bar.get_width() / 2.0, value),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=FIG_FONT_SIZE - 4,
                        clip_on=False,
                    )

        present_statuses = [
            status
            for status in ("fallback_equal", "no_solution")
            if status in set(statuses.to_numpy(dtype=object).ravel())
        ]
        handles, labels_ = ax.get_legend_handles_labels()
        status_handles = [
            Patch(facecolor="white", edgecolor="black", hatch=STATUS_HATCHES[status], label=STATUS_LABELS[status])
            for status in present_statuses
        ]
        legend = ax.legend(
            [*handles, *status_handles],
            [*labels_, *[handle.get_label() for handle in status_handles]],
            frameon=True,
            loc="upper right",
            ncols=2,
            fontsize=FIG_FONT_SIZE - 2,
            columnspacing=0.9,
            labelspacing=0.18,
            handletextpad=0.38,
            borderaxespad=0.32,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.90)
        legend.get_frame().set_linewidth(0.0)

    axes[-1].set_xticks(x, labels, rotation=0, ha="center")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.95, hspace=0.18)
    fig.savefig(output_path)
    plt.close(fig)


def _finite_positive_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (y > 0.0)


def _style_overlay_legend(legend: object) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.90)
    frame.set_linewidth(0.0)


def _positive_log_limits(values: np.ndarray, *, pad: float = 1.7) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1.0e-6, 1.0
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if np.isclose(low, high):
        return low / pad, high * pad
    return low / pad, high * pad


def _set_y_axis_color(ax: plt.Axes, color: str, *, side: str) -> None:
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="y", colors=color)
    ax.spines[side].set_color(color)


def _set_x_axis_color(ax: plt.Axes, color: str, *, side: str) -> None:
    ax.xaxis.label.set_color(color)
    ax.tick_params(axis="x", colors=color)
    ax.spines[side].set_color(color)


def _set_x_axis_spacing(ax: plt.Axes, *, tick_pad: float = 2.0, label_pad: float = 3.0) -> None:
    ax.tick_params(axis="x", pad=tick_pad)
    ax.xaxis.labelpad = label_pad


def _panel_label_below(ax: plt.Axes, label: str, *, y: float = -0.30) -> None:
    ax.text(0.5, y, label, transform=ax.transAxes, ha="center", va="top", fontsize=FIG_FONT_SIZE)


def _plot_shift_panel(
    ax: plt.Axes,
    calibration: np.ndarray,
    heldout: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    panel_label: str,
) -> None:
    x = np.asarray(calibration, dtype=float)
    y = np.asarray(heldout, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    ax.scatter(
        x[finite],
        y[finite],
        s=34,
        color=OKABE_ITO["blue"],
        alpha=0.76,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )
    if finite.any():
        lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
        hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
        span = max(hi - lo, 1.0e-12)
        lower = min(0.0, lo - 0.04 * span)
        upper = hi + 0.06 * span
        ax.plot([lower, upper], [lower, upper], color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.1, zorder=2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE - 1)
    _panel_label_below(ax, panel_label, y=-0.33)


def _save_calibration_heldout_shift(output_path: Path, processed_dir: Path = PROCESSED_REAL_DATA_DIR) -> None:
    m5_calibration, m5_heldout = _read_m5_shift_data(processed_dir)
    power_calibration, power_heldout = _read_power_shift_data(processed_dir)
    french_calibration, french_heldout = _read_french_shift_data(processed_dir)

    fig, axes = plt.subplots(1, 3, figsize=(WIDE_FIG_WIDTH_IN, 3.15))
    _plot_shift_panel(
        axes[0],
        m5_calibration,
        m5_heldout,
        xlabel="Calibration mean demand",
        ylabel="Held-out mean demand",
        panel_label="(a) M5",
    )
    _plot_shift_panel(
        axes[1],
        power_calibration,
        power_heldout,
        xlabel="Calibration residual-flow volatility",
        ylabel="Held-out residual-flow volatility",
        panel_label="(b) Power",
    )
    _plot_shift_panel(
        axes[2],
        french_calibration,
        french_heldout,
        xlabel="Calibration return volatility",
        ylabel="Held-out return volatility",
        panel_label="(c) French",
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.26, top=0.96, wspace=0.48)
    fig.savefig(output_path)
    plt.close(fig)


def _save_budget_driver_overlay(
    output_path: Path,
    processed_dir: Path = PROCESSED_REAL_DATA_DIR,
    runs_dir: Path = REAL_DATA_RUNS_DIR,
) -> None:
    m5 = _read_m5_budget_overlay_data(processed_dir, runs_dir)
    power = _read_power_budget_overlay_data(processed_dir, runs_dir)
    french = _read_french_budget_overlay_data(processed_dir, runs_dir)

    fig, axes = plt.subplots(3, 1, figsize=(WIDE_FIG_WIDTH_IN, 7.6))

    # M5: same primitive demand-risk x-axis, certificate-specific optimized shares.
    ax = axes[0]
    ax_right = ax.twinx()
    x_m5 = m5["economic_demand_risk"].to_numpy(dtype=float)
    y_cvar = m5["cvar_budget_share"].to_numpy(dtype=float)
    y_cantelli = m5["cantelli_budget_share"].to_numpy(dtype=float)
    mask_cvar = _finite_positive_xy(x_m5, y_cvar)
    mask_cantelli = _finite_positive_xy(x_m5, y_cantelli)
    cvar_scatter = ax.scatter(
        x_m5[mask_cvar],
        y_cvar[mask_cvar],
        s=30,
        color=OKABE_ITO["blue"],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.35,
        label="CVaR budget",
    )
    cantelli_scatter = ax_right.scatter(
        x_m5[mask_cantelli],
        y_cantelli[mask_cantelli],
        s=30,
        color=OKABE_ITO["vermillion"],
        alpha=0.72,
        marker="D",
        edgecolor="black",
        linewidth=0.35,
        label="Cantelli budget",
    )
    ax.set_xlabel(r"Demand std $\times$ median price")
    _set_x_axis_spacing(ax)
    ax.set_ylabel("CVaR budget share", color=OKABE_ITO["blue"])
    ax_right.set_ylabel("Cantelli budget share", color=OKABE_ITO["vermillion"])
    ax.set_yscale("log")
    ax_right.set_yscale("log")
    ax.set_ylim(*_positive_log_limits(y_cvar[mask_cvar]))
    ax_right.set_ylim(*_positive_log_limits(y_cantelli[mask_cantelli]))
    _set_y_axis_color(ax, OKABE_ITO["blue"], side="left")
    _set_y_axis_color(ax_right, OKABE_ITO["vermillion"], side="right")
    ax_right.grid(False)
    legend = ax.legend(
        [cvar_scatter, cantelli_scatter],
        ["CVaR budget", "Cantelli budget"],
        frameon=True,
        loc="upper left",
        fontsize=FIG_FONT_SIZE - 2,
        labelspacing=0.18,
        handletextpad=0.35,
    )
    _style_overlay_legend(legend)
    _panel_label_below(ax, "(a) M5", y=-0.34)

    # Power: one optimized Bernstein budget share with two horizontal diagnostics.
    ax = axes[1]
    ax_top = ax.twiny()
    y_power = power["budget_share"].to_numpy(dtype=float)
    x_kkt = power["bernstein_kkt_product"].to_numpy(dtype=float)
    x_audit = power["heldout_rate_times_flow_std_times_utilization"].to_numpy(dtype=float)
    comparable = power["is_kkt_comparable"].to_numpy(dtype=bool)
    mask_kkt = _finite_positive_xy(x_kkt, y_power) & comparable
    mask_audit = _finite_positive_xy(x_audit, y_power)
    audit_scatter = ax.scatter(
        x_audit[mask_audit],
        y_power[mask_audit],
        s=32,
        color=OKABE_ITO["orange"],
        marker="s",
        alpha=0.72,
        edgecolor="black",
        linewidth=0.35,
        label="Held-out audit proxy",
    )
    kkt_scatter = ax_top.scatter(
        x_kkt[mask_kkt],
        y_power[mask_kkt],
        s=56,
        color=OKABE_ITO["bluish_green"],
        alpha=0.82,
        edgecolor="black",
        linewidth=0.35,
        label="Bernstein KKT",
    )
    ax.set_xlabel(r"Held-out violation rate $\times$ residual std $\times$ utilization")
    ax_top.set_xlabel(r"Bernstein KKT product $\mu_i^\star\theta_i^\star$")
    _set_x_axis_spacing(ax)
    _set_x_axis_spacing(ax_top)
    ax.set_ylabel("Bernstein budget share")
    ax.set_yscale("log")
    ax.set_ylim(*_positive_log_limits(y_power))
    _set_x_axis_color(ax, OKABE_ITO["orange"], side="bottom")
    _set_x_axis_color(ax_top, OKABE_ITO["bluish_green"], side="top")
    ax_top.grid(False)
    legend = ax.legend(
        [kkt_scatter, audit_scatter],
        ["Bernstein KKT", "Held-out audit proxy"],
        frameon=True,
        loc="center right",
        fontsize=FIG_FONT_SIZE - 2,
        labelspacing=0.18,
        handletextpad=0.35,
    )
    _style_overlay_legend(legend)
    _panel_label_below(ax, "(b) Power", y=-0.34)

    # French: common exposure x-axis for budgets, twin y-axis for certificate-specific shares.
    ax = axes[2]
    ax_right = ax.twinx()
    ax_top = ax_right.twiny()
    cvar = french["cvar"]
    bernstein = french["bernstein"]
    x_cvar_exposure = cvar["weight_volatility"].to_numpy(dtype=float)
    y_cvar_budget = cvar["budget_share"].to_numpy(dtype=float)
    x_bern_exposure = bernstein["weight_volatility"].to_numpy(dtype=float)
    y_bern_budget = bernstein["budget_share"].to_numpy(dtype=float)
    x_bern_kkt = bernstein["bernstein_kkt_product"].to_numpy(dtype=float)
    bern_comparable = bernstein["is_kkt_comparable"].to_numpy(dtype=bool)
    mask_cvar = _finite_positive_xy(x_cvar_exposure, y_cvar_budget)
    mask_bern = _finite_positive_xy(x_bern_exposure, y_bern_budget)
    mask_bern_kkt = _finite_positive_xy(x_bern_kkt, y_bern_budget) & bern_comparable
    french_cvar_scatter = ax.scatter(
        x_cvar_exposure[mask_cvar],
        y_cvar_budget[mask_cvar],
        s=34,
        color=OKABE_ITO["blue"],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.35,
        label="CVaR budget",
    )
    french_bern_scatter = ax_right.scatter(
        x_bern_exposure[mask_bern],
        y_bern_budget[mask_bern],
        s=34,
        color=OKABE_ITO["orange"],
        marker="s",
        alpha=0.74,
        edgecolor="black",
        linewidth=0.35,
        label="Bernstein budget",
    )
    french_kkt_scatter = ax_top.scatter(
        x_bern_kkt[mask_bern_kkt],
        y_bern_budget[mask_bern_kkt],
        s=58,
        color=OKABE_ITO["bluish_green"],
        marker="^",
        alpha=0.84,
        edgecolor="black",
        linewidth=0.35,
        label="Bernstein KKT",
    )
    ax.set_xlabel(r"Weight $\times$ volatility")
    ax.set_ylabel("CVaR budget share")
    ax_right.set_ylabel("Bernstein budget share")
    ax_top.set_xlabel(r"Bernstein KKT product $\mu_i^\star\theta_i^\star$")
    _set_x_axis_spacing(ax)
    _set_x_axis_spacing(ax_top)
    ax.set_yscale("log")
    ax_right.set_yscale("log")
    ax.set_ylim(*_positive_log_limits(y_cvar_budget[mask_cvar]))
    ax_right.set_ylim(*_positive_log_limits(y_bern_budget[mask_bern]))
    _set_y_axis_color(ax, OKABE_ITO["blue"], side="left")
    _set_y_axis_color(ax_right, OKABE_ITO["orange"], side="right")
    _set_x_axis_color(ax_top, OKABE_ITO["bluish_green"], side="top")
    ax_top.grid(False)
    legend = ax.legend(
        [french_cvar_scatter, french_bern_scatter, french_kkt_scatter],
        ["CVaR budget", "Bernstein budget", "Bernstein KKT"],
        frameon=True,
        loc="upper left",
        fontsize=FIG_FONT_SIZE - 2,
        labelspacing=0.18,
        handletextpad=0.35,
    )
    _style_overlay_legend(legend)
    _panel_label_below(ax, "(c) French", y=-0.34)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.09, top=0.94, hspace=0.92)
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)
    runs_dir = Path(args.runs_dir)
    processed_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_safety_factor_vs_budget(output_dir / "safety_factor_vs_budget.pdf")
    _save_aggregate_objective_improvement(output_dir / "aggregate_objective_improvement.pdf", summary_dir)
    _save_aggregate_improvement_vs_violation(output_dir / "aggregate_improvement_vs_violation.pdf", summary_dir)
    _save_cross_joint_violation(output_dir / "cross_joint_violation.pdf", summary_dir)
    _save_budget_driver_overlay(output_dir / "budget_driver_overlay.pdf", processed_dir, runs_dir)
    _save_calibration_heldout_shift(output_dir / "calibration_heldout_shift.pdf", processed_dir)


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
