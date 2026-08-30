from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:  # SciencePlots registers Matplotlib styles on import.
    import scienceplots  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciencePlots is required for report figures. Install the scienceplots package and rerun the report build.") from exc

try:
    plt.style.use(["science", "grid", "no-latex"])
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Could not apply the SciencePlots style ['science', 'grid', 'no-latex'].") from exc

FIG_FONT_SIZE = 12

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": FIG_FONT_SIZE,
        "axes.labelsize": FIG_FONT_SIZE,
        "axes.titlesize": FIG_FONT_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE,
        "ytick.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE,
        "legend.title_fontsize": FIG_FONT_SIZE,
        "lines.linewidth": 1.6,
        "patch.linewidth": 0.6,
    }
)


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

COLORS = {
    "equal": OKABE_ITO["blue"],
    "optimized": OKABE_ITO["orange"],
    "accent": OKABE_ITO["bluish_green"],
    "danger": OKABE_ITO["vermillion"],
    "neutral": OKABE_ITO["black"],
}


def _joint_column(summary: pd.DataFrame) -> str:
    if "heldout_joint_violation" in summary.columns:
        return "heldout_joint_violation"
    if "heldout_emergency_joint_violation" in summary.columns:
        return "heldout_emergency_joint_violation"
    return "empirical_joint_violation"


def _annotate_bars(ax: plt.Axes, values: np.ndarray, *, percent: bool = False) -> None:
    ymax = max(float(np.nanmax(values)) if values.size else 0.0, 1.0e-12)
    for patch, value in zip(ax.patches, values, strict=False):
        label = f"{100.0 * value:.1f}" if percent else f"{value:.3f}"
        y = patch.get_height()
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, y),
            xytext=(0, 3 if y > 0 else 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
            clip_on=False,
        )
    ax.set_ylim(top=max(ax.get_ylim()[1], ymax * 1.18 + 1.0e-6))


def _finish(fig: plt.Figure, output_path: str | Path, *, tight: bool = True) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def joint_violation_chart(
    summary: pd.DataFrame,
    alpha: float,
    output_path: str | Path,
    *,
    evaluation_label: str = "Held-out",
) -> Path:
    frame = summary.copy()
    labels = frame["certificate"].str.upper() + "\n" + frame["allocation"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    col = _joint_column(frame)
    values = frame[col].to_numpy(dtype=float)
    colors = [COLORS["optimized"] if value == "optimized" else COLORS["equal"] for value in frame["allocation"]]
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(alpha, color=COLORS["danger"], linestyle="--", linewidth=1.2, label=f"target alpha={alpha:.2f}")
    ax.set_ylabel(f"{evaluation_label} joint violation")
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    _annotate_bars(ax, values)
    legend_handles = [
        Patch(facecolor=COLORS["equal"], edgecolor="black", label="equal allocation"),
        Patch(facecolor=COLORS["optimized"], edgecolor="black", label="optimized allocation"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        [*legend_handles, *line_handles],
        [h.get_label() for h in legend_handles] + line_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncols=2,
        fontsize=FIG_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.78)
    return _finish(fig, output_path, tight=False)


def budget_scatter(
    driver: np.ndarray,
    alpha_vec: np.ndarray,
    xlabel: str,
    output_path: str | Path,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    log_y: bool = False,
    top_k: int = 0,
) -> Path:
    alpha = np.asarray(alpha_vec, dtype=float)
    x = np.asarray(driver, dtype=float)
    share = alpha / alpha.sum()
    with_rank_panel = labels is not None and top_k > 0
    if with_rank_panel:
        fig, (ax, panel) = plt.subplots(
            1,
            2,
            figsize=(10.2, 5.1),
            gridspec_kw={"width_ratios": [3.9, 2.2], "wspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        panel = None
    finite = np.isfinite(x) & np.isfinite(share) & (share > 0.0)
    ax.scatter(
        x[finite],
        share[finite],
        s=30,
        color="#8a8a8a",
        alpha=0.58,
        edgecolor="white",
        linewidth=0.2,
        label="All scalar constraints",
    )
    if log_y and finite.any():
        ax.set_yscale("log")
        ymin = max(float(np.nanmin(share[finite])) * 0.65, 1.0e-9)
        ymax = min(max(float(np.nanmax(share[finite])) * 1.45, ymin * 10.0), 1.2)
        ax.set_ylim(ymin, ymax)
    top_indices: list[int] = []
    if labels is not None and top_k > 0 and finite.any():
        label_array = np.asarray(labels, dtype=object)
        order = np.argsort(share)[::-1][: min(int(top_k), len(share))]
        top_indices = [int(index) for index in order if finite[index]]
        if top_indices:
            ax.scatter(
                x[top_indices],
                share[top_indices],
                s=105,
                color=COLORS["danger"],
                edgecolor="black",
                linewidth=0.55,
                zorder=3,
                label=f"Top {len(top_indices)} budget shares",
            )
        for rank, index in enumerate(top_indices, start=1):
            # The exact top-rank identities are shown in the side panel. Avoid
            # numeric labels on the scatter itself because clustered constraints
            # can otherwise become unreadable.
            _ = rank, index
        if panel is not None:
            panel.set_axis_off()
            panel.text(0.0, 0.98, "Top budget ranks", ha="left", va="top", fontsize=FIG_FONT_SIZE, weight="bold")
            panel.text(
                0.0,
                0.88,
                "Red points mark these rows;\nmarker size is highlight only.",
                ha="left",
                va="top",
                fontsize=FIG_FONT_SIZE,
                color=COLORS["neutral"],
            )
            y = 0.64
            for rank, index in enumerate(top_indices, start=1):
                short = textwrap.shorten(str(label_array[index]), width=22, placeholder="...")
                panel.text(
                    0.0,
                    y,
                    f"{rank}. {short} ({100.0 * share[index]:.1f}%)",
                    ha="left",
                    va="top",
                    fontsize=FIG_FONT_SIZE,
                    color="#222222",
                )
                y -= 0.105
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Optimized budget share")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper left")
    if with_rank_panel:
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.16, top=0.88, wspace=0.06)
    return _finish(fig, output_path, tight=not with_rank_panel)


def top_budget_bar(labels: list[str], shares: np.ndarray, output_path: str | Path) -> Path:
    clean_labels = [textwrap.shorten(str(label), width=42, placeholder="...") for label in labels]
    fig, ax = plt.subplots(figsize=(7.8, max(3.6, 0.25 * len(clean_labels) + 0.8)))
    y = np.arange(len(labels))
    ax.barh(y, shares, color=COLORS["optimized"])
    ax.set_yticks(y, clean_labels)
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Budget share")
    return _finish(fig, output_path)


def calibration_vs_heldout_scatter(
    calibration: np.ndarray,
    heldout: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    x = np.asarray(calibration, dtype=float)
    y = np.asarray(heldout, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(x, y, color=COLORS["equal"], alpha=0.75, edgecolor="black", linewidth=0.2)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
        hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
        ax.plot([lo, hi], [lo, hi], color=COLORS["danger"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE)
    return _finish(fig, output_path)
