from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.cases.french.build_instance import FrenchInstance
from risk_budget_jccp.real_data.common.artifacts import (
    cleanup_legacy_budget_artifacts,
    remove_certificate_budget_artifacts,
    should_publish_budget_assets,
)
from risk_budget_jccp.real_data.common.evaluation import evaluation_label, write_split_summary
from risk_budget_jccp.real_data.common.latex import copy_asset, write_latex_table
from risk_budget_jccp.real_data.common.logging_utils import read_json
from risk_budget_jccp.real_data.common.paths import PROCESSED_ROOT, RESULTS_ROOT
from risk_budget_jccp.real_data.common.plotting import (
    COLORS,
    FIG_FONT_SIZE,
    calibration_vs_heldout_scatter,
    joint_violation_chart,
    top_budget_bar,
)


def _solution_status(payload: dict[str, object]) -> str:
    if payload.get("result_status"):
        return str(payload["result_status"])
    if payload.get("fallback_used"):
        return "fallback_equal"
    return str(payload.get("solver_status", "unknown"))


def _payload_vector(payload: dict[str, object], key: str, length: int, *, default: float = np.nan) -> np.ndarray:
    values = np.asarray(payload.get(key, np.full(length, default)), dtype=float)
    if values.size != length:
        return np.full(length, default, dtype=float)
    return values


def _french_budget_metadata(
    instance: FrenchInstance,
    payload: dict[str, object],
    *,
    split: str = "out_of_sample",
) -> pd.DataFrame:
    alpha = np.asarray(payload["alpha_vector"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    margins = np.asarray(payload["margin_buffers"], dtype=float)
    metadata = instance.metadata.copy()
    metadata["optimized_weight"] = weights
    metadata["optimized_margin"] = margins
    metadata["optimized_alpha"] = alpha
    metadata["budget_share"] = alpha / alpha.sum()
    metadata["weight_volatility"] = metadata["volatility"].to_numpy(dtype=float) * weights
    metadata["weight_downside_cvar_95"] = metadata["downside_cvar_95"].to_numpy(dtype=float) * weights
    metadata["allocated_tail_cvar"] = _payload_vector(payload, "allocated_tail_cvar", len(metadata))
    metadata["margin_contribution"] = _payload_vector(payload, "margin_contribution", len(metadata), default=0.0)
    metadata["cvar_marginal_budget_value"] = _payload_vector(payload, "cvar_marginal_budget_value", len(metadata))
    metadata["cantelli_marginal_budget_value"] = _payload_vector(payload, "cantelli_marginal_budget_value", len(metadata))
    metadata["certificate_dual_value"] = _payload_vector(payload, "certificate_dual_values", len(metadata))
    metadata["theta_variable"] = _payload_vector(payload, "theta_variables", len(metadata))
    metadata["theory_budget_driver"] = _payload_vector(payload, "theory_budget_driver", len(metadata))
    metadata["theory_driver_source"] = str(payload.get("theory_driver_source", "unavailable"))
    certificate_values = _payload_vector(payload, "certificate_values", len(metadata))
    alpha_lower_bound = float(payload.get("alpha_lower_bound", np.nan))
    if np.isfinite(alpha_lower_bound):
        is_alpha_interior = alpha > alpha_lower_bound * (1.0 + 1.0e-5)
    else:
        is_alpha_interior = alpha > 0.0
    is_kkt_comparable = (
        np.isfinite(certificate_values)
        & (np.abs(certificate_values) <= 1.0e-4)
        & is_alpha_interior
        & np.isfinite(metadata["certificate_dual_value"].to_numpy(dtype=float))
        & (metadata["certificate_dual_value"].to_numpy(dtype=float) > 1.0e-10)
        & np.isfinite(metadata["theta_variable"].to_numpy(dtype=float))
    )
    kkt_driver = metadata["theory_budget_driver"].to_numpy(dtype=float)
    kkt_total = float(np.nansum(kkt_driver[is_kkt_comparable]))
    kkt_implied = np.full(len(metadata), np.nan, dtype=float)
    if kkt_total > 0.0:
        kkt_implied[is_kkt_comparable] = kkt_driver[is_kkt_comparable] / kkt_total
    metadata["certificate_value"] = certificate_values
    metadata["is_kkt_comparable"] = is_kkt_comparable
    metadata["kkt_implied_budget_share"] = kkt_implied
    if split == "in_sample":
        y_eval = -instance.returns_train * weights[None, :] - margins[None, :]
        rate_name = "calibration_scalar_violation_rate"
    else:
        y_eval = -instance.returns_test * weights[None, :] - margins[None, :]
        rate_name = "heldout_scalar_violation_rate"
    metadata[rate_name] = (y_eval > 1.0e-9).mean(axis=0)
    metadata["rank"] = metadata["budget_share"].rank(ascending=False, method="first").astype(int)
    return metadata


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str, mask: pd.Series | None = None) -> float:
    pair = pd.concat([x.astype(float), y.astype(float)], axis=1).replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        pair = pair.loc[mask.astype(bool)]
    pair = pair.dropna()
    if len(pair) < 3 or pair.iloc[:, 0].std() <= 0.0 or pair.iloc[:, 1].std() <= 0.0:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _top_overlap(x: pd.Series, y: pd.Series, *, k: int = 6, mask: pd.Series | None = None) -> int:
    pair = pd.concat([x.astype(float).rename("x"), y.astype(float).rename("y")], axis=1).replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        pair = pair.loc[mask.astype(bool)]
    pair = pair.dropna()
    if pair.empty:
        return 0
    k_eff = min(k, len(pair))
    return len(set(pair.sort_values("x", ascending=False).head(k_eff).index) & set(pair.sort_values("y", ascending=False).head(k_eff).index))


def _write_french_driver_correlation_table(metadata: pd.DataFrame, tables: Path, *, certificate: str) -> None:
    candidates = [
        ("Weight x volatility", "weight_volatility", "exposure proxy"),
        ("Weight x downside CVaR 95", "weight_downside_cvar_95", "tail exposure proxy"),
        ("Weight x allocated CVaR(alpha_i)", "allocated_tail_cvar", "allocated tail proxy"),
        ("Margin contribution x_i", "margin_contribution", "objective contribution proxy"),
        ("Held-out scalar violation rate", "heldout_scalar_violation_rate", "audit only"),
    ]
    if certificate == "cvar":
        candidates.insert(4, ("CVaR marginal budget value", "cvar_marginal_budget_value", "theory diagnostic"))
    elif certificate == "cantelli":
        candidates.insert(4, ("Cantelli marginal equalization value", "cantelli_marginal_budget_value", "equalization diagnostic"))
    elif certificate == "bernstein":
        candidates.insert(4, ("Bernstein KKT-implied budget share", "kkt_implied_budget_share", "Bernstein active-set KKT"))
    rows = []
    for label, column, use in candidates:
        if column not in metadata:
            continue
        x = metadata[column]
        if x.replace([np.inf, -np.inf], np.nan).dropna().empty:
            continue
        mask = metadata["is_kkt_comparable"] if certificate == "bernstein" and column == "kkt_implied_budget_share" else None
        rows.append(
            {
                "Proxy": label,
                "Use": use,
                "Rows used": int(mask.sum()) if mask is not None else int(x.replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
                "Pearson": _safe_corr(x, metadata["budget_share"], method="pearson", mask=mask),
                "Spearman": _safe_corr(x, metadata["budget_share"], method="spearman", mask=mask),
                "Top-6 overlap": _top_overlap(x, metadata["budget_share"], k=6, mask=mask),
            }
        )
    write_latex_table(pd.DataFrame(rows), tables / f"tab_french_{certificate}_driver_correlations.tex")


def _plot_french_budget_driver_panels(metadata: pd.DataFrame, output_path: Path, *, title: str, certificate: str) -> None:
    theory_label = {
        "cvar": "CVaR marginal diagnostic",
        "cantelli": "Cantelli equalization diagnostic",
        "bernstein": "Bernstein active-set KKT",
    }.get(certificate, "Theory driver")
    theory_column = {
        "cvar": "cvar_marginal_budget_value",
        "cantelli": "cantelli_marginal_budget_value",
        "bernstein": "kkt_implied_budget_share",
    }.get(certificate, "theory_budget_driver")
    panels = [
        ("Exposure proxy", "weight_volatility", "Weight x volatility"),
        ("Tail exposure proxy", "weight_downside_cvar_95", "Weight x downside CVaR 95"),
        ("Objective contribution", "margin_contribution", "Margin contribution x_i"),
        (theory_label, theory_column, "KKT-implied budget share" if certificate == "bernstein" else "Certificate diagnostic"),
    ]
    if certificate == "cvar":
        panels = [
            ("Tail exposure proxy", "weight_downside_cvar_95", "Weight x downside CVaR 95"),
            ("Allocated tail exposure", "allocated_tail_cvar", r"Weight x CVaR(alpha_i)"),
            ("Objective contribution", "margin_contribution", "Margin contribution x_i"),
            ("CVaR marginal diagnostic", "cvar_marginal_budget_value", r"-w_i q_i'(alpha_i)"),
        ]
    y = metadata["budget_share"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharey=True)
    for ax, (panel_title, column, xlabel) in zip(axes.ravel(), panels, strict=True):
        x = metadata[column].to_numpy(dtype=float) if column in metadata else np.full(len(metadata), np.nan)
        finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if column == "kkt_implied_budget_share":
            finite &= metadata["is_kkt_comparable"].to_numpy(dtype=bool)
        colors = [COLORS["accent"] if column == "kkt_implied_budget_share" else "#8a8a8a" for _ in metadata.index]
        sizes = [65 if column == "kkt_implied_budget_share" else 30 for _ in metadata.index]
        ax.scatter(x[finite], y[finite], s=np.asarray(sizes)[finite], color=np.asarray(colors, dtype=object)[finite], alpha=0.75, edgecolor="black", linewidth=0.35)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelsize=FIG_FONT_SIZE)
    axes[0, 0].set_ylabel("Optimized budget share")
    axes[1, 0].set_ylabel("Optimized budget share")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a8a", markeredgecolor="black", markersize=7, label="All industries"),
    ]
    if certificate == "bernstein":
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["accent"], markeredgecolor="black", markersize=8, label="KKT comparable"))
    fig.legend(handles=handles, frameon=False, loc="upper center", ncols=len(handles), bbox_to_anchor=(0.5, 1.02), fontsize=FIG_FONT_SIZE)
    fig.suptitle(title, y=1.08, fontsize=FIG_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_french_certificate_budget_assets(
    instance: FrenchInstance,
    payload: dict[str, object],
    *,
    certificate: str,
    figures: Path,
    tables: Path,
    split: str = "out_of_sample",
) -> pd.DataFrame:
    if not should_publish_budget_assets(payload):
        remove_certificate_budget_artifacts("french", certificate, figures, tables)
        return pd.DataFrame()
    alpha = np.asarray(payload["alpha_vector"], dtype=float)
    metadata = _french_budget_metadata(instance, payload, split=split)
    top = metadata.sort_values("budget_share", ascending=False).head(10)
    status = _solution_status(payload)
    rate_name = "calibration_scalar_violation_rate" if split == "in_sample" else "heldout_scalar_violation_rate"
    columns = [
        "rank",
        "industry",
        "mean_return",
        "volatility",
        "downside_cvar_95",
        "optimized_weight",
        "optimized_margin",
        "weight_downside_cvar_95",
        "allocated_tail_cvar",
        "margin_contribution",
        "cvar_marginal_budget_value",
        "cantelli_marginal_budget_value",
        "theory_budget_driver",
        "kkt_implied_budget_share",
        "is_kkt_comparable",
        "optimized_alpha",
        "budget_share",
        rate_name,
    ]
    write_latex_table(top.loc[:, columns], tables / f"tab_french_{certificate}_top_budgets.tex")
    labels = metadata["industry"].astype(str).tolist()
    _write_french_driver_correlation_table(metadata, tables, certificate=certificate)
    _plot_french_budget_driver_panels(
        metadata,
        figures / f"fig_french_{certificate}_budget_vs_portfolio_tail_exposure.pdf",
        title=f"French {certificate.upper()} budget allocation ({status})",
        certificate=certificate,
    )
    top_budget_bar(
        top["industry"].astype(str).tolist(),
        top["budget_share"].to_numpy(dtype=float),
        figures / f"fig_french_{certificate}_top_budget_industries.pdf",
    )
    return metadata


def build_french_report_assets(
    instance: FrenchInstance,
    summary: pd.DataFrame,
    results_dir: str | Path,
    processed_dir: str | Path | None = None,
) -> None:
    root = Path(results_dir)
    processed_root = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "french"
    report_ready = root.resolve().is_relative_to((RESULTS_ROOT / "runs").resolve())
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_budget_artifacts(root)
    write_latex_table(summary, tables / "tab_french_summary.tex")
    split_path = processed_root / "split_metadata.csv"
    if split_path.is_file():
        write_latex_table(pd.read_csv(split_path), tables / "tab_french_split_metadata.tex")
    regime_path = processed_root / "regime_shift.csv"
    if regime_path.is_file():
        regime = pd.read_csv(regime_path)
        write_latex_table(regime.sort_values("heldout_to_calibration_volatility_ratio", ascending=False).head(30), tables / "tab_french_regime_shift.tex")
    assumptions = pd.DataFrame(
        [
            {"Certificate": "CVaR", "Interpretation": "Empirical tail certificate; most natural for heavy-tailed returns."},
            {"Certificate": "Bernstein", "Interpretation": "Nominal Gaussian/sub-Gaussian comparison using daily moments."},
            {"Certificate": "Cantelli", "Interpretation": "Distribution-free one-sided moment certificate; conservative."},
        ]
    )
    write_latex_table(assumptions, tables / "tab_french_certificate_assumptions.tex")

    optimized = read_json(root / "solutions" / "french_cvar_optimized.json")
    weights = np.asarray(optimized["weights"], dtype=float)
    metadata = _french_budget_metadata(instance, optimized)
    for certificate in ("cvar", "bernstein", "cantelli"):
        payload = read_json(root / "solutions" / f"french_{certificate}_optimized.json")
        _write_french_certificate_budget_assets(instance, payload, certificate=certificate, figures=figures, tables=tables)

    joint_violation_chart(summary, instance.alpha, figures / "fig_french_joint_violation.pdf")
    if {"heldout_volatility", "volatility"}.issubset(metadata.columns):
        calibration_vs_heldout_scatter(
            metadata["volatility"].to_numpy(dtype=float),
            metadata["heldout_volatility"].to_numpy(dtype=float),
            xlabel="Calibration volatility",
            ylabel="Held-out volatility",
            output_path=figures / "fig_french_calibration_vs_heldout_volatility.pdf",
        )

    for split in ("in_sample", "out_of_sample"):
        split_root = RESULTS_ROOT / split / "french" if report_ready else root / split
        split_figures = split_root / "figures"
        split_tables = split_root / "tables"
        split_figures.mkdir(parents=True, exist_ok=True)
        split_tables.mkdir(parents=True, exist_ok=True)
        cleanup_legacy_budget_artifacts(split_root)
        split_summary = write_split_summary(summary, case="french", split=split, tables_dir=split_tables)
        split_payloads = {certificate: read_json(root / "solutions" / f"french_{certificate}_optimized.json") for certificate in ("cvar", "bernstein", "cantelli")}
        split_metadata = _french_budget_metadata(instance, optimized, split=split)
        margins = np.asarray(optimized["margin_buffers"], dtype=float)
        if split == "in_sample":
            y_split = -instance.returns_train * weights[None, :] - margins[None, :]
            rate_name = "calibration_scalar_violation_rate"
        else:
            y_split = -instance.returns_test * weights[None, :] - margins[None, :]
            rate_name = "heldout_scalar_violation_rate"
        split_metadata[rate_name] = (y_split > 1.0e-9).mean(axis=0)
        split_metadata["rank"] = split_metadata["budget_share"].rank(ascending=False, method="first").astype(int)
        for certificate, payload in split_payloads.items():
            _write_french_certificate_budget_assets(instance, payload, certificate=certificate, figures=split_figures, tables=split_tables, split=split)
        for audit_name in ("tab_french_split_metadata.tex", "tab_french_regime_shift.tex"):
            audit_path = tables / audit_name
            if audit_path.is_file():
                copy_asset(audit_path, split_tables)
        joint_violation_chart(split_summary, instance.alpha, split_figures / "fig_french_joint_violation.pdf", evaluation_label=evaluation_label(split))
        if {"heldout_volatility", "volatility"}.issubset(split_metadata.columns):
            calibration_vs_heldout_scatter(
                split_metadata["volatility"].to_numpy(dtype=float),
                split_metadata["heldout_volatility"].to_numpy(dtype=float),
                xlabel="Calibration volatility",
                ylabel="Held-out volatility",
                output_path=split_figures / "fig_french_calibration_vs_heldout_volatility.pdf",
            )
