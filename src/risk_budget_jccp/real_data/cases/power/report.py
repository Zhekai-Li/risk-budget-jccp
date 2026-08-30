from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from risk_budget_jccp.real_data.cases.power.build_instance import PowerInstance
from risk_budget_jccp.real_data.cases.power.solve import _y_values
from risk_budget_jccp.real_data.common.artifacts import (
    cleanup_legacy_budget_artifacts,
    remove_certificate_budget_artifacts,
    should_publish_budget_assets,
)
from risk_budget_jccp.real_data.common.evaluation import evaluation_label, write_split_summary
from risk_budget_jccp.real_data.common.latex import write_latex_table
from risk_budget_jccp.real_data.common.logging_utils import read_json
from risk_budget_jccp.real_data.common.paths import PROCESSED_ROOT, RAW_ROOT, RESULTS_ROOT
from risk_budget_jccp.real_data.common.plotting import (
    COLORS,
    FIG_FONT_SIZE,
    joint_violation_chart,
    top_budget_bar,
)


def _solution_payloads(root: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for path in sorted((root / "solutions").glob("power_*_*.json")):
        payload = read_json(path)
        payload["_solution_file"] = path.name
        payloads.append(payload)
    return payloads


def _metadata(processed_root: Path | None = None) -> pd.DataFrame:
    root = processed_root if processed_root is not None else PROCESSED_ROOT / "power"
    return pd.read_csv(root / "line_metadata.csv")


def _read_nonempty_csv(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return None
    return frame if not frame.empty else None


def _solution_status(payload: dict[str, object]) -> str:
    if payload.get("result_status"):
        return str(payload["result_status"])
    if payload.get("fallback_used"):
        return "fallback_equal"
    return str(payload.get("solver_status", "unknown"))


def _power_budget_metadata(meta: pd.DataFrame, payload: dict[str, object]) -> pd.DataFrame:
    alpha = np.asarray(payload.get("alpha_vector", []), dtype=float)
    frame = meta.copy()
    if alpha.size != len(frame):
        raise ValueError("Power alpha vector length does not match line metadata")
    thermal = frame["thermal_limit"].to_numpy(dtype=float)
    base_flow = frame["base_flow"].to_numpy(dtype=float)
    flow_std = frame["flow_std"].to_numpy(dtype=float)
    base_utilization = frame["base_utilization"].to_numpy(dtype=float)
    renewable_exposure = frame["renewable_ptdf_exposure"].to_numpy(dtype=float)
    headroom = np.maximum(
        thermal - np.abs(base_flow),
        1.0e-6,
    )
    normalized_headroom = np.maximum(headroom / np.maximum(thermal, 1.0e-6), 1.0e-6)
    theta = np.asarray(payload.get("theta_variables", np.zeros(len(frame))), dtype=float)
    certificate_values = np.asarray(payload.get("certificate_values", np.zeros(len(frame))), dtype=float)
    if theta.size != len(frame):
        theta = np.zeros(len(frame), dtype=float)
    if certificate_values.size != len(frame):
        certificate_values = np.zeros(len(frame), dtype=float)
    heldout_rate = np.asarray(
        payload.get("heldout_scalar_violation_rates", np.zeros(len(frame))),
        dtype=float,
    )
    if heldout_rate.size != len(frame):
        heldout_rate = np.zeros(len(frame), dtype=float)
    calibration_rate = np.asarray(
        payload.get("calibration_scalar_violation_rates", np.zeros(len(frame))),
        dtype=float,
    )
    if calibration_rate.size != len(frame):
        calibration_rate = np.zeros(len(frame), dtype=float)
    certificate_dual = np.asarray(payload.get("certificate_dual_values", np.full(len(frame), np.nan)), dtype=float)
    if certificate_dual.size != len(frame):
        certificate_dual = np.full(len(frame), np.nan, dtype=float)
    alpha_lower_dual = np.asarray(payload.get("alpha_lower_duals", np.full(len(frame), np.nan)), dtype=float)
    if alpha_lower_dual.size != len(frame):
        alpha_lower_dual = np.full(len(frame), np.nan, dtype=float)
    alpha_lower_bound = float(payload.get("alpha_lower_bound", np.nan))
    active_indicator = (np.abs(certificate_values) <= 1.0e-4).astype(float)
    is_active = active_indicator.astype(bool)
    if np.isfinite(alpha_lower_bound):
        is_alpha_interior = alpha > alpha_lower_bound * (1.0 + 1.0e-5)
    else:
        is_alpha_interior = alpha > 0.0
    is_dual_finite = np.isfinite(certificate_dual) & np.isfinite(theta)
    is_kkt_comparable = is_active & is_alpha_interior & is_dual_finite & (certificate_dual > 1.0e-10)
    theory_budget_driver = certificate_dual * theta
    kkt_total = float(np.nansum(theory_budget_driver[is_kkt_comparable]))
    kkt_implied_budget_share = np.full(len(frame), np.nan, dtype=float)
    if kkt_total > 0.0:
        kkt_implied_budget_share[is_kkt_comparable] = theory_budget_driver[is_kkt_comparable] / kkt_total
    marginal_budget_value = np.divide(
        theory_budget_driver,
        alpha,
        out=np.full(len(frame), np.nan, dtype=float),
        where=alpha > 0.0,
    )
    frame["available_headroom"] = headroom
    frame["normalized_headroom"] = normalized_headroom
    frame["flow_std_over_headroom"] = flow_std / headroom
    frame["flow_std_times_utilization"] = flow_std * base_utilization
    frame["abs_base_flow_std_over_headroom"] = np.abs(base_flow) * flow_std / headroom
    frame["flow_std_times_ptdf_exposure"] = flow_std * renewable_exposure
    frame["flow_std_util_over_normalized_headroom"] = flow_std * base_utilization / normalized_headroom
    frame["theta_variable"] = theta
    frame["certificate_value"] = certificate_values
    frame["certificate_active_indicator"] = active_indicator
    frame["certificate_dual_value"] = certificate_dual
    frame["alpha_lower_dual"] = alpha_lower_dual
    frame["alpha_lower_bound"] = alpha_lower_bound
    frame["bernstein_uncertainty_scale"] = theta
    frame["theory_budget_driver"] = theory_budget_driver
    frame["kkt_implied_budget_share"] = kkt_implied_budget_share
    frame["marginal_budget_value"] = marginal_budget_value
    frame["is_certificate_active"] = is_active
    frame["is_alpha_interior"] = is_alpha_interior
    frame["is_kkt_comparable"] = is_kkt_comparable
    frame["theta_times_flow_std_times_utilization"] = theta * flow_std * base_utilization
    frame["active_indicator_times_flow_std_times_utilization"] = active_indicator * flow_std * base_utilization
    frame["heldout_rate_times_flow_std_times_utilization"] = heldout_rate * flow_std * base_utilization
    frame["optimized_alpha"] = alpha
    frame["budget_share"] = alpha / alpha.sum()
    frame["heldout_scalar_violation_rate"] = heldout_rate
    frame["calibration_scalar_violation_rate"] = calibration_rate
    frame["rank"] = frame["budget_share"].rank(ascending=False, method="first").astype(int)
    return frame


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str, mask: pd.Series | None = None) -> float:
    pair = pd.concat([x.astype(float), y.astype(float)], axis=1).replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        pair = pair.loc[mask.astype(bool)]
    pair = pair.dropna()
    if len(pair) < 3 or pair.iloc[:, 0].std() <= 0.0 or pair.iloc[:, 1].std() <= 0.0:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _top_overlap(x: pd.Series, y: pd.Series, *, k: int = 4, mask: pd.Series | None = None) -> int:
    pair = pd.concat(
        [x.astype(float).rename("x"), y.astype(float).rename("y")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        pair = pair.loc[mask.astype(bool)]
    pair = pair.dropna()
    if pair.empty:
        return 0
    k_eff = min(k, len(pair))
    top_x = set(pair.sort_values("x", ascending=False).head(k_eff).index)
    top_y = set(pair.sort_values("y", ascending=False).head(k_eff).index)
    return len(top_x & top_y)


def _power_driver_correlation_table(frame: pd.DataFrame, tables: Path, *, certificate: str) -> None:
    candidates = [
        ("Residual std / headroom", "flow_std_over_headroom", "ex ante", None),
        ("Residual std x base utilization", "flow_std_times_utilization", "ex ante", None),
        ("Abs. base flow x residual std / headroom", "abs_base_flow_std_over_headroom", "ex ante", None),
        ("Residual std x utilization / normalized headroom", "flow_std_util_over_normalized_headroom", "ex ante", None),
        ("Residual std x renewable PTDF exposure", "flow_std_times_ptdf_exposure", "ex ante", None),
        ("Bernstein KKT-implied budget share", "kkt_implied_budget_share", "Bernstein active-set KKT", "is_kkt_comparable"),
        ("Theta x residual std x utilization", "theta_times_flow_std_times_utilization", "certificate diagnostic", None),
        ("Active indicator x residual std x utilization", "active_indicator_times_flow_std_times_utilization", "certificate diagnostic", None),
        ("Held-out scalar violation rate", "heldout_scalar_violation_rate", "audit only", None),
        ("Held-out violation rate x residual std x utilization", "heldout_rate_times_flow_std_times_utilization", "audit only", None),
    ]
    rows = []
    for label, column, timing, mask_column in candidates:
        mask = frame[mask_column] if mask_column is not None and mask_column in frame.columns else None
        rows.append(
            {
                "Proxy": label,
                "Use": timing,
                "Rows used": int(mask.sum()) if mask is not None else int(frame[column].replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
                "Pearson": _safe_corr(frame[column], frame["budget_share"], method="pearson", mask=mask),
                "Spearman": _safe_corr(frame[column], frame["budget_share"], method="spearman", mask=mask),
                "Top-4 overlap": _top_overlap(frame[column], frame["budget_share"], k=4, mask=mask),
            }
        )
    write_latex_table(pd.DataFrame(rows), tables / f"tab_power_{certificate}_driver_correlations.tex")


def _plot_power_budget_driver_comparison(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    panels = [
        ("Ex-ante risk/tightness", "flow_std_over_headroom", "Residual std / headroom"),
        ("Ex-ante value proxy", "flow_std_util_over_normalized_headroom", "Residual std x utilization / normalized headroom"),
        ("Bernstein active-set KKT", "kkt_implied_budget_share", "KKT-implied budget share"),
        ("Held-out audit proxy", "heldout_rate_times_flow_std_times_utilization", "Held-out violation rate x residual std x utilization"),
    ]
    y = frame["budget_share"].to_numpy(dtype=float)
    comparable = set(frame.index[frame["is_kkt_comparable"].astype(bool)].tolist())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharey=True)
    for ax, (panel_title, column, xlabel) in zip(axes.ravel(), panels, strict=True):
        x = frame[column].to_numpy(dtype=float)
        colors = []
        sizes = []
        for idx in frame.index:
            if column == "kkt_implied_budget_share" and idx in comparable:
                colors.append(COLORS["accent"])
                sizes.append(70)
            else:
                colors.append("#8a8a8a")
                sizes.append(30)
        finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if column == "kkt_implied_budget_share":
            finite &= frame["is_kkt_comparable"].to_numpy(dtype=bool)
        ax.scatter(x[finite], y[finite], s=np.asarray(sizes)[finite], color=np.asarray(colors, dtype=object)[finite], alpha=0.75, edgecolor="black", linewidth=0.35)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelsize=FIG_FONT_SIZE)
    axes[0, 0].set_ylabel("Optimized budget share")
    axes[1, 0].set_ylabel("Optimized budget share")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a8a", markeredgecolor="black", markersize=7, label="All constraints"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["accent"], markeredgecolor="black", markersize=8, label="KKT comparable"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.02), fontsize=FIG_FONT_SIZE)
    fig.suptitle(title, y=1.08, fontsize=FIG_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_power_certificate_budget_assets(
    meta: pd.DataFrame,
    payload: dict[str, object],
    *,
    certificate: str,
    figures: Path,
    tables: Path,
    split: str = "out_of_sample",
) -> pd.DataFrame:
    if not should_publish_budget_assets(payload):
        remove_certificate_budget_artifacts("power", certificate, figures, tables)
        return pd.DataFrame()
    alpha = np.asarray(payload.get("alpha_vector", []), dtype=float)
    frame = _power_budget_metadata(meta, payload)
    rate_name = "calibration_scalar_violation_rate" if split == "in_sample" else "heldout_scalar_violation_rate"
    top = frame.sort_values("budget_share", ascending=False).head(10)
    status = _solution_status(payload)
    columns = [
        "rank",
        "snapshot",
        "branch_id",
        "from_bus",
        "to_bus",
        "direction",
        "risk_tier",
        "thermal_limit",
        "base_flow",
        "base_utilization",
        "flow_std",
        "available_headroom",
        "flow_std_over_headroom",
        "flow_std_times_utilization",
        "abs_base_flow_std_over_headroom",
        "flow_std_util_over_normalized_headroom",
        "flow_std_times_ptdf_exposure",
        "theta_variable",
        "certificate_active_indicator",
        "certificate_dual_value",
        "bernstein_uncertainty_scale",
        "theory_budget_driver",
        "kkt_implied_budget_share",
        "marginal_budget_value",
        "is_kkt_comparable",
        "theta_times_flow_std_times_utilization",
        "active_indicator_times_flow_std_times_utilization",
        "heldout_rate_times_flow_std_times_utilization",
        "renewable_ptdf_exposure",
        "optimized_alpha",
        "budget_share",
        rate_name,
    ]
    write_latex_table(top.loc[:, columns], tables / f"tab_power_{certificate}_top_budgets.tex")
    labels = (
        frame["snapshot"].astype(str)
        + " "
        + frame["branch_id"].astype(str)
        + " "
        + frame["direction"].astype(str)
    ).tolist()
    _power_driver_correlation_table(frame, tables, certificate=certificate)
    _plot_power_budget_driver_comparison(
        frame,
        figures / f"fig_power_{certificate}_budget_vs_flow_risk_driver.pdf",
        title=f"Power {certificate.upper()} budget allocation ({status})",
    )
    top_budget_bar(labels=np.asarray(labels, dtype=object)[top.index].tolist(), shares=top["budget_share"].to_numpy(dtype=float), output_path=figures / f"fig_power_{certificate}_top_budget_lines.pdf")
    return frame


def _certificate_residual_audit(payloads: list[dict[str, object]], meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for payload in payloads:
        cert = np.asarray(payload.get("certificate_values", []), dtype=float)
        alpha = np.asarray(payload.get("alpha_vector", []), dtype=float)
        if cert.size != len(meta) or alpha.size != len(meta):
            continue
        frame = meta.copy()
        alpha_sum = float(alpha.sum()) if alpha.size else 0.0
        frame["certificate"] = str(payload["certificate"])
        frame["allocation"] = str(payload["allocation"])
        frame["solver_status"] = str(payload.get("solver_status", ""))
        frame["fallback_used"] = bool(payload.get("fallback_used", False))
        frame["alpha_i"] = alpha
        frame["budget_share"] = alpha / alpha_sum if alpha_sum > 0.0 else np.nan
        frame["certificate_value"] = cert
        frame["positive_certificate_violation"] = np.maximum(cert, 0.0)
        frame["calibration_scalar_violation_rate"] = np.asarray(
            payload.get("calibration_scalar_violation_rates", np.zeros(len(meta))),
            dtype=float,
        )
        frame["heldout_scalar_violation_rate"] = np.asarray(
            payload.get("heldout_scalar_violation_rates", np.zeros(len(meta))),
            dtype=float,
        )
        rows.append(frame.sort_values("certificate_value", ascending=False).head(10))
    if not rows:
        return pd.DataFrame(
            columns=[
                "certificate",
                "allocation",
                "branch_id",
                "direction",
                "certificate_value",
                "positive_certificate_violation",
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    columns = [
        "certificate",
        "allocation",
        "snapshot",
        "branch_id",
        "from_bus",
        "to_bus",
        "direction",
        "risk_tier",
        "thermal_limit",
        "base_flow",
        "base_utilization",
        "flow_std",
        "alpha_i",
        "budget_share",
        "certificate_value",
        "positive_certificate_violation",
        "calibration_scalar_violation_rate",
        "heldout_scalar_violation_rate",
        "fallback_used",
        "solver_status",
    ]
    return result.loc[:, [column for column in columns if column in result.columns]]


def _cantelli_infeasibility_diagnostics(payloads: list[dict[str, object]], instance: PowerInstance, meta: pd.DataFrame) -> pd.DataFrame:
    payload = next(
        (item for item in payloads if item.get("certificate") == "cantelli" and item.get("allocation") == "equal"),
        None,
    )
    if payload is None:
        return pd.DataFrame()
    p_value = np.asarray(payload.get("dispatch_mw", []), dtype=float)
    alpha = np.asarray(payload.get("alpha_vector", []), dtype=float)
    if p_value.size == 0 or alpha.size != len(meta):
        return pd.DataFrame()
    y_train = _y_values(instance, p_value, instance.flow_residual_train)
    mu = y_train.mean(axis=0)
    sigma = y_train.std(axis=0, ddof=0)
    q = np.sqrt((1.0 - alpha) / alpha)
    required_margin = sigma * q
    available_margin = np.maximum(-mu, 0.0)
    shortfall = required_margin - available_margin
    denominator = sigma**2 + available_margin**2
    beta = np.where(denominator > 0.0, sigma**2 / denominator, 0.0)
    beta = np.where(mu < 0.0, beta, 1.0)
    frame = meta.copy()
    frame["alpha_i_equal"] = alpha
    frame["mean_normalized_Y"] = mu
    frame["std_normalized_Y"] = sigma
    frame["required_cantelli_margin"] = required_margin
    frame["available_negative_mean_margin"] = available_margin
    frame["cantelli_margin_shortfall"] = shortfall
    frame["minimal_cantelli_budget_beta"] = beta
    frame["status"] = str(payload.get("solver_status", ""))
    columns = [
        "snapshot",
        "branch_id",
        "from_bus",
        "to_bus",
        "direction",
        "risk_tier",
        "thermal_limit",
        "base_flow",
        "base_utilization",
        "flow_std",
        "alpha_i_equal",
        "mean_normalized_Y",
        "std_normalized_Y",
        "required_cantelli_margin",
        "available_negative_mean_margin",
        "cantelli_margin_shortfall",
        "minimal_cantelli_budget_beta",
        "status",
    ]
    return frame.sort_values("cantelli_margin_shortfall", ascending=False).loc[:, columns].head(30)


def _write_power_diagnostic_tables(instance: PowerInstance, root: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = _solution_payloads(root)
    meta = _metadata()
    audit = _certificate_residual_audit(payloads, meta)
    cantelli = _cantelli_infeasibility_diagnostics(payloads, instance, meta)
    audit.to_csv(tables / "power_certificate_residual_audit.csv", index=False)
    cantelli.to_csv(tables / "power_cantelli_infeasibility.csv", index=False)
    write_latex_table(audit, tables / "tab_power_certificate_residual_audit.tex")
    write_latex_table(cantelli, tables / "tab_power_cantelli_infeasibility.tex")
    return audit, cantelli


def build_power_report_assets(
    instance: PowerInstance,
    summary: pd.DataFrame,
    results_dir: str | Path,
    processed_dir: str | Path | None = None,
) -> None:
    root = Path(results_dir)
    processed_root = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "power"
    report_ready = root.resolve().is_relative_to((RESULTS_ROOT / "runs").resolve())
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_budget_artifacts(root)
    write_latex_table(summary, tables / "tab_power_summary.tex")
    provenance_path = RAW_ROOT / "power" / "provenance.json"
    provenance = pd.DataFrame([{"Item": "Renewable/load source", "Value": "RTS-GMLC built-in time series or recorded fallback"}])
    if provenance_path.is_file():
        raw = read_json(provenance_path)
        provenance = pd.DataFrame([{"Item": key, "Value": str(value)} for key, value in raw.items()])
    write_latex_table(provenance, tables / "tab_power_data_provenance.tex")
    gen = pd.read_csv(processed_root / "generator_metadata.csv")
    cost_diag = pd.DataFrame(
        [
            {"Metric": "Dispatchable thermal generators", "Value": int(len(gen))},
            {"Metric": "Unique marginal costs", "Value": int(gen["cost"].nunique())},
            {"Metric": "Fallback cost count", "Value": int(gen["cost_source"].astype(str).str.contains("fallback").sum())},
            {"Metric": "Minimum cost", "Value": float(gen["cost"].min())},
            {"Metric": "Median cost", "Value": float(gen["cost"].median())},
            {"Metric": "Maximum cost", "Value": float(gen["cost"].max())},
        ]
    )
    write_latex_table(cost_diag, tables / "tab_power_generator_cost_diagnostics.tex")
    assumptions = pd.DataFrame(
        [
            {"Certificate": "CVaR", "Interpretation": "Empirical residual-flow tail certificate."},
            {"Certificate": "Bernstein", "Interpretation": "Theta/Bernstein residual-flow moment comparison."},
            {"Certificate": "Cantelli", "Interpretation": "Moment-based line-flow certificate."},
        ]
    )
    write_latex_table(assumptions, tables / "tab_power_certificate_assumptions.tex")
    _write_power_diagnostic_tables(instance, root, tables)
    for name in ("split_metadata", "constraint_design_audit", "gamma_selection_score"):
        path = processed_root / f"{name}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            write_latex_table(frame.head(30), tables / f"tab_power_{name}.tex")

    optimized = read_json(root / "solutions" / "power_bernstein_optimized.json")
    meta = _metadata(processed_root)
    meta = _power_budget_metadata(meta, optimized)
    for certificate in ("cvar", "bernstein", "cantelli"):
        payload = read_json(root / "solutions" / f"power_{certificate}_optimized.json")
        _write_power_certificate_budget_assets(_metadata(processed_root), payload, certificate=certificate, figures=figures, tables=tables)
    for name in ("candidate_constraints", "selected_constraints", "excluded_constraints"):
        path = processed_root / f"{name}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            write_latex_table(frame.head(30), tables / f"tab_power_{name}.tex")
    joint_violation_chart(summary, instance.alpha, figures / "fig_power_joint_violation.pdf")
    for split in ("in_sample", "out_of_sample"):
        split_root = RESULTS_ROOT / split / "power" if report_ready else root / split
        split_figures = split_root / "figures"
        split_tables = split_root / "tables"
        split_figures.mkdir(parents=True, exist_ok=True)
        split_tables.mkdir(parents=True, exist_ok=True)
        cleanup_legacy_budget_artifacts(split_root)
        split_summary = write_split_summary(summary, case="power", split=split, tables_dir=split_tables)
        split_payloads = {certificate: read_json(root / "solutions" / f"power_{certificate}_optimized.json") for certificate in ("cvar", "bernstein", "cantelli")}
        for certificate, payload in split_payloads.items():
            _write_power_certificate_budget_assets(_metadata(processed_root), payload, certificate=certificate, figures=split_figures, tables=split_tables, split=split)
        for diagnostic_name in ("power_certificate_residual_audit", "power_cantelli_infeasibility"):
            diagnostic_path = tables / f"{diagnostic_name}.csv"
            if diagnostic_path.is_file():
                diagnostic = _read_nonempty_csv(diagnostic_path)
                if diagnostic is not None:
                    write_latex_table(diagnostic, split_tables / f"tab_{diagnostic_name}.tex")
        for name in (
            "candidate_constraints",
            "selected_constraints",
            "excluded_constraints",
            "gamma_tuning_diagnostics",
            "selection_diagnostics",
            "split_metadata",
            "constraint_design_audit",
            "gamma_selection_score",
        ):
            path = processed_root / f"{name}.csv"
            if path.is_file():
                frame = pd.read_csv(path)
                write_latex_table(frame.head(30), split_tables / f"tab_power_{name}.tex")
        joint_violation_chart(split_summary, instance.alpha, split_figures / "fig_power_joint_violation.pdf", evaluation_label=evaluation_label(split))
