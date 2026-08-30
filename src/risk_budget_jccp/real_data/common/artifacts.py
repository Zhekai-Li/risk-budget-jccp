from __future__ import annotations

from pathlib import Path


CASES = ("m5", "power", "french")
CERTIFICATES = ("cvar", "bernstein", "cantelli")


def is_legacy_budget_artifact(path: str | Path) -> bool:
    """Return True for old budget artifacts that did not encode certificate type."""
    name = Path(path).name
    for case in CASES:
        if name == f"tab_{case}_top_budgets.tex":
            return True
        if name.startswith(f"fig_{case}_budget_"):
            return True
        if name.startswith(f"fig_{case}_top_budget_"):
            return True
    return False


def is_retired_real_data_artifact(path: str | Path) -> bool:
    """Return True for real-data figures retired from the publication artifact set."""
    name = Path(path).name
    if name.endswith("_budget_sorted.pdf"):
        return True
    if name.endswith("_violated_count_hist.pdf"):
        return True
    if name == "fig_power_gamma_tradeoff.pdf":
        return True
    if name in {"fig_real_data_algorithm_status.pdf", "fig_real_data_quality_categories.pdf"}:
        return True
    if name in {f"fig_{case}_objective_improvement.pdf" for case in CASES}:
        return True
    if name.startswith("fig_m5_") and name.endswith("_budget_vs_cv_price.pdf"):
        return True
    if name.startswith("fig_power_") and name.endswith("_budget_vs_volatility_utilization.pdf"):
        return True
    if name.startswith("fig_french_") and name.endswith("_budget_vs_weight_volatility.pdf"):
        return True
    return False


def cleanup_legacy_budget_artifacts(root: str | Path) -> list[Path]:
    """Delete stale or retired real-data figures/tables below root."""
    base = Path(root)
    if not base.exists():
        return []
    removed: list[Path] = []
    for path in base.rglob("*"):
        if (path.is_file() or path.is_symlink()) and (
            is_legacy_budget_artifact(path) or is_retired_real_data_artifact(path)
        ):
            path.unlink()
            removed.append(path)
    return removed


def remove_certificate_budget_artifacts(case: str, certificate: str, figures: str | Path, tables: str | Path) -> list[Path]:
    """Delete certificate-specific budget artifacts for a failed/fallback optimized row."""
    removed: list[Path] = []
    fig_dir = Path(figures)
    table_dir = Path(tables)
    patterns = [
        f"fig_{case}_{certificate}_budget_*.pdf",
        f"fig_{case}_{certificate}_top_budget_*.pdf",
    ]
    for pattern in patterns:
        for path in fig_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(path)
    table_path = table_dir / f"tab_{case}_{certificate}_top_budgets.tex"
    if table_path.is_file() or table_path.is_symlink():
        table_path.unlink()
        removed.append(table_path)
    return removed


def should_publish_budget_assets(payload: dict[str, object]) -> bool:
    """Publish budget interpretation only for accepted optimized-allocation rows."""
    status = str(payload.get("result_status", payload.get("solver_status", "")))
    if status != "success":
        return False
    if bool(payload.get("fallback_used", False)):
        return False
    if "valid_optimization" in payload and not bool(payload.get("valid_optimization")):
        return False
    return True
