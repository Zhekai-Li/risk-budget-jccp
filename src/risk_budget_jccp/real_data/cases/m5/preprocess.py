from __future__ import annotations

from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.common.paths import case_processed_dir


def _read_member(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if Path(name).name == member:
                with archive.open(name) as handle:
                    return pd.read_csv(handle)
    raise FileNotFoundError(f"{member} not found in {zip_path}")


def _quantile_bins(values: pd.Series, bins: int) -> pd.Series:
    unique = values.nunique(dropna=True)
    if unique <= 1:
        return pd.Series(["all"] * len(values), index=values.index)
    q = min(int(bins), int(unique))
    labels = [f"q{i + 1}" for i in range(q)]
    return pd.qcut(values.rank(method="first"), q=q, labels=labels, duplicates="drop").astype(str)


def _select_stratified(
    eligible: pd.DataFrame,
    *,
    n_series: int,
    max_category_share: float,
    revenue_quantile_bins: int,
    cv_quantile_bins: int,
) -> pd.DataFrame:
    frame = eligible.copy()
    frame["revenue_bin"] = _quantile_bins(frame["revenue_mean"], revenue_quantile_bins)
    frame["cv_bin"] = _quantile_bins(frame["coefficient_of_variation"], cv_quantile_bins)
    revenue_rank = frame["revenue_mean"].rank(pct=True)
    cv_rank = frame["coefficient_of_variation"].rank(pct=True)
    frame["_selection_score"] = revenue_rank + cv_rank
    frame = frame.sort_values(["_selection_score", "revenue_mean", "id"], ascending=[False, False, True])

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    max_per_category = max(1, int(np.ceil(float(max_category_share) * int(n_series))))
    category_counts: dict[str, int] = {}

    groups = ["cat_id", "state_id", "revenue_bin", "cv_bin"]
    for group in groups:
        for _, part in frame.groupby(group, sort=True):
            for idx in part.index:
                cat = str(frame.loc[idx, "cat_id"])
                if idx in selected_set or category_counts.get(cat, 0) >= max_per_category:
                    continue
                selected_set.add(int(idx))
                selected_indices.append(int(idx))
                category_counts[cat] = category_counts.get(cat, 0) + 1
                break
            if len(selected_indices) >= int(n_series):
                break
        if len(selected_indices) >= int(n_series):
            break

    for idx in frame.index:
        cat = str(frame.loc[idx, "cat_id"])
        if idx in selected_set or category_counts.get(cat, 0) >= max_per_category:
            continue
        selected_set.add(int(idx))
        selected_indices.append(int(idx))
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(selected_indices) >= int(n_series):
            break

    if len(selected_indices) < int(n_series):
        for idx in frame.index:
            if idx in selected_set:
                continue
            selected_set.add(int(idx))
            selected_indices.append(int(idx))
            if len(selected_indices) >= int(n_series):
                break

    return frame.loc[selected_indices].copy()


def preprocess_m5(
    zip_path: str | Path,
    processed_dir: str | Path | None = None,
    *,
    n_series: int,
    max_train_days: int,
    max_test_days: int,
    min_active_days: int,
    selection_policy: str = "stratified_revenue_cv_category_store",
    max_category_share: float = 0.55,
    revenue_quantile_bins: int = 3,
    cv_quantile_bins: int = 3,
    stability_filter: bool = True,
    min_calibration_mean_ratio: float = 0.25,
    max_calibration_mean_ratio: float = 4.0,
    max_calibration_zero_rate_shift: float = 0.60,
) -> Path:
    zip_file = Path(zip_path)
    output = Path(processed_dir) if processed_dir is not None else case_processed_dir("m5")
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file) as archive:
        names = {Path(name).name for name in archive.namelist()}
    sales_member = "sales_train_evaluation.csv" if "sales_train_evaluation.csv" in names else "sales_train_validation.csv"
    sales = _read_member(zip_file, sales_member)
    prices = _read_member(zip_file, "sell_prices.csv")

    d_cols = [column for column in sales.columns if column.startswith("d_")]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    demand = sales.loc[:, d_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n_test = min(int(max_test_days), max(1, demand.shape[1] // 3))
    train_end = demand.shape[1] - n_test
    train_start = max(0, train_end - int(max_train_days))
    demand_calibration = demand.iloc[:, train_start:train_end]
    if demand_calibration.empty:
        raise ValueError("M5 split produced empty calibration data before series selection")
    split_mid = max(1, demand_calibration.shape[1] // 2)
    calibration_early = demand_calibration.iloc[:, :split_mid]
    calibration_late = demand_calibration.iloc[:, split_mid:]
    if calibration_late.empty:
        calibration_late = calibration_early
    active_days = (demand_calibration > 0.0).sum(axis=1)

    price_by_item_store = (
        prices.groupby(["item_id", "store_id"], sort=False)["sell_price"].median().rename("median_price")
    )
    metadata = sales.loc[:, id_cols].copy()
    metadata = metadata.join(price_by_item_store, on=["item_id", "store_id"])
    metadata["median_price"] = metadata["median_price"].fillna(
        metadata.groupby(["store_id", "cat_id"])["median_price"].transform("median")
    )
    metadata["median_price"] = metadata["median_price"].fillna(
        metadata.groupby("cat_id")["median_price"].transform("median")
    )
    metadata["median_price"] = metadata["median_price"].fillna(metadata["median_price"].median())
    metadata["mean_demand"] = demand_calibration.mean(axis=1)
    metadata["std_demand"] = demand_calibration.std(axis=1, ddof=0)
    metadata["active_days"] = active_days
    metadata["revenue_mean"] = metadata["mean_demand"] * metadata["median_price"]
    metadata["coefficient_of_variation"] = np.divide(
        metadata["std_demand"],
        metadata["mean_demand"],
        out=np.zeros(len(metadata), dtype=float),
        where=metadata["mean_demand"].to_numpy(dtype=float) > 0.0,
    )
    metadata["calibration_early_mean"] = calibration_early.mean(axis=1)
    metadata["calibration_late_mean"] = calibration_late.mean(axis=1)
    metadata["calibration_mean_ratio_late_over_early"] = np.divide(
        metadata["calibration_late_mean"],
        metadata["calibration_early_mean"],
        out=np.full(len(metadata), np.nan, dtype=float),
        where=metadata["calibration_early_mean"].to_numpy(dtype=float) > 0.0,
    )
    metadata["calibration_early_zero_rate"] = (calibration_early <= 0.0).mean(axis=1)
    metadata["calibration_late_zero_rate"] = (calibration_late <= 0.0).mean(axis=1)
    metadata["calibration_zero_rate_shift"] = (
        metadata["calibration_late_zero_rate"] - metadata["calibration_early_zero_rate"]
    ).abs()
    early_std = calibration_early.std(axis=1, ddof=0)
    late_std = calibration_late.std(axis=1, ddof=0)
    metadata["calibration_early_cv"] = np.divide(
        early_std,
        metadata["calibration_early_mean"],
        out=np.zeros(len(metadata), dtype=float),
        where=metadata["calibration_early_mean"].to_numpy(dtype=float) > 0.0,
    )
    metadata["calibration_late_cv"] = np.divide(
        late_std,
        metadata["calibration_late_mean"],
        out=np.zeros(len(metadata), dtype=float),
        where=metadata["calibration_late_mean"].to_numpy(dtype=float) > 0.0,
    )
    metadata["calibration_cv_shift"] = (
        metadata["calibration_late_cv"] - metadata["calibration_early_cv"]
    ).abs()

    eligible = metadata.loc[metadata["active_days"] >= int(min_active_days)].copy()
    if len(eligible) < int(n_series):
        raise ValueError(
            f"M5 preprocessing found only {len(eligible)} eligible series; requested n_series={n_series}"
        )
    stability_filter_applied = False
    if bool(stability_filter):
        ratio = eligible["calibration_mean_ratio_late_over_early"]
        stable_mask = (
            ratio.between(float(min_calibration_mean_ratio), float(max_calibration_mean_ratio), inclusive="both")
            & (eligible["calibration_zero_rate_shift"] <= float(max_calibration_zero_rate_shift))
        )
        stable_eligible = eligible.loc[stable_mask].copy()
        if len(stable_eligible) >= int(n_series):
            eligible = stable_eligible
            stability_filter_applied = True
    policy = str(selection_policy)
    if policy == "calibration_revenue_top":
        selected = eligible.sort_values(["revenue_mean", "id"], ascending=[False, True]).head(int(n_series))
    elif policy == "stratified_revenue_cv_category_store":
        selected = _select_stratified(
            eligible,
            n_series=int(n_series),
            max_category_share=float(max_category_share),
            revenue_quantile_bins=int(revenue_quantile_bins),
            cv_quantile_bins=int(cv_quantile_bins),
        )
    else:
        raise ValueError(f"unknown M5 selection_policy={selection_policy!r}")
    selected_indices = selected.index.to_numpy()
    selected_demand = demand.iloc[selected_indices].T
    selected_demand.index = d_cols
    selected_demand.columns = selected["id"].tolist()

    train = selected_demand.iloc[train_start:train_end].copy()
    test = selected_demand.iloc[train_end:].copy()
    if train.empty or test.empty:
        raise ValueError("M5 split produced empty calibration or held-out data")

    selected = selected.reset_index(drop=True)
    selected["selection_policy"] = policy
    selected["selection_uses_heldout"] = False
    selected["split_policy"] = "time_ordered_calibration_then_heldout"
    selected["stability_filter_applied"] = bool(stability_filter_applied)
    selected["stability_filter_requested"] = bool(stability_filter)
    diagnostics_rows = []
    for source_name, frame in (("eligible", eligible), ("selected", selected)):
        for group_col in ("cat_id", "state_id", "store_id"):
            counts = frame[group_col].value_counts(normalize=False)
            shares = frame[group_col].value_counts(normalize=True)
            for value, count in counts.items():
                diagnostics_rows.append(
                    {
                        "source": source_name,
                        "group": group_col,
                        "value": value,
                        "count": int(count),
                        "share": float(shares[value]),
                        "selection_policy": policy,
                    }
                )
    train.to_csv(output / "demand_train.csv")
    test.to_csv(output / "demand_test.csv")
    selected.to_csv(output / "series_metadata.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(output / "selection_diagnostics.csv", index=False)
    stability_columns = [
        "id",
        "item_id",
        "store_id",
        "cat_id",
        "state_id",
        "mean_demand",
        "std_demand",
        "coefficient_of_variation",
        "calibration_early_mean",
        "calibration_late_mean",
        "calibration_mean_ratio_late_over_early",
        "calibration_early_zero_rate",
        "calibration_late_zero_rate",
        "calibration_zero_rate_shift",
        "calibration_early_cv",
        "calibration_late_cv",
        "calibration_cv_shift",
        "selection_policy",
        "selection_uses_heldout",
        "stability_filter_requested",
        "stability_filter_applied",
    ]
    selected.loc[:, stability_columns].to_csv(output / "selection_stability.csv", index=False)
    split_metadata = {
        "case": "m5",
        "split_policy": "time_ordered_calibration_then_heldout",
        "selection_uses_heldout": False,
        "sales_member": sales_member,
        "n_total_days": int(demand.shape[1]),
        "n_calibration_days": int(train.shape[0]),
        "n_heldout_days": int(test.shape[0]),
        "calibration_start_column": str(train.index[0]),
        "calibration_end_column": str(train.index[-1]),
        "heldout_start_column": str(test.index[0]),
        "heldout_end_column": str(test.index[-1]),
        "train_start_position": int(train_start),
        "train_end_position_exclusive": int(train_end),
        "selection_policy": policy,
        "stability_filter_requested": bool(stability_filter),
        "stability_filter_applied": bool(stability_filter_applied),
        "min_calibration_mean_ratio": float(min_calibration_mean_ratio),
        "max_calibration_mean_ratio": float(max_calibration_mean_ratio),
        "max_calibration_zero_rate_shift": float(max_calibration_zero_rate_shift),
        "n_series": int(len(selected)),
    }
    (output / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    pd.DataFrame([split_metadata]).to_csv(output / "split_metadata.csv", index=False)
    return output
