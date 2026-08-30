from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from risk_budget_jccp.real_data.cases.french.preprocess import preprocess_french
from risk_budget_jccp.real_data.cases.m5.preprocess import preprocess_m5
from risk_budget_jccp.real_data.cases.power.preprocess import _risk_tier, _select_diverse_branches


def test_m5_stratified_selection_covers_categories(tmp_path: Path) -> None:
    raw = tmp_path / "m5.zip"
    ids = []
    rows = []
    prices = []
    categories = ["FOODS", "HOUSEHOLD", "HOBBIES"]
    for idx in range(18):
        cat = categories[idx % len(categories)]
        item = f"{cat}_{idx:03d}"
        store = f"S_{idx % 3}"
        ids.append(f"{item}_{store}_evaluation")
        rows.append(
            {
                "id": ids[-1],
                "item_id": item,
                "dept_id": f"{cat}_1",
                "cat_id": cat,
                "store_id": store,
                "state_id": ["CA", "TX", "WI"][idx % 3],
                **{f"d_{day}": float(5 + idx + (day % 4)) for day in range(1, 31)},
            }
        )
        prices.append({"item_id": item, "store_id": store, "wm_yr_wk": 1, "sell_price": 1.0 + idx / 10})
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("sales_train_evaluation.csv", pd.DataFrame(rows).to_csv(index=False))
        archive.writestr("sell_prices.csv", pd.DataFrame(prices).to_csv(index=False))
        archive.writestr("calendar.csv", "d,date\n")
    out = preprocess_m5(
        raw,
        tmp_path / "processed",
        n_series=9,
        max_train_days=20,
        max_test_days=5,
        min_active_days=5,
        max_category_share=0.45,
    )
    meta = pd.read_csv(out / "series_metadata.csv")
    assert meta["cat_id"].nunique() >= 3
    assert meta["cat_id"].value_counts(normalize=True).max() <= 0.45 + 1.0 / len(meta)
    assert (out / "selection_diagnostics.csv").is_file()


def test_french_first_after_split_uses_earliest_heldout_rows(tmp_path: Path) -> None:
    dates = pd.bdate_range("2018-12-20", periods=40)
    frame = pd.DataFrame(
        {
            "A": np.linspace(0.0, 3.9, len(dates)),
            "B": np.linspace(1.0, 4.9, len(dates)),
        },
        index=dates,
    )
    lines = [",A,B"]
    lines.extend(f"{idx:%Y%m%d},{row.A:.4f},{row.B:.4f}" for idx, row in frame.iterrows())
    raw = tmp_path / "french.csv"
    raw.write_text("\n".join(lines), encoding="latin1")
    out = preprocess_french(
        raw,
        tmp_path / "processed",
        start_date="2018-12-20",
        train_end_date="2018-12-31",
        test_start_date="2019-01-01",
        test_end_date="2019-02-28",
        max_assets=2,
        max_train_scenarios=5,
        max_test_scenarios=4,
        heldout_policy="first_after_split",
    )
    test = pd.read_csv(out / "returns_test.csv", index_col=0, parse_dates=True)
    assert test.index[0] == pd.Timestamp("2019-01-01")
    assert len(test) == 4


def test_power_tier_helpers_exclude_above_ste_and_keep_diversity() -> None:
    abs_flow = np.array([50.0, 95.0, 115.0, 135.0, 180.0])
    cont = np.full(5, 100.0)
    lte = np.full(5, 120.0)
    ste = np.full(5, 150.0)
    assert _risk_tier(abs_flow, cont, lte, ste, 0.02, 0.80).tolist() == [
        "normal",
        "lte",
        "lte",
        "ste",
        "above_ste",
    ]
    candidates = pd.DataFrame(
        {
            "branch_pos": [0, 1, 2, 3, 4],
            "branch_id": ["n", "near", "lte", "ste", "bad"],
            "selection_tier": ["normal", "near_cont", "lte", "ste", "above_ste"],
            "selection_score": [1.0, 2.0, 3.0, 4.0, 10.0],
            "deterministic_infeasible": [False, False, False, False, True],
        }
    )
    selected = _select_diverse_branches(candidates, 4)
    assert 4 not in selected
    selected_tiers = set(candidates.loc[candidates["branch_pos"].isin(selected), "selection_tier"])
    assert {"normal", "near_cont", "lte", "ste"}.issubset(selected_tiers)


def test_power_seeded_branch_selection_is_reproducible_and_randomized() -> None:
    candidates = pd.DataFrame(
        {
            "branch_pos": list(range(12)),
            "branch_id": [f"L{i}" for i in range(12)],
            "selection_tier": ["normal", "near_cont", "lte", "ste"] * 3,
            "selection_score": np.linspace(0.0, 1.0, 12),
            "deterministic_infeasible": [False] * 12,
        }
    )
    first = _select_diverse_branches(candidates, 6, random_seed=123)
    second = _select_diverse_branches(candidates, 6, random_seed=123)
    different = _select_diverse_branches(candidates, 6, random_seed=456)
    assert first.tolist() == second.tolist()
    assert first.tolist() != different.tolist()
