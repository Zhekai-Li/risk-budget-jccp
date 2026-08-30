from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from risk_budget_jccp.real_data.common.paths import case_processed_dir
from risk_budget_jccp.data.rts_gmlc import build_rts_gmlc_power_dispatch_instance, load_rts_gmlc_source_data


def _normalize_column(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {_normalize_column(c): c for c in frame.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_column(candidate))
        if match is not None:
            return match
    raise KeyError(f"missing required column; tried {candidates}")


def _load_distribution(bus: pd.DataFrame, bus_ids: tuple[str, ...], load_regions: tuple[str, ...]) -> np.ndarray:
    bus_col = _find_column(bus, ("Bus ID", "Bus"))
    area_col = _find_column(bus, ("Area",))
    load_col = _find_column(bus, ("MW Load", "Pd", "PD"))
    bus_index = {str(bus_id): idx for idx, bus_id in enumerate(bus_ids)}
    weights = np.zeros((len(load_regions), len(bus_ids)), dtype=float)
    for region_index, region in enumerate(load_regions):
        mask = bus[area_col].astype(str) == str(region)
        if not mask.any():
            continue
        load = pd.to_numeric(bus.loc[mask, load_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        denom = float(load.sum())
        if denom <= 0.0:
            load = np.ones_like(load)
            denom = float(load.sum())
        for bus_id, value in zip(bus.loc[mask, bus_col].astype(str), load, strict=False):
            weights[region_index, bus_index[bus_id]] = float(value / denom)
    return weights


def _wind_distribution(gen: pd.DataFrame, bus_ids: tuple[str, ...], wind_regions: tuple[str, ...]) -> np.ndarray:
    gen_col = _find_column(gen, ("GEN UID", "Gen UID", "UID"))
    bus_col = _find_column(gen, ("Bus ID", "Bus", "GEN_BUS"))
    bus_index = {str(bus_id): idx for idx, bus_id in enumerate(bus_ids)}
    gen_to_bus = dict(zip(gen[gen_col].astype(str), gen[bus_col].astype(str), strict=False))
    weights = np.zeros((len(wind_regions), len(bus_ids)), dtype=float)
    for row, gen_id in enumerate(wind_regions):
        if str(gen_id) in gen_to_bus:
            weights[row, bus_index[gen_to_bus[str(gen_id)]]] = 1.0
    return weights


def _dispatchable_generators(instance, gen: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    gen_col = _find_column(gen, ("GEN UID", "Gen UID", "UID"))
    fuel_col_raw = _find_column(gen, ("Fuel",))
    thermal_fuels = {"Coal", "Oil", "NG", "Nuclear"}
    gen_fuel = dict(zip(gen[gen_col].astype(str), gen[fuel_col_raw].astype(str), strict=False))
    dispatchable = np.array(
        [
            idx
            for idx, gid in enumerate(instance.generator_ids)
            if gid not in set(instance.wind_regions) and gen_fuel.get(str(gid), "") in thermal_fuels
        ],
        dtype=int,
    )
    gen_frame = gen.copy()
    gen_frame["_gid"] = gen_frame[gen_col].astype(str)
    gen_frame = gen_frame.set_index("_gid", drop=False)
    fuel_col = _find_column(gen_frame, ("Fuel Price $/MMBTU", "Fuel Price"))
    vom_col = _find_column(gen_frame, ("VOM", "Variable O&M", "Variable O&M $/MWh"))
    hr_cols = [
        _find_column(gen_frame, ("HR_incr_3", "HR Incr 3", "HR_incr_4")),
        _find_column(gen_frame, ("HR_incr_2", "HR Incr 2")),
        _find_column(gen_frame, ("HR_incr_1", "HR Incr 1")),
        _find_column(gen_frame, ("HR_avg_0", "HR Avg 0")),
    ]
    costs = []
    cost_rows: list[dict[str, object]] = []
    for gid in np.asarray(instance.generator_ids, dtype=object)[dispatchable]:
        row = gen_frame.loc[str(gid)]
        fuel_price = float(pd.to_numeric(row[fuel_col], errors="coerce"))
        vom = float(pd.to_numeric(row[vom_col], errors="coerce"))
        selected_hr = np.nan
        selected_hr_col = ""
        for hr_col in hr_cols:
            value = float(pd.to_numeric(row[hr_col], errors="coerce"))
            if np.isfinite(value) and value > 0.0:
                selected_hr = value
                selected_hr_col = hr_col
                break
        if np.isfinite(fuel_price) and np.isfinite(vom) and np.isfinite(selected_hr):
            cost = fuel_price * selected_hr / 1000.0 + vom
            source = f"{fuel_col}*{selected_hr_col}/1000+{vom_col}"
        else:
            cost = 25.0
            source = "fallback_25_missing_heat_rate_or_fuel_price"
        costs.append(float(cost))
        cost_rows.append(
            {
                "generator_id": str(gid),
                "fuel_price_per_mmbtu": fuel_price,
                "heat_rate_btu_per_kwh": selected_hr,
                "heat_rate_column": selected_hr_col,
                "vom": vom,
                "cost_source": source,
                "cost": float(cost),
            }
        )
    return dispatchable, np.asarray(costs, dtype=float), pd.DataFrame(cost_rows)


def _branch_rating_vectors(branch: pd.DataFrame, branch_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uid_col = _find_column(branch, ("UID", "Line", "Branch ID"))
    cont_col = _find_column(branch, ("Cont Rating", "Continuous Rating", "RATE_A", "Rate A"))
    lte_col = _find_column(branch, ("LTE Rating", "Long Term Emergency", "RATE_B", "Rate B"))
    ste_col = _find_column(branch, ("STE Rating", "Short Term Emergency", "RATE_C", "Rate C"))
    indexed = branch.copy()
    indexed["_uid"] = indexed[uid_col].astype(str)
    indexed = indexed.set_index("_uid", drop=False)
    cont: list[float] = []
    lte: list[float] = []
    ste: list[float] = []
    for branch_id in branch_ids:
        row = indexed.loc[str(branch_id)]
        cont.append(float(pd.to_numeric(row[cont_col], errors="coerce")))
        lte.append(float(pd.to_numeric(row[lte_col], errors="coerce")))
        ste.append(float(pd.to_numeric(row[ste_col], errors="coerce")))
    cont_arr = np.asarray(cont, dtype=float)
    lte_arr = np.asarray(lte, dtype=float)
    ste_arr = np.asarray(ste, dtype=float)
    if np.any(~np.isfinite(cont_arr)) or np.any(cont_arr <= 0.0):
        raise ValueError("RTS-GMLC continuous ratings must be finite and positive")
    lte_arr = np.where(np.isfinite(lte_arr) & (lte_arr > 0.0), lte_arr, cont_arr)
    ste_arr = np.where(np.isfinite(ste_arr) & (ste_arr > 0.0), ste_arr, lte_arr)
    lte_arr = np.maximum(lte_arr, cont_arr)
    ste_arr = np.maximum(ste_arr, lte_arr)
    return cont_arr, lte_arr, ste_arr


def _representative_indices(net_load: np.ndarray, renewable: np.ndarray, n_snapshots: int) -> np.ndarray:
    ramp = np.abs(np.diff(renewable, prepend=renewable[0]))
    candidates = [int(np.argmax(net_load)), int(np.argmax(renewable)), int(np.argmax(ramp))]
    if int(n_snapshots) >= 4:
        candidates.append(int(np.argsort(np.abs(net_load - np.median(net_load)))[0]))
    for idx in np.argsort(net_load)[::-1]:
        candidates.append(int(idx))
        if len(dict.fromkeys(candidates)) >= n_snapshots:
            break
    return np.array(list(dict.fromkeys(candidates))[:n_snapshots], dtype=int)


def _economic_dispatch(
    *,
    demand: float,
    costs: np.ndarray,
    pmin: np.ndarray,
    pmax: np.ndarray,
) -> np.ndarray:
    objective = np.concatenate([costs, np.array([1.0e6, 1.0e6], dtype=float)])
    result = linprog(
        objective,
        A_eq=np.hstack([np.ones((1, len(costs)), dtype=float), np.array([[1.0, -1.0]])]),
        b_eq=np.array([float(demand)], dtype=float),
        bounds=list(zip(pmin, pmax, strict=True)) + [(0.0, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"forecast economic dispatch failed: {result.message}")
    return np.asarray(result.x[: len(costs)], dtype=float)


def _risk_tier(
    abs_flow: np.ndarray,
    cont: np.ndarray,
    lte: np.ndarray,
    ste: np.ndarray,
    margin: float,
    near_cont_threshold: float = 0.80,
) -> np.ndarray:
    safe = 1.0 - float(margin)
    tier = np.full(abs_flow.shape, "above_ste", dtype=object)
    tier[abs_flow <= safe * ste] = "ste"
    tier[abs_flow <= safe * lte] = "lte"
    tier[abs_flow <= float(near_cont_threshold) * cont] = "normal"
    return tier


def _risk_limit_from_tier(tier: str, cont: float, lte: float, ste: float) -> float:
    if tier == "normal":
        return float(cont)
    if tier == "lte":
        return float(lte)
    return float(ste)


def _certificate_margin_frame(
    candidates: pd.DataFrame,
    *,
    selected_indices: np.ndarray,
    nominal_flow_all: np.ndarray,
    train_flow: np.ndarray,
    cont_all: np.ndarray,
    lte_all: np.ndarray,
    ste_all: np.ndarray,
    alpha_per_constraint: float,
    tier_margin: float,
    near_cont_threshold: float,
) -> pd.DataFrame:
    bernstein_q = float(np.sqrt(2.0 * np.log(1.0 / max(float(alpha_per_constraint), 1.0e-12))))
    cantelli_q = float(np.sqrt((1.0 - float(alpha_per_constraint)) / max(float(alpha_per_constraint), 1.0e-12)))
    rows: list[dict[str, float | bool | str]] = []
    for branch_pos in candidates["branch_pos"].to_numpy(dtype=int):
        margins: list[float] = []
        headrooms: list[float] = []
        empirical_requirements: list[float] = []
        bernstein_requirements: list[float] = []
        cantelli_requirements: list[float] = []
        tiers: list[str] = []
        deterministic_infeasible = False
        for snapshot_index in selected_indices:
            for sign in (1.0, -1.0):
                signed_base = sign * float(nominal_flow_all[int(snapshot_index), branch_pos])
                abs_base = abs(signed_base)
                tier = str(
                    _risk_tier(
                        np.asarray([abs_base], dtype=float),
                        np.asarray([cont_all[branch_pos]], dtype=float),
                        np.asarray([lte_all[branch_pos]], dtype=float),
                        np.asarray([ste_all[branch_pos]], dtype=float),
                        float(tier_margin),
                        float(near_cont_threshold),
                    )[0]
                )
                tiers.append(tier)
                if tier == "above_ste":
                    deterministic_infeasible = True
                    margins.append(-np.inf)
                    continue
                limit = _risk_limit_from_tier(
                    tier,
                    float(cont_all[branch_pos]),
                    float(lte_all[branch_pos]),
                    float(ste_all[branch_pos]),
                )
                headroom = float(limit - abs_base)
                signed_residual = sign * train_flow[:, branch_pos]
                empirical_required = float(np.max(signed_residual))
                bernstein_required = float(np.mean(signed_residual) + bernstein_q * np.std(signed_residual, ddof=0))
                cantelli_required = float(np.mean(signed_residual) + cantelli_q * np.std(signed_residual, ddof=0))
                required = max(empirical_required, bernstein_required, cantelli_required)
                headrooms.append(headroom)
                empirical_requirements.append(empirical_required)
                bernstein_requirements.append(bernstein_required)
                cantelli_requirements.append(cantelli_required)
                margins.append(headroom - required)
        finite_margins = [value for value in margins if np.isfinite(value)]
        rows.append(
            {
                "branch_pos": int(branch_pos),
                "certificate_alpha_per_constraint": float(alpha_per_constraint),
                "certificate_headroom_margin": float(min(finite_margins)) if finite_margins and not deterministic_infeasible else -np.inf,
                "min_certificate_headroom": float(min(headrooms)) if headrooms else np.nan,
                "max_empirical_required_headroom": float(max(empirical_requirements)) if empirical_requirements else np.nan,
                "max_bernstein_required_headroom": float(max(bernstein_requirements)) if bernstein_requirements else np.nan,
                "max_cantelli_required_headroom": float(max(cantelli_requirements)) if cantelli_requirements else np.nan,
                "certificate_feasible_candidate": bool((not deterministic_infeasible) and finite_margins and min(finite_margins) >= 0.0),
                "certificate_screen_tiers": ",".join(sorted(set(tiers))),
            }
        )
    return pd.DataFrame(rows)


def _select_diverse_branches(
    candidates: pd.DataFrame,
    max_branches: int,
    *,
    require_certificate_feasible: bool = False,
    random_seed: int = 20260525,
) -> np.ndarray:
    mask = ~candidates["deterministic_infeasible"]
    if require_certificate_feasible:
        mask &= candidates.get("certificate_feasible_candidate", False)
    eligible = candidates.loc[mask].copy()
    if eligible.empty:
        return np.asarray([], dtype=int)
    quotas = {
        "normal": 0.35,
        "near_cont": 0.30,
        "lte": 0.25,
        "ste": 0.10,
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    rng = np.random.default_rng(int(random_seed))
    for tier, share in quotas.items():
        quota = max(1, int(round(float(max_branches) * share)))
        part = eligible.loc[eligible["selection_tier"] == tier]
        order = rng.permutation(part.index.to_numpy()) if not part.empty else []
        for idx in order:
            branch_pos = int(eligible.loc[idx, "branch_pos"])
            if branch_pos in selected_set:
                continue
            selected.append(branch_pos)
            selected_set.add(branch_pos)
            if len([x for x in selected if candidates.loc[candidates["branch_pos"] == x, "selection_tier"].iloc[0] == tier]) >= quota:
                break
            if len(selected) >= int(max_branches):
                break
        if len(selected) >= int(max_branches):
            break
    fallback_order = rng.permutation(eligible.index.to_numpy())
    for idx in fallback_order:
        row = eligible.loc[idx]
        branch_pos = int(row["branch_pos"])
        if branch_pos in selected_set:
            continue
        selected.append(branch_pos)
        selected_set.add(branch_pos)
        if len(selected) >= int(max_branches):
            break
    return np.asarray(selected[: int(max_branches)], dtype=int)


def _risk_limit_vector(gamma: float, cont: np.ndarray, emergency: np.ndarray) -> np.ndarray:
    g = float(np.clip(gamma, 0.0, 1.0))
    return np.asarray(cont, dtype=float) + g * (np.asarray(emergency, dtype=float) - np.asarray(cont, dtype=float))


def _baseline_joint_violation(
    *,
    branch_order: np.ndarray,
    selected_indices: np.ndarray,
    nominal_flow_all: np.ndarray,
    train_flow: np.ndarray,
    risk_limits: np.ndarray,
) -> float:
    if len(branch_order) == 0:
        return 1.0
    violations: list[np.ndarray] = []
    for snapshot_index in selected_indices:
        for branch_pos in branch_order:
            limit = float(risk_limits[int(branch_pos)])
            base = float(nominal_flow_all[int(snapshot_index), int(branch_pos)])
            violations.append(base + train_flow[:, int(branch_pos)] - limit)
            violations.append(-base - train_flow[:, int(branch_pos)] - limit)
    y = np.vstack(violations).T
    return float((np.max(y, axis=1) > 1.0e-9).mean())


def _select_objective_power_branches(
    candidates: pd.DataFrame,
    *,
    requested_max_branches: int,
    selected_indices: np.ndarray,
    nominal_flow_all: np.ndarray,
    train_flow: np.ndarray,
    cont_all: np.ndarray,
    emergency_all: np.ndarray,
    gamma_grid: list[float],
    target_joint_violation: float,
    joint_violation_band: tuple[float, float],
    random_seed: int,
    exclude_above_ste: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object], pd.DataFrame]:
    requested = max(1, int(requested_max_branches))
    diagnostics: list[dict[str, object]] = []
    best: tuple[float, np.ndarray, pd.DataFrame, dict[str, object]] | None = None
    low, high = float(joint_violation_band[0]), float(joint_violation_band[1])
    for gamma in [float(value) for value in gamma_grid]:
        limits = _risk_limit_vector(gamma, cont_all, emergency_all)
        screened = candidates.copy()
        screened["risk_limit_gamma"] = gamma
        screened["tuned_risk_limit"] = limits
        screened["base_utilization_vs_tuned_limit"] = screened["max_abs_base_flow"].to_numpy(dtype=float) / limits
        screened["deterministic_infeasible"] = screened["max_abs_base_flow"].to_numpy(dtype=float) > limits + 1.0e-9
        if exclude_above_ste:
            screened["deterministic_infeasible"] |= screened["risk_tier"].astype(str).eq("above_ste")
        order = _select_diverse_branches(
            screened,
            requested,
            random_seed=int(random_seed),
            require_certificate_feasible=False,
        )
        calibration_joint = _baseline_joint_violation(
            branch_order=order,
            selected_indices=selected_indices,
            nominal_flow_all=nominal_flow_all,
            train_flow=train_flow,
            risk_limits=limits,
        )
        feasible_count = int((~screened["deterministic_infeasible"]).sum())
        in_band = bool(low <= calibration_joint <= high and len(order) == requested)
        penalty = 0.0 if in_band else min(abs(calibration_joint - low), abs(calibration_joint - high))
        score = penalty + abs(calibration_joint - float(target_joint_violation))
        row = {
            "risk_limit_gamma": gamma,
            "requested_max_branches": requested,
            "selected_branch_count": int(len(order)),
            "candidate_count": feasible_count,
            "calibration_baseline_joint_violation": calibration_joint,
            "target_calibration_joint_violation": float(target_joint_violation),
            "calibration_joint_violation_band_low": low,
            "calibration_joint_violation_band_high": high,
            "in_target_band": in_band,
            "selection_score": float(score),
            "random_seed": int(random_seed),
        }
        diagnostics.append(row)
        if len(order) == requested and (best is None or score < best[0]):
            best = (float(score), order, screened, row)
    if best is None:
        raise ValueError("Power preprocessing found no objectively eligible branches under the gamma grid")
    _, branch_order, selected_candidates, selected_diag = best
    selected_candidates["selected_by_seeded_stratified_sampling"] = selected_candidates["branch_pos"].isin(branch_order)
    selected_diag = dict(selected_diag)
    selected_diag["selection_rule"] = "calibration_tuned_gamma_seeded_stratified_random_sampling"
    selected_diag["selection_reduced_for_certificate_feasibility"] = False
    return branch_order, selected_candidates, selected_diag, pd.DataFrame(diagnostics)


def _select_certificate_feasible_branches(
    candidates: pd.DataFrame,
    *,
    requested_max_branches: int,
    alpha: float,
    n_snapshots: int,
    selected_indices: np.ndarray,
    nominal_flow_all: np.ndarray,
    train_flow: np.ndarray,
    cont_all: np.ndarray,
    lte_all: np.ndarray,
    ste_all: np.ndarray,
    tier_margin: float,
    near_cont_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    requested = max(1, int(requested_max_branches))
    best_candidates = candidates.copy()
    diagnostics: dict[str, object] = {
        "requested_max_branches": requested,
        "selected_branch_count": 0,
        "selection_reduced_for_certificate_feasibility": True,
        "certificate_screen": "empirical_max_and_equal_bernstein_cantelli_headroom",
    }
    for branch_count in range(requested, 0, -1):
        m = int(n_snapshots) * 2 * branch_count
        alpha_per_constraint = float(alpha) / float(m)
        margin = _certificate_margin_frame(
            candidates,
            selected_indices=selected_indices,
            nominal_flow_all=nominal_flow_all,
            train_flow=train_flow,
            cont_all=cont_all,
            lte_all=lte_all,
            ste_all=ste_all,
            alpha_per_constraint=alpha_per_constraint,
            tier_margin=tier_margin,
            near_cont_threshold=near_cont_threshold,
        )
        screened = candidates.drop(
            columns=[col for col in margin.columns if col != "branch_pos" and col in candidates.columns],
            errors="ignore",
        ).merge(margin, on="branch_pos", how="left")
        order = _select_diverse_branches(screened, branch_count, require_certificate_feasible=True)
        best_candidates = screened
        if len(order) == branch_count:
            diagnostics.update(
                {
                    "selected_branch_count": int(branch_count),
                    "selection_reduced_for_certificate_feasibility": bool(branch_count < requested),
                    "certificate_alpha_per_constraint": alpha_per_constraint,
                    "certificate_feasible_candidate_count": int(screened["certificate_feasible_candidate"].sum()),
                }
            )
            return order, screened, diagnostics
    fallback = best_candidates.loc[~best_candidates["deterministic_infeasible"]].sort_values(
        ["certificate_headroom_margin", "selection_score", "branch_id"],
        ascending=[False, False, True],
    )
    if fallback.empty:
        raise ValueError("Power preprocessing found no branches below STE for formal constraints")
    diagnostics.update(
        {
            "selected_branch_count": 1,
            "certificate_alpha_per_constraint": float(alpha) / float(2 * int(n_snapshots)),
            "certificate_feasible_candidate_count": 0,
            "warning": "no branch passed the conservative certificate headroom screen; selected the best diagnostic branch",
        }
    )
    return fallback["branch_pos"].head(1).to_numpy(dtype=int), best_candidates, diagnostics


def preprocess_power(
    rts_repo_dir: str | Path,
    processed_dir: str | Path | None = None,
    *,
    alpha: float,
    n_snapshots: int,
    max_branches: int,
    max_train_scenarios: int,
    max_test_scenarios: int,
    slack_bus: str | int = "101",
    primary_rating: str = "lte",
    snapshot_selection_policy: str = "forecast_representative_diverse",
    branch_selection_policy: str = "stratified_tier_volatility",
    risk_tier_policy: str = "next_available_rating",
    risk_limit_gamma_grid: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    target_calibration_joint_violation: float = 0.15,
    calibration_joint_violation_band: list[float] | tuple[float, float] = (0.08, 0.35),
    random_seed: int = 20260525,
    tier_margin: float = 0.02,
    near_cont_threshold: float = 0.80,
    exclude_above_ste: bool = True,
) -> Path:
    repo = Path(rts_repo_dir)
    output = Path(processed_dir) if processed_dir is not None else case_processed_dir("power")
    output.mkdir(parents=True, exist_ok=True)
    instance = build_rts_gmlc_power_dispatch_instance(repo / "RTS_Data", slack_bus=slack_bus, train_fraction=0.7)
    source = load_rts_gmlc_source_data(repo / "RTS_Data")

    load_dist = _load_distribution(source["bus"], instance.bus_ids, instance.load_regions)
    wind_dist = _wind_distribution(source["gen"], instance.bus_ids, instance.wind_regions)
    load_forecast_bus = instance.load_forecast @ load_dist
    load_actual_bus = instance.load_actual @ load_dist
    wind_forecast_bus = instance.wind_forecast @ wind_dist
    wind_actual_bus = instance.wind_actual @ wind_dist

    n_train = min(instance.load_error_train.shape[0], instance.wind_error_train.shape[0], int(max_train_scenarios))
    train_error_bus = instance.wind_error_train[-n_train:] @ wind_dist - instance.load_error_train[-n_train:] @ load_dist
    train_flow = train_error_bus @ instance.ptdf.T

    n_test = min(instance.load_error_test.shape[0], instance.wind_error_test.shape[0], int(max_test_scenarios))
    test_error_bus = instance.wind_error_test[-n_test:] @ wind_dist - instance.load_error_test[-n_test:] @ load_dist
    test_flow = test_error_bus @ instance.ptdf.T
    test_start = load_forecast_bus.shape[0] - n_test
    test_rows = np.arange(test_start, load_forecast_bus.shape[0])

    forecast_net_bus = wind_forecast_bus[test_rows] - load_forecast_bus[test_rows]
    actual_net_bus = wind_actual_bus[test_rows] - load_actual_bus[test_rows]
    forecast_flow_offset_all = forecast_net_bus @ instance.ptdf.T
    actual_flow_offset_all = actual_net_bus @ instance.ptdf.T

    dispatchable, costs, cost_metadata = _dispatchable_generators(instance, source["gen"])
    raw_pmin = instance.generator_pmin_mw[dispatchable]
    pmin = np.zeros(len(dispatchable), dtype=float)
    pmax = instance.generator_pmax_mw[dispatchable]
    gen_bus = np.zeros((len(instance.bus_ids), len(dispatchable)), dtype=float)
    for col, gen_idx in enumerate(dispatchable):
        gen_bus[int(instance.generator_bus[gen_idx]), col] = 1.0
    full_dispatch_flow_matrix = instance.ptdf @ gen_bus

    selected_net_load = load_forecast_bus[test_rows].sum(axis=1) - wind_forecast_bus[test_rows].sum(axis=1)
    selected_renewable = wind_forecast_bus[test_rows].sum(axis=1)
    if str(snapshot_selection_policy) != "forecast_representative_diverse":
        raise ValueError(f"unknown Power snapshot_selection_policy={snapshot_selection_policy!r}")
    selected_indices = _representative_indices(selected_net_load, selected_renewable, int(n_snapshots))

    cont_all, lte_all, ste_all = _branch_rating_vectors(source["branch"], instance.branch_ids)
    rating_lookup = {"cont": cont_all, "continuous": cont_all, "lte": lte_all, "ste": ste_all}
    rating_key = str(primary_rating).lower()
    if rating_key not in rating_lookup:
        raise ValueError(f"unknown primary_rating={primary_rating!r}; expected one of {sorted(rating_lookup)}")

    forecast_dispatch = np.vstack(
        [
            _economic_dispatch(
                demand=float(load_forecast_bus[row].sum() - wind_forecast_bus[row].sum()),
                costs=costs,
                pmin=pmin,
                pmax=pmax,
            )
            for row in test_rows
        ]
    )
    nominal_flow_all = forecast_dispatch @ full_dispatch_flow_matrix.T + forecast_flow_offset_all
    selected_nominal_all = nominal_flow_all[selected_indices, :]
    branch_abs_base = np.max(np.abs(selected_nominal_all), axis=0)
    base_util = branch_abs_base / cont_all
    volatility = train_flow.std(axis=0, ddof=0) / cont_all
    max_tier = _risk_tier(
        branch_abs_base,
        cont_all,
        lte_all,
        ste_all,
        float(tier_margin),
        float(near_cont_threshold),
    )
    selection_tier = np.where(
        max_tier == "normal",
        np.where(base_util >= 0.75, "near_cont", "normal"),
        max_tier,
    )
    candidates = pd.DataFrame(
        {
            "branch_pos": np.arange(len(instance.branch_ids), dtype=int),
            "branch_id": np.asarray(instance.branch_ids, dtype=object),
            "max_abs_base_flow": branch_abs_base,
            "base_utilization_vs_cont": base_util,
            "base_utilization_vs_lte": branch_abs_base / lte_all,
            "base_utilization_vs_ste": branch_abs_base / ste_all,
            "flow_std": train_flow.std(axis=0, ddof=0),
            "flow_std_normalized": volatility,
            "risk_tier": max_tier,
            "selection_tier": selection_tier,
            "deterministic_infeasible": max_tier == "above_ste",
            "selection_score": base_util + volatility,
        }
    )
    selection_diagnostics: dict[str, object] = {}
    gamma_diagnostics = pd.DataFrame()
    if str(branch_selection_policy) == "top_congestion_volatility":
        pool = candidates.loc[~candidates["deterministic_infeasible"] | ~bool(exclude_above_ste)]
        branch_order = pool.sort_values(["selection_score", "branch_id"], ascending=[False, True])["branch_pos"].head(int(max_branches)).to_numpy(dtype=int)
        m = int(n_snapshots) * 2 * max(1, len(branch_order))
        margin = _certificate_margin_frame(
            candidates,
            selected_indices=selected_indices,
            nominal_flow_all=nominal_flow_all,
            train_flow=train_flow,
            cont_all=cont_all,
            lte_all=lte_all,
            ste_all=ste_all,
            alpha_per_constraint=float(alpha) / float(m),
            tier_margin=float(tier_margin),
            near_cont_threshold=float(near_cont_threshold),
        )
        candidates = candidates.drop(
            columns=[col for col in margin.columns if col != "branch_pos" and col in candidates.columns],
            errors="ignore",
        ).merge(margin, on="branch_pos", how="left")
        selection_diagnostics = {
            "requested_max_branches": int(max_branches),
            "selected_branch_count": int(len(branch_order)),
            "selection_reduced_for_certificate_feasibility": False,
            "certificate_screen": "diagnostic_only_for_top_congestion_volatility",
            "certificate_alpha_per_constraint": float(alpha) / float(m),
            "certificate_feasible_candidate_count": int(candidates["certificate_feasible_candidate"].sum()),
            "random_seed": int(random_seed),
        }
    elif str(branch_selection_policy) == "stratified_tier_volatility":
        band = tuple(float(value) for value in calibration_joint_violation_band)
        if len(band) != 2:
            raise ValueError("calibration_joint_violation_band must contain two values")
        branch_order, candidates, selection_diagnostics, gamma_diagnostics = _select_objective_power_branches(
            candidates,
            requested_max_branches=int(max_branches),
            selected_indices=selected_indices,
            nominal_flow_all=nominal_flow_all,
            train_flow=train_flow,
            cont_all=cont_all,
            emergency_all=rating_lookup[rating_key],
            gamma_grid=[float(value) for value in risk_limit_gamma_grid],
            target_joint_violation=float(target_calibration_joint_violation),
            joint_violation_band=(band[0], band[1]),
            random_seed=int(random_seed),
            exclude_above_ste=bool(exclude_above_ste),
        )
        margin = _certificate_margin_frame(
            candidates,
            selected_indices=selected_indices,
            nominal_flow_all=nominal_flow_all,
            train_flow=train_flow,
            cont_all=cont_all,
            lte_all=lte_all,
            ste_all=ste_all,
            alpha_per_constraint=float(alpha) / float(int(n_snapshots) * 2 * max(1, len(branch_order))),
            tier_margin=float(tier_margin),
            near_cont_threshold=float(near_cont_threshold),
        )
        candidates = candidates.drop(
            columns=[col for col in margin.columns if col != "branch_pos" and col in candidates.columns],
            errors="ignore",
        ).merge(margin, on="branch_pos", how="left")
    else:
        raise ValueError(f"unknown Power branch_selection_policy={branch_selection_policy!r}")

    selected_train = train_flow[:, branch_order]
    selected_test = test_flow[:, branch_order]
    cont_limits = cont_all[branch_order]
    lte_limits = lte_all[branch_order]
    ste_limits = ste_all[branch_order]
    if str(risk_tier_policy) not in {"next_available_rating", "fixed_primary_rating"}:
        raise ValueError(f"unknown Power risk_tier_policy={risk_tier_policy!r}")
    selected_gamma = float(selection_diagnostics.get("risk_limit_gamma", 1.0))
    risk_limits = _risk_limit_vector(selected_gamma, cont_all[branch_order], rating_lookup[rating_key][branch_order])
    one_snapshot_side_train = np.concatenate([selected_train, -selected_train], axis=1)
    one_snapshot_side_test = np.concatenate([selected_test, -selected_test], axis=1)
    side_train = np.hstack([one_snapshot_side_train for _ in selected_indices])
    side_test = np.hstack([one_snapshot_side_test for _ in selected_indices])
    dispatch_flow_matrix = full_dispatch_flow_matrix[branch_order]

    selected_nominal_flow = nominal_flow_all[np.ix_(selected_indices, branch_order)]
    snapshot_labels = [f"snap_{idx + 1}" for idx in range(len(selected_indices))]
    branch_ids = np.asarray(instance.branch_ids, dtype=object)[branch_order]
    from_bus = np.asarray(instance.bus_ids, dtype=object)[instance.branch_from_bus[branch_order]]
    to_bus = np.asarray(instance.bus_ids, dtype=object)[instance.branch_to_bus[branch_order]]
    metadata_rows: list[dict[str, object]] = []
    for snapshot_pos, snapshot_label in enumerate(snapshot_labels):
        for direction in ("positive", "negative"):
            sign = 1.0 if direction == "positive" else -1.0
            for local_branch_pos, branch_id in enumerate(branch_ids):
                base_flow = sign * float(selected_nominal_flow[snapshot_pos, local_branch_pos])
                abs_base = abs(base_flow)
                tier = _risk_tier(
                    np.asarray([abs_base], dtype=float),
                    np.asarray([cont_limits[local_branch_pos]], dtype=float),
                    np.asarray([lte_limits[local_branch_pos]], dtype=float),
                    np.asarray([ste_limits[local_branch_pos]], dtype=float),
                    float(tier_margin),
                    float(near_cont_threshold),
                )[0]
                row_risk_limit = float(risk_limits[local_branch_pos])
                metadata_rows.append(
                    {
                        "snapshot": snapshot_label,
                        "snapshot_index": int(selected_indices[snapshot_pos]),
                        "branch_id": str(branch_id),
                        "from_bus": str(from_bus[local_branch_pos]),
                        "to_bus": str(to_bus[local_branch_pos]),
                        "direction": direction,
                        "thermal_limit": float(row_risk_limit),
                        "risk_limit": float(row_risk_limit),
                        "primary_rating": rating_key,
                        "risk_tier": str(tier),
                        "risk_tier_policy": str(risk_tier_policy),
                        "risk_limit_gamma": selected_gamma,
                        "risk_limit_multiplier": float(row_risk_limit / cont_limits[local_branch_pos]),
                        "tier_margin": float(tier_margin),
                        "near_cont_threshold": float(near_cont_threshold),
                        "cont_rating": float(cont_limits[local_branch_pos]),
                        "lte_rating": float(lte_limits[local_branch_pos]),
                        "ste_rating": float(ste_limits[local_branch_pos]),
                        "lte_multiplier": float(lte_limits[local_branch_pos] / cont_limits[local_branch_pos]),
                        "ste_multiplier": float(ste_limits[local_branch_pos] / cont_limits[local_branch_pos]),
                        "base_flow": base_flow,
                        "base_utilization": abs(base_flow) / float(cont_limits[local_branch_pos]),
                        "base_utilization_vs_cont": abs(base_flow) / float(cont_limits[local_branch_pos]),
                        "base_utilization_vs_lte": abs(base_flow) / float(lte_limits[local_branch_pos]),
                        "base_utilization_vs_ste": abs(base_flow) / float(ste_limits[local_branch_pos]),
                        "nominal_exceeds_cont": bool(abs(base_flow) > float(cont_limits[local_branch_pos])),
                        "nominal_exceeds_lte": bool(abs(base_flow) > float(lte_limits[local_branch_pos])),
                        "deterministic_infeasible": bool(str(tier) == "above_ste"),
                        "flow_std": float(selected_train[:, local_branch_pos].std(ddof=0)),
                        "renewable_ptdf_exposure": float(np.abs(dispatch_flow_matrix[local_branch_pos]).mean()),
                        "certificate_headroom_margin": float(
                            candidates.loc[candidates["branch_pos"] == branch_order[local_branch_pos], "certificate_headroom_margin"].iloc[0]
                        ),
                    }
                )

    side_metadata = pd.DataFrame(metadata_rows)
    gen_metadata = cost_metadata.copy()
    gen_metadata["raw_pmin"] = raw_pmin
    gen_metadata["pmin"] = pmin
    gen_metadata["pmax"] = pmax
    np.savez(
        output / "power_instance.npz",
        case_name=np.array("RTS-GMLC"),
        primary_rating=np.array(rating_key),
        risk_limit_gamma=np.array(selected_gamma),
        random_seed=np.array(int(random_seed)),
        branch_selection_policy=np.array(str(branch_selection_policy)),
        snapshot_selection_policy=np.array(str(snapshot_selection_policy)),
        risk_tier_policy=np.array(str(risk_tier_policy)),
        flow_residual_train=side_train,
        flow_residual_test=side_test,
        generator_bus_matrix=gen_bus,
        dispatch_flow_matrix=dispatch_flow_matrix,
        forecast_flow_offset=forecast_flow_offset_all[np.ix_(selected_indices, branch_order)],
        actual_flow_offset=actual_flow_offset_all[:, branch_order],
        load_forecast_total=load_forecast_bus[test_rows].sum(axis=1),
        renewable_forecast_total=wind_forecast_bus[test_rows].sum(axis=1),
        load_actual_total=load_actual_bus[test_rows].sum(axis=1),
        renewable_actual_total=wind_actual_bus[test_rows].sum(axis=1),
        selected_test_indices=selected_indices,
    )
    side_metadata.to_csv(output / "line_metadata.csv", index=False)
    candidates.to_csv(output / "candidate_constraints.csv", index=False)
    candidates.loc[candidates["branch_pos"].isin(branch_order)].to_csv(output / "selected_constraints.csv", index=False)
    candidates.loc[candidates["deterministic_infeasible"]].to_csv(output / "excluded_constraints.csv", index=False)
    pd.DataFrame([selection_diagnostics]).to_csv(output / "selection_diagnostics.csv", index=False)
    gamma_diagnostics.to_csv(output / "gamma_tuning_diagnostics.csv", index=False)
    gamma_diagnostics.to_csv(output / "gamma_selection_score.csv", index=False)
    constraint_audit = pd.DataFrame(
        [
            {
                "case": "power",
                "selection_uses_heldout": False,
                "split_policy": "time_ordered_calibration_then_heldout",
                "selected_branch_count": int(len(branch_order)),
                "scalar_constraint_count": int(side_metadata.shape[0]),
                "risk_limit_gamma": selected_gamma,
                "primary_rating": rating_key,
                "nominal_exceeds_selected_risk_limit": int(
                    (side_metadata["base_flow"].abs() > side_metadata["risk_limit"]).sum()
                ),
                "nominal_exceeds_cont": int(side_metadata["nominal_exceeds_cont"].sum()),
                "nominal_exceeds_lte": int(side_metadata["nominal_exceeds_lte"].sum()),
                "mean_base_utilization_vs_cont": float(side_metadata["base_utilization_vs_cont"].mean()),
                "max_base_utilization_vs_cont": float(side_metadata["base_utilization_vs_cont"].max()),
                "mean_base_utilization_vs_selected_limit": float(
                    (side_metadata["base_flow"].abs() / side_metadata["risk_limit"]).mean()
                ),
                "max_base_utilization_vs_selected_limit": float(
                    (side_metadata["base_flow"].abs() / side_metadata["risk_limit"]).max()
                ),
                "mean_flow_std": float(side_metadata["flow_std"].mean()),
                "max_flow_std": float(side_metadata["flow_std"].max()),
                "calibration_baseline_joint_violation": float(
                    selection_diagnostics.get("calibration_baseline_joint_violation", np.nan)
                ),
                "target_calibration_joint_violation": float(target_calibration_joint_violation),
                "calibration_joint_violation_band_low": float(calibration_joint_violation_band[0]),
                "calibration_joint_violation_band_high": float(calibration_joint_violation_band[1]),
                "random_seed": int(random_seed),
            }
        ]
    )
    constraint_audit.to_csv(output / "constraint_design_audit.csv", index=False)
    split_metadata = {
        "case": "power",
        "split_policy": "time_ordered_calibration_then_heldout_residuals",
        "selection_uses_heldout": False,
        "n_calibration_scenarios": int(n_train),
        "n_heldout_scenarios": int(n_test),
        "n_snapshots": int(len(selected_indices)),
        "selected_snapshot_indices_within_heldout_window": [int(value) for value in selected_indices],
        "heldout_window_start_row": int(test_start),
        "heldout_window_end_row_exclusive": int(load_forecast_bus.shape[0]),
        "selected_branch_count": int(len(branch_order)),
        "scalar_constraint_count": int(side_metadata.shape[0]),
        "branch_selection_policy": str(branch_selection_policy),
        "snapshot_selection_policy": str(snapshot_selection_policy),
        "risk_tier_policy": str(risk_tier_policy),
        "risk_limit_gamma": selected_gamma,
        "random_seed": int(random_seed),
    }
    (output / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    pd.DataFrame([split_metadata]).to_csv(output / "split_metadata.csv", index=False)
    gen_metadata.to_csv(output / "generator_metadata.csv", index=False)
    return output
