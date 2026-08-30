from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.algorithms.equal_allocation import equal_allocation
from risk_budget_jccp.algorithms.optimized_separable import (
    solve_separable_bernstein,
    solve_separable_cantelli,
    solve_separable_normal_cvar,
)
from risk_budget_jccp.evaluation.metrics import (
    normalized_entropy,
    relative_improvement_percent,
)
from risk_budget_jccp.models.synthetic_service import (
    exact_gaussian_joint_violation,
    make_service_instance,
    service_objective,
)
from risk_budget_jccp.utils.io import load_yaml, write_csv


OBJECTIVE_REPORTING_ATOL = 1e-8
OBJECTIVE_REPORTING_RTOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic service benchmarks.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--raw-output",
        help="Optional path for row-level replication results (defaults beside --output).",
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        help="Optional list of m values overriding the config.",
    )
    parser.add_argument(
        "--heterogeneity-values",
        nargs="+",
        type=float,
        help="Optional list of heterogeneity values overriding the config.",
    )
    return parser.parse_args()


def _reported_solution(
    *,
    equal_objective: float,
    equal_allocation: np.ndarray,
    optimized_objective: float,
    optimized_allocation: np.ndarray,
) -> tuple[float, np.ndarray]:
    if (
        optimized_objective > equal_objective
        and np.isclose(
            optimized_objective,
            equal_objective,
            atol=OBJECTIVE_REPORTING_ATOL,
            rtol=OBJECTIVE_REPORTING_RTOL,
        )
    ):
        return equal_objective, np.asarray(equal_allocation, dtype=float)
    return optimized_objective, np.asarray(optimized_allocation, dtype=float)


def _result_row(
    *,
    method: str,
    m: int,
    heterogeneity: float,
    equal_objective: float,
    optimized_objective: float,
    allocation: np.ndarray,
    replication: int,
    seed: int,
) -> dict[str, float | int | str]:
    alpha_vec = np.asarray(allocation, dtype=float)
    return {
        "method": method,
        "m": m,
        "heterogeneity": heterogeneity,
        "replication": replication,
        "seed": seed,
        "equal_objective": equal_objective,
        "optimized_objective": optimized_objective,
        "improvement_percent": relative_improvement_percent(
            equal_value=equal_objective,
            optimized_value=optimized_objective,
        ),
        "max_share": float(alpha_vec.max() / alpha_vec.sum()),
        "entropy": normalized_entropy(alpha_vec),
        "exact_joint_violation": exact_gaussian_joint_violation(alpha_vec, method),
        "alpha_allocation": json.dumps(alpha_vec.tolist(), separators=(",", ":")),
    }


METHOD_SOLVERS = {
    "bernstein": solve_separable_bernstein,
    "cantelli": solve_separable_cantelli,
    "cvar": solve_separable_normal_cvar,
}


def _summarize(raw_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["method", "m", "heterogeneity"]
    value_columns = [
        "equal_objective",
        "optimized_objective",
        "improvement_percent",
        "max_share",
        "entropy",
        "exact_joint_violation",
    ]
    grouped = raw_results.groupby(group_columns, sort=True)
    summary = grouped[value_columns].mean().reset_index()
    summary["improvement_percent_std"] = grouped["improvement_percent"].std(ddof=0).to_numpy()
    summary["replications"] = grouped.size().to_numpy()
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    alpha = float(config["alpha"])
    base_seed = int(config.get("base_seed", config.get("seed", 0)))
    replications = int(config.get("replications", 1))
    if replications <= 0:
        raise ValueError("replications must be positive")
    methods = [str(method).lower() for method in config.get("methods", METHOD_SOLVERS)]
    unknown_methods = sorted(set(methods) - set(METHOD_SOLVERS))
    if unknown_methods:
        raise ValueError(f"unsupported methods: {unknown_methods}")
    m_values = list(args.m_values) if args.m_values else list(config["m_values"])
    heterogeneity_values = (
        list(args.heterogeneity_values)
        if args.heterogeneity_values
        else list(config["heterogeneity_values"])
    )

    rows: list[dict[str, float | int | str]] = []
    for replication in range(replications):
        seed = base_seed + replication
        for m in m_values:
            for heterogeneity in heterogeneity_values:
                instance = make_service_instance(m=m, heterogeneity=heterogeneity, seed=seed)
                alpha_equal = equal_allocation(m=m, alpha=alpha)
                for method in methods:
                    alpha_optimized = METHOD_SOLVERS[method](instance.weights, alpha)
                    equal_objective = service_objective(instance.weights, alpha_equal, method)
                    optimized_objective = service_objective(instance.weights, alpha_optimized, method)
                    reported_objective, reported_allocation = _reported_solution(
                        equal_objective=equal_objective,
                        equal_allocation=alpha_equal,
                        optimized_objective=optimized_objective,
                        optimized_allocation=alpha_optimized,
                    )
                    rows.append(
                        _result_row(
                            method=method,
                            m=m,
                            heterogeneity=heterogeneity,
                            equal_objective=equal_objective,
                            optimized_objective=reported_objective,
                            allocation=reported_allocation,
                            replication=replication,
                            seed=seed,
                        )
                    )

    raw_results = pd.DataFrame(rows)
    output_path = Path(args.output)
    raw_output_path = (
        Path(args.raw_output)
        if args.raw_output
        else output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    )
    write_csv(raw_results, raw_output_path)
    write_csv(_summarize(raw_results), output_path)


if __name__ == "__main__":
    main()
