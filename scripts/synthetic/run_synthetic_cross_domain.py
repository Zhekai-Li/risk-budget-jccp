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
from risk_budget_jccp.evaluation.metrics import normalized_entropy, relative_improvement_percent
from risk_budget_jccp.models.synthetic_service import (
    exact_gaussian_joint_violation,
    make_service_instance,
    service_objective,
)
from risk_budget_jccp.utils.io import load_yaml, write_csv


SOLVERS = {
    "bernstein": solve_separable_bernstein,
    "cantelli": solve_separable_cantelli,
    "cvar": solve_separable_normal_cvar,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run application-motivated controlled synthetic configurations."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to aggregate output CSV.")
    parser.add_argument("--raw-output", required=True, help="Path to row-level output CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    alpha = float(config["alpha"])
    base_seed = int(config["base_seed"])
    replications = int(config["replications"])
    if replications <= 0:
        raise ValueError("replications must be positive")

    rows: list[dict[str, float | int | str]] = []
    for domain in config["domains"]:
        method = str(domain["method"]).lower()
        if method not in SOLVERS:
            raise ValueError(f"unsupported method: {method}")
        m = int(domain["m"])
        heterogeneity = float(domain["heterogeneity"])
        seed_offset = int(domain.get("seed_offset", 0))
        for replication in range(replications):
            seed = base_seed + seed_offset + replication
            instance = make_service_instance(m=m, heterogeneity=heterogeneity, seed=seed)
            equal_alpha = equal_allocation(m=m, alpha=alpha)
            optimized_alpha = SOLVERS[method](instance.weights, alpha)
            equal_objective = service_objective(instance.weights, equal_alpha, method)
            optimized_objective = service_objective(instance.weights, optimized_alpha, method)
            if optimized_objective > equal_objective and np.isclose(
                optimized_objective, equal_objective, atol=1e-8, rtol=1e-12
            ):
                optimized_objective = equal_objective
                optimized_alpha = equal_alpha
            rows.append(
                {
                    "domain": str(domain["domain"]),
                    "label": str(domain["label"]),
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
                    "max_share": float(optimized_alpha.max() / optimized_alpha.sum()),
                    "entropy": normalized_entropy(optimized_alpha),
                    "exact_joint_violation": exact_gaussian_joint_violation(
                        optimized_alpha, method
                    ),
                    "alpha_allocation": json.dumps(
                        optimized_alpha.tolist(), separators=(",", ":")
                    ),
                }
            )

    raw_results = pd.DataFrame(rows)
    group_columns = ["domain", "label", "method", "m", "heterogeneity"]
    value_columns = [
        "equal_objective",
        "optimized_objective",
        "improvement_percent",
        "max_share",
        "entropy",
        "exact_joint_violation",
    ]
    grouped = raw_results.groupby(group_columns, sort=False)
    summary = grouped[value_columns].mean().reset_index()
    summary["improvement_percent_std"] = grouped["improvement_percent"].std(ddof=0).to_numpy()
    summary["replications"] = grouped.size().to_numpy()
    write_csv(raw_results, Path(args.raw_output))
    write_csv(summary, Path(args.output))


if __name__ == "__main__":
    main()
