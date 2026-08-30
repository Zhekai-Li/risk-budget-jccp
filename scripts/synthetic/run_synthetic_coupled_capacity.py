from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.algorithms.equal_allocation import equal_allocation
from risk_budget_jccp.algorithms.optimized_bernstein_mm import solve_coupled_bernstein_mm
from risk_budget_jccp.evaluation.metrics import relative_improvement_percent
from risk_budget_jccp.models.synthetic_capacity import (
    exact_gaussian_joint_violation,
    make_capacity_instance,
    solve_fixed_bernstein,
)
from risk_budget_jccp.utils.io import load_yaml, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic coupled-capacity benchmarks.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    return parser.parse_args()


def _result_row(
    *,
    heterogeneity: float,
    equal_objective: float,
    optimized_objective: float,
    alpha_vec,
    exact_joint_violation: float,
    iterations: int,
    runtime: float,
    final_residual: float,
) -> dict[str, float | int]:
    return {
        "heterogeneity": heterogeneity,
        "equal_objective": equal_objective,
        "optimized_objective": optimized_objective,
        "improvement_percent": relative_improvement_percent(
            equal_value=equal_objective,
            optimized_value=optimized_objective,
        ),
        "max_share": float(alpha_vec.max() / alpha_vec.sum()),
        "exact_joint_violation": exact_joint_violation,
        "iterations": iterations,
        "runtime": runtime,
        "final_residual": final_residual,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    alpha = float(config["alpha"])
    seed = int(config["seed"])
    dimension = int(config["dimension"])
    num_constraints = int(config["num_constraints"])
    heterogeneity_values = list(config["heterogeneity_values"])
    demand_level = float(config.get("demand_level", 1.0))
    sigma_scale = float(config.get("sigma_scale", 1.0))
    solver_config = dict(config.get("solver", {}))

    rows: list[dict[str, float | int]] = []
    for heterogeneity in heterogeneity_values:
        instance = make_capacity_instance(
            dimension=dimension,
            num_constraints=num_constraints,
            heterogeneity=float(heterogeneity),
            seed=seed,
            demand_level=demand_level,
            sigma_scale=sigma_scale,
        )
        alpha_equal = equal_allocation(m=num_constraints, alpha=alpha)
        equal_solution = solve_fixed_bernstein(instance, alpha_equal)
        optimized_solution = solve_coupled_bernstein_mm(
            instance,
            alpha=alpha,
            eps_alpha=float(solver_config.get("eps_alpha", 1e-6)),
            eps_theta=float(solver_config.get("eps_theta", 1e-6)),
            max_iter=int(solver_config.get("max_iter", 50)),
            tol=float(solver_config.get("tol", 1e-6)),
        )

        rows.append(
            _result_row(
                heterogeneity=float(heterogeneity),
                equal_objective=float(equal_solution.objective),
                optimized_objective=float(optimized_solution.objective),
                alpha_vec=optimized_solution.alpha,
                exact_joint_violation=exact_gaussian_joint_violation(
                    instance,
                    optimized_solution.x,
                ),
                iterations=int(optimized_solution.iterations),
                runtime=float(optimized_solution.runtime),
                final_residual=float(optimized_solution.final_residual),
            )
        )

    results = pd.DataFrame(rows)
    write_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
