from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cvxpy as cp
from cvxpy.error import SolverError


VALID_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True)
class SolveDiagnostics:
    status: str
    solver: str
    runtime: float
    value: float | None


def installed_solver_name() -> str:
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver in installed:
            return solver
    raise RuntimeError("no supported CVXPY solver installed; need CLARABEL, ECOS, or SCS")


def installed_solver_names() -> list[str]:
    installed = set(cp.installed_solvers())
    solvers = [solver for solver in ("CLARABEL", "ECOS", "SCS") if solver in installed]
    if not solvers:
        raise RuntimeError("no supported CVXPY solver installed; need CLARABEL, ECOS, or SCS")
    return solvers


def solve_problem(problem: cp.Problem, *, warm_start: bool = True, **kwargs: Any) -> SolveDiagnostics:
    explicit_solver = kwargs.pop("solver", None)
    solvers = [explicit_solver] if explicit_solver is not None else installed_solver_names()
    errors: list[str] = []
    total_runtime = 0.0
    for solver in solvers:
        start = time.perf_counter()
        try:
            problem.solve(solver=solver, warm_start=warm_start, **kwargs)
        except SolverError as exc:
            total_runtime += time.perf_counter() - start
            errors.append(f"{solver}: {exc}")
            continue
        runtime = time.perf_counter() - start
        total_runtime += runtime
        diagnostics = SolveDiagnostics(
            status=str(problem.status),
            solver=solver,
            runtime=float(total_runtime),
            value=None if problem.value is None else float(problem.value),
        )
        if diagnostics.status in VALID_STATUSES or explicit_solver is not None:
            return diagnostics
        errors.append(f"{solver}: status={diagnostics.status}")
    return SolveDiagnostics(
        status="solver_failed:" + " | ".join(errors),
        solver=",".join(str(solver) for solver in solvers),
        runtime=float(total_runtime),
        value=None,
    )


def require_success(diagnostics: SolveDiagnostics) -> None:
    if diagnostics.status not in VALID_STATUSES:
        raise RuntimeError(
            f"CVXPY solve failed with status={diagnostics.status}, solver={diagnostics.solver}"
        )
