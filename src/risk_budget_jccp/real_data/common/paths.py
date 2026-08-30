from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("could not locate repository root from src/risk_budget_jccp/real_data")


REPO_ROOT = repo_root()
RAW_ROOT = REPO_ROOT / "data" / "raw" / "real_data"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "real_data"
RESULTS_ROOT = REPO_ROOT / "results" / "real_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "full"


def case_raw_dir(case: str) -> Path:
    return RAW_ROOT / case


def case_processed_dir(case: str) -> Path:
    return PROCESSED_ROOT / case


def runtime_results_root(output_root: str | Path | None = None) -> Path:
    """Return the writable real-data root for a new experiment run."""
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    return root.expanduser().resolve() / "real_data"


def case_results_dir(case: str, output_root: str | Path | None = None) -> Path:
    return runtime_results_root(output_root) / "runs" / case


def ensure_case_dirs(case: str, output_root: str | Path | None = None) -> dict[str, Path]:
    results = case_results_dir(case, output_root)
    paths = {
        "raw": case_raw_dir(case),
        "processed": case_processed_dir(case),
        "results": results,
        "figures": results / "figures",
        "tables": results / "tables",
        "solutions": results / "solutions",
        "logs": results / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def publish_split_outputs(run_root: Path, case: str, output_root: str | Path | None = None) -> None:
    """Move report split assets beside ``runs/`` under the candidate root."""
    root = runtime_results_root(output_root)
    for split in ("in_sample", "out_of_sample"):
        source = run_root / split
        if not source.exists():
            continue
        destination = root / split / case
        if destination.exists():
            raise FileExistsError(f"refusing to replace existing split output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
