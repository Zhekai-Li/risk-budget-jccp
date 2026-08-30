from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG = REPO_ROOT / "configs" / "real_data" / "main.yaml"
SENSITIVITY_CONFIG = REPO_ROOT / "configs" / "real_data" / "sensitivity.yaml"
SEED = 20260525


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce all JCCP paper artifacts in an isolated output tree.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "full"))
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered commands without executing them.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing output tree and skip completed stages.")
    parser.add_argument("--no-compare", action="store_true", help="Validate completeness without comparing references.")
    return parser.parse_args(argv)


def _commands(output_root: Path, no_compare: bool) -> list[tuple[str, list[str], tuple[Path, ...]]]:
    py = sys.executable
    synthetic = output_root / "synthetic"
    real_data = output_root / "real_data"
    paper = output_root / "paper"
    commands: list[tuple[str, list[str], tuple[Path, ...]]] = [
        (
            "preflight",
            [py, "scripts/real_data/check_real_data.py", "--config", str(MAIN_CONFIG)],
            (),
        ),
        (
            "synthetic-service",
            [py, "scripts/synthetic/run_synthetic.py", "--config", "configs/synthetic/synthetic_service.yaml", "--output", str(synthetic / "synthetic_service.csv")],
            (synthetic / "synthetic_service.csv", synthetic / "synthetic_service_raw.csv"),
        ),
        (
            "synthetic-cross-domain",
            [
                py,
                "scripts/synthetic/run_synthetic_cross_domain.py",
                "--config",
                "configs/synthetic/synthetic_cross_domain.yaml",
                "--output",
                str(synthetic / "synthetic_cross_domain.csv"),
                "--raw-output",
                str(synthetic / "synthetic_cross_domain_raw.csv"),
            ],
            (
                synthetic / "synthetic_cross_domain.csv",
                synthetic / "synthetic_cross_domain_raw.csv",
            ),
        ),
        (
            "synthetic-capacity",
            [py, "scripts/synthetic/run_synthetic_coupled_capacity.py", "--config", "configs/synthetic/synthetic_coupled_capacity.yaml", "--output", str(synthetic / "synthetic_coupled_capacity.csv")],
            (synthetic / "synthetic_coupled_capacity.csv",),
        ),
        (
            "prepare-real-data",
            [
                py,
                "scripts/real_data/prepare_all_real_data.py",
                "--config",
                str(MAIN_CONFIG),
                "--output-root",
                str(output_root),
            ],
            (),
        ),
        (
            "real-data-main",
            [py, "scripts/real_data/run_all_real_data.py", "--config", str(MAIN_CONFIG), "--output-root", str(output_root), "--require-prepared-data"],
            (
                real_data / "summary" / "real_data_summary_in_sample.csv",
                real_data / "summary" / "real_data_summary_out_of_sample.csv",
            ),
        ),
        (
            "sensitivity",
            [py, "scripts/real_data/run_real_data_sensitivity.py", "--config", str(SENSITIVITY_CONFIG), "--output-root", str(output_root), "--require-prepared-data"],
            (real_data / "sensitivity" / "summary" / "sensitivity_summary.csv",),
        ),
        (
            "profile-selection-audit",
            [py, "scripts/real_data/select_real_data_main_profile.py", "--output-root", str(output_root)],
            (real_data / "summary" / "profile_selection_audit.csv",),
        ),
        (
            "paper-figures",
            [
                py,
                "scripts/reporting/make_paper_figures.py",
                "--output-dir",
                str(paper),
                "--summary-dir",
                str(real_data / "summary"),
                "--runs-dir",
                str(real_data / "runs"),
                "--processed-dir",
                str(REPO_ROOT / "data" / "processed" / "real_data"),
            ],
            (paper / "safety_factor_vs_budget.pdf", paper / "aggregate_improvement_vs_violation.pdf"),
        ),
    ]
    validation = [
        py,
        "scripts/reporting/validate_results.py",
        "--candidate-root",
        str(output_root),
        "--reference-root",
        str(REPO_ROOT / "results"),
    ]
    if no_compare:
        validation.append("--no-compare")
    commands.append(("validate", validation, ()))
    return commands


def _ensure_safe_output(root: Path, *, resume: bool, dry_run: bool) -> None:
    if root.exists() and any(root.iterdir()) and not resume:
        raise SystemExit(f"refusing non-empty output directory without --resume: {root}")
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "matplotlib", "scienceplots", "pyyaml")
    versions = {name: metadata.version(name) for name in names}
    versions["python"] = platform.python_version()
    return versions


def _git_revision() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    _ensure_safe_output(output_root, resume=args.resume, dry_run=args.dry_run)
    stages = _commands(output_root, args.no_compare)
    if args.dry_run:
        for name, command, _ in stages:
            print(f"[{name}] {' '.join(command)}")
        return

    started = datetime.now(timezone.utc)
    records: list[dict[str, object]] = []
    for name, command, expected in stages:
        if args.resume and expected and all(path.is_file() and path.stat().st_size > 0 for path in expected):
            records.append({"name": name, "command": command, "status": "skipped-resume", "elapsed_seconds": 0.0})
            continue
        stage_start = time.monotonic()
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
        elapsed = time.monotonic() - stage_start
        records.append({"name": name, "command": command, "status": "passed" if completed.returncode == 0 else "failed", "elapsed_seconds": elapsed})
        if completed.returncode != 0:
            raise SystemExit(f"stage {name!r} failed with exit code {completed.returncode}")
    files = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": 1,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "git_revision": _git_revision(),
        "random_seeds": {"synthetic": 7, "real_data": SEED},
        "versions": _versions(),
        "stages": records,
        "checksums_sha256": {str(path.relative_to(output_root)): _sha256(path) for path in files},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reproduction complete: {output_root}")


if __name__ == "__main__":
    main()
