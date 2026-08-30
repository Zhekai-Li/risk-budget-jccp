from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "tables"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tables_latex"
REQUIRED_INPUTS = (
    "synthetic_service.csv",
    "synthetic_cross_domain.csv",
    "synthetic_coupled_capacity.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LaTeX table fragments from experiment CSVs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing result CSVs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where LaTeX table fragments will be written.",
    )
    return parser.parse_args(argv)


def _require_inputs(input_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for filename in REQUIRED_INPUTS:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
        resolved[filename] = path
    return resolved


def _latex_table(frame: pd.DataFrame, *, columns: list[str], rename: dict[str, str]) -> str:
    table = frame.loc[:, columns].rename(columns=rename)
    return table.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3f}")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    inputs = _require_inputs(input_dir)

    synthetic_service = pd.read_csv(inputs["synthetic_service.csv"])
    synthetic_cross_domain = pd.read_csv(inputs["synthetic_cross_domain.csv"])
    synthetic_capacity = pd.read_csv(inputs["synthetic_coupled_capacity.csv"])

    bernstein_service = synthetic_service.loc[synthetic_service["method"] == "bernstein"].reset_index(drop=True)
    cantelli_service = synthetic_service.loc[synthetic_service["method"] == "cantelli"].reset_index(drop=True)
    cvar_service = synthetic_service.loc[synthetic_service["method"] == "cvar"].reset_index(drop=True)

    _write_text(
        output_dir / "service_bernstein.tex",
        _latex_table(
            bernstein_service,
            columns=[
                "m",
                "heterogeneity",
                "equal_objective",
                "optimized_objective",
                "improvement_percent",
                "max_share",
                "entropy",
            ],
            rename={
                "m": "$m$",
                "heterogeneity": "$H$",
                "equal_objective": "Equal",
                "optimized_objective": "Optimized",
                "improvement_percent": "Improvement (\\%)",
                "max_share": "Max share",
                "entropy": "Entropy",
            },
        ),
    )
    _write_text(
        output_dir / "service_cantelli.tex",
        _latex_table(
            cantelli_service,
            columns=[
                "m",
                "heterogeneity",
                "equal_objective",
                "optimized_objective",
                "improvement_percent",
                "max_share",
                "entropy",
            ],
            rename={
                "m": "$m$",
                "heterogeneity": "$H$",
                "equal_objective": "Equal",
                "optimized_objective": "Optimized",
                "improvement_percent": "Improvement (\\%)",
                "max_share": "Max share",
                "entropy": "Entropy",
            },
        ),
    )
    _write_text(
        output_dir / "service_cvar.tex",
        _latex_table(
            cvar_service,
            columns=[
                "m",
                "heterogeneity",
                "equal_objective",
                "optimized_objective",
                "improvement_percent",
                "max_share",
                "entropy",
            ],
            rename={
                "m": "$m$",
                "heterogeneity": "$H$",
                "equal_objective": "Equal",
                "optimized_objective": "Optimized",
                "improvement_percent": "Improvement (\\%)",
                "max_share": "Max share",
                "entropy": "Entropy",
            },
        ),
    )
    _write_text(
        output_dir / "synthetic_cross_domain.tex",
        _latex_table(
            synthetic_cross_domain,
            columns=[
                "label",
                "method",
                "m",
                "equal_objective",
                "optimized_objective",
                "improvement_percent",
                "max_share",
                "exact_joint_violation",
            ],
            rename={
                "label": "Domain",
                "method": "Certificate",
                "m": "$m$",
                "equal_objective": "Equal",
                "optimized_objective": "Optimized",
                "improvement_percent": "Improvement (\\%)",
                "max_share": "Max share",
                "exact_joint_violation": "Joint violation",
            },
        ),
    )
    _write_text(
        output_dir / "coupled_capacity.tex",
        _latex_table(
            synthetic_capacity,
            columns=[
                "heterogeneity",
                "equal_objective",
                "optimized_objective",
                "improvement_percent",
                "max_share",
                "exact_joint_violation",
                "iterations",
            ],
            rename={
                "heterogeneity": "$H$",
                "equal_objective": "Equal",
                "optimized_objective": "Optimized",
                "improvement_percent": "Improvement (\\%)",
                "max_share": "Max share",
                "exact_joint_violation": "Joint violation",
                "iterations": "Iter.",
            },
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
