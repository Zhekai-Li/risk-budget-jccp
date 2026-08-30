from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.reporting.validation import (
    validate_canonical_results,
    validate_reproduction,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical or reproduced JCCP results.")
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "results" / "tables"),
        help="Directory containing canonical synthetic CSVs.",
    )
    parser.add_argument("--candidate-root", help="Validate a complete candidate reproduction tree.")
    parser.add_argument("--reference-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--no-compare", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.candidate_root:
        report = validate_reproduction(
            Path(args.candidate_root),
            Path(args.reference_root),
            compare=not args.no_compare,
        )
        print(
            f"validated {len(report.checked_files)} artifacts; "
            f"compared {len(report.compared_files)} CSVs"
        )
        return
    checked = validate_canonical_results(Path(args.input_dir))
    print(f"canonical result validation passed ({len(checked)} CSVs)")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
