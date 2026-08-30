from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.real_data.common.config import case_config, load_config, validate_common_config
from risk_budget_jccp.real_data.common.data_status import check_case_data, format_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real-data raw/processed status.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "real_data" / "main.yaml"))
    parser.add_argument("--case", choices=("m5", "power", "french"), action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_common_config(config)
    cases = args.case or ["m5", "power", "french"]
    blocks = []
    for case in cases:
        blocks.append(format_status(check_case_data(case, case_config(config, case))))
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
