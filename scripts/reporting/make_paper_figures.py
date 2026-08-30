from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_budget_jccp.reporting.paper_figures import main


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
