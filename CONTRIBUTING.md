# Contributing

Use Python 3.11 or newer. For the pinned reference environment, install `requirements.txt` and then the package without dependency resolution:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest -q
```

Keep reusable algorithms and models in `src/risk_budget_jccp/`; keep command-line orchestration in `scripts/`; keep configurations declarative in `configs/`; and add behavior tests under `tests/`. Public solver APIs in `algorithms/`, `models/`, and the case solvers should remain backward compatible.

Run `python scripts/pipeline/smoke.py` before submitting a change. The smoke suite must remain offline.

`results/` is the reviewed reference record. Ordinary development and reproduction commands must write beneath `artifacts/` and must not mutate tracked results. If a paper artifact intentionally changes, explain the reason, regenerate it explicitly, validate it, and include a visual review where appropriate. Never commit downloaded raw data, credentials, local absolute paths, or environment-specific commands.
