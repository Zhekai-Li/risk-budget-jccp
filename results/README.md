# Reference results

This directory contains lightweight reviewed outputs used by the paper. It is intentionally tracked so readers can inspect the headline findings without downloading data or solving the optimization problems.

- `real_data/summary/` contains the canonical cross-case CSVs, compact LaTeX tables, result-quality checks, and profile-selection audit.
- `paper/` contains cross-case paper figures.
- `tables/` and `tables_latex/` contain the three canonical synthetic studies.

Treat these files as immutable reference artifacts during normal runs. `scripts/pipeline/reproduce.py` writes candidates beneath `artifacts/` and compares deterministic CSV fields to this directory. Runtime and log timestamps are excluded from numerical comparison. The `ejor-submission-v1` GitHub Release asset holds full logs, solutions, split reports, sensitivity outputs, and checksum manifest.
