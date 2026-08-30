# Optimized Risk-Budget Allocation for Joint Chance Constraints

## Introduction

This repository is the reference implementation for **“Optimized Risk-Budget Allocation for Safe Approximations of Joint Chance Constraints.”** Joint chance constraints are commonly made tractable by certifying scalar constraints and assigning each one a share of a system violation budget. The usual equal-allocation baseline uses `alpha_i = alpha / m`; this code instead optimizes the risk budgets together with the physical decision.

The implementation supports Bernstein, Cantelli, and CVaR certificates, separable and coupled synthetic benchmarks, and chronological M5, Power, and French real-data studies. Solver success, calibration-certificate acceptance, and reliability on held-out data are reported separately.

## Core Results

The committed 30-seed synthetic sweep uses seeds 7–36. For heterogeneity `H`, the cost and uncertainty factors are generated independently with lognormal scale `H / sqrt(2)`, so their product has log-scale `H`. The strongest mean gains over equal allocation are 7.66% for Bernstein, 28.41% for Cantelli, and 8.79% for Gaussian CVaR. The single fixed-seed coupled benchmark reaches 14.48%. The five application-motivated cross-domain rows are controlled synthetic configurations—not five additional real-data experiments—and yield gains from 3.35% to 14.88%.

The verified chronological real-data results are unchanged:

| Case / certificate | Objective gain | Calibration JVP | Held-out JVP | Interpretation |
| --- | ---: | ---: | ---: | --- |
| M5 / CVaR | 4.7% | 1.4% | 35.3% | Accepted calibration certificate; unreliable after demand shift |
| M5 / Cantelli | 5.0% | 0.0% | 0.0% | Conservative and reliable |
| Power / Bernstein | 2.3% | 1.0% | 6.3% | Objective gain with modest target exceedance |
| French / CVaR | 7.9% | 1.5% | 4.8% | Objective gain with held-out reliability |
| French / Bernstein | 13.6% | 2.5% | 7.5% | Largest gain, but above the held-out target |

“Objective gain” compares optimized and equal allocations. “Calibration JVP” is measured on the calibration sample and is a certificate-backed bound only where the recorded certificate contract says so. “Held-out JVP” is a chronological reliability audit, not a formal guarantee.

## Repository Structure

```text
src/risk_budget_jccp/   Algorithms, models, real-data cases, reporting utilities
configs/                One real-data profile set and one config per synthetic study
scripts/                Synthetic, real-data, reporting, and pipeline entry points
tests/                  Unit, integration, offline fixture, and hygiene tests
results/                Read-only lightweight reference CSV, LaTeX, and PDF artifacts
artifacts/              Gitignored outputs from new runs and downloaded release assets
data/                   Gitignored raw and processed third-party inputs
```

Committed `results/` files are reference artifacts. All reproduction commands write new outputs beneath gitignored `artifacts/`; they do not overwrite `results/`.

## Installation

Python 3.11 and 3.12 are supported; Python 3.12 is the reference environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` pins the environment used for verification. For package development without the pinned file, use `python -m pip install -e '.[dev]'`.

## Quick Start

```bash
git clone https://github.com/Zhekai-Li/risk-budget-jccp.git
cd risk-budget-jccp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/pipeline/smoke.py
python scripts/reporting/validate_results.py
```

Then inspect [`results/tables/synthetic_service.csv`](results/tables/synthetic_service.csv) and [`results/real_data/summary/real_data_summary_out_of_sample.csv`](results/real_data/summary/real_data_summary_out_of_sample.csv).

## Smoke Test

```bash
python scripts/pipeline/smoke.py
```

The smoke test is offline and requires neither network access nor real datasets. It covers all three synthetic runners, tiny M5/Power/French fixtures, table and paper-figure reporting, configuration checks, and both canonical and reproduction validators. On a recent laptop it normally finishes in about 1–3 minutes and ends with:

```text
offline smoke pipeline passed
```

## Full Run

Review the complete command sequence without downloading data or writing outputs:

```bash
python scripts/pipeline/reproduce.py --output-root artifacts/full --dry-run
```

Run the full reproduction, or resume an interrupted output tree:

```bash
python scripts/pipeline/reproduce.py --output-root artifacts/full
python scripts/pipeline/reproduce.py --output-root artifacts/full --resume
```

The pipeline prepares M5 competition data, the RTS-GMLC power-system data, and Kenneth R. French’s 49 Industry Portfolios; runs the synthetic, main real-data, and sensitivity experiments; rebuilds the six paper figures; validates deterministic fields with `rtol=1e-4` and `atol=1e-6`; and writes `artifacts/full/manifest.json` with commands, versions, durations, seeds, and SHA-256 checksums. Network access is required only when raw inputs are absent. The reference profile uses roughly 350 MB of raw/processed data locally; reserve at least 2 GB and several hours for the full sensitivity run. Exact time depends strongly on solver and network performance.

Without `--resume`, a non-empty output directory is rejected. With `--resume`, stages whose expected outputs already exist are skipped, allowing recovery after a download or solver failure. No mode writes to committed `results/`.

## Result Artifacts

| Experiment or paper item | Configuration | Reproduction command | Committed reference |
| --- | --- | --- | --- |
| Separable Bernstein/Cantelli/CVaR sweep | [`configs/synthetic/synthetic_service.yaml`](configs/synthetic/synthetic_service.yaml) | `python scripts/synthetic/run_synthetic.py --config configs/synthetic/synthetic_service.yaml --output artifacts/synthetic_service.csv` | [`results/tables/synthetic_service.csv`](results/tables/synthetic_service.csv), raw CSV, and three LaTeX fragments |
| Application-motivated controlled configurations | [`configs/synthetic/synthetic_cross_domain.yaml`](configs/synthetic/synthetic_cross_domain.yaml) | `python scripts/synthetic/run_synthetic_cross_domain.py --config configs/synthetic/synthetic_cross_domain.yaml --output artifacts/synthetic_cross_domain.csv --raw-output artifacts/synthetic_cross_domain_raw.csv` | [`results/tables/synthetic_cross_domain.csv`](results/tables/synthetic_cross_domain.csv) and raw CSV |
| Coupled capacity benchmark | [`configs/synthetic/synthetic_coupled_capacity.yaml`](configs/synthetic/synthetic_coupled_capacity.yaml) | `python scripts/synthetic/run_synthetic_coupled_capacity.py --config configs/synthetic/synthetic_coupled_capacity.yaml --output artifacts/synthetic_coupled_capacity.csv` | [`results/tables/synthetic_coupled_capacity.csv`](results/tables/synthetic_coupled_capacity.csv) |
| M5, Power, French main tables | [`configs/real_data/main.yaml`](configs/real_data/main.yaml) | `python scripts/real_data/run_all_real_data.py --config configs/real_data/main.yaml --output-root artifacts/full --require-prepared-data` | [`results/real_data/summary/`](results/real_data/summary/) |
| Sensitivity profiles | [`configs/real_data/sensitivity.yaml`](configs/real_data/sensitivity.yaml) | `python scripts/real_data/run_real_data_sensitivity.py --config configs/real_data/sensitivity.yaml --output-root artifacts/full --require-prepared-data` | Full Release asset |
| Six paper figures | Main results and generated run details | `python scripts/reporting/make_paper_figures.py --output-dir artifacts/full/paper --summary-dir artifacts/full/real_data/summary --runs-dir artifacts/full/real_data/runs --processed-dir data/processed/real_data` | [`results/paper/`](results/paper/) |
| Full paper reproduction | All configs above | `python scripts/pipeline/reproduce.py --output-root artifacts/full` | Release `ejor-submission-v1` |

The GitHub Release asset `risk-budget-jccp-ejor-submission-v1-full-results.tar.gz` contains complete logs, solution JSON, split reports, sensitivity outputs, and a checksum manifest. The repository keeps only lightweight canonical artifacts.

## Data, Citation, and License

- M5 inputs are subject to the M5 Forecasting Accuracy competition’s access and reuse terms. The preparation code accepts an existing competition ZIP and records provenance.
- Power inputs come from the public [RTS-GMLC repository](https://github.com/GridMod/RTS-GMLC); any optional NREL service use remains subject to its own terms.
- French returns come from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and remain subject to the library’s terms.

If you use the software, cite the paper and authors listed in [`CITATION.cff`](CITATION.cff). The software authors are Chonghe Jiang and Zhekai Li. No Zenodo software record or DOI is used.

The code is released under the [MIT License](LICENSE). Third-party datasets are not redistributed by this repository and retain their original terms.
