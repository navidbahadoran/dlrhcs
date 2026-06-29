# dlrhcs — Replication Package

**Cross-Fitted Debiased Inference for Dynamic Panels with Low-Rank Heterogeneous
Coefficients.**

This repository contains the code that *generates* every number in the paper's
simulation and empirical sections from scratch. Nothing is transcribed by hand: the
simulation is reproduced with `run_all.py`, the two empirical applications with the
scripts in `scripts/`, and whatever the code prints is what goes in the tables. If the
code's output ever disagreed with an earlier draft, the draft was corrected, never the
code.

The package implements the estimator, the cross-fitting scheme, the debiasing step,
data-driven rank selection, the Monte Carlo design, and two empirical applications;
each simulation experiment corresponds to a theorem in the paper.

---

## Contents

1. [What the method does](#1-what-the-method-does)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick check (minutes)](#4-quick-check-minutes)
5. [The oracle checkpoint](#5-the-oracle-checkpoint)
6. [Reproducing the simulation](#6-reproducing-the-simulation)
7. [Reproducing the empirical applications](#7-reproducing-the-empirical-applications)
8. [Data](#8-data)
9. [Headline results](#9-headline-results)
10. [Reproducibility notes](#10-reproducibility-notes)
11. [Design notes](#11-design-notes)
12. [Citation and license](#12-citation-and-license)

---

## 1. What the method does

The model is a heterogeneous dynamic panel in which every coefficient varies across
both unit *i* and time *t*, but the coefficient *surfaces* are low rank:

```
y_it = sum_m  Z^(m)_it * Gamma^(m)_it  +  h_it  +  u_it
```

Each coefficient surface `Gamma^(m)` (and the interactive nuisance `h`) is an unknown
low-rank `T x N` matrix. The target is not a single scalar but a smooth functional of
these surfaces — an individual entry, a (group) average coefficient, a between-group
contrast, or a derived dynamic functional such as cumulative persistence, an impulse
response, the long-run multiplier, or the companion spectral radius.

The estimator (a) fits the low-rank surfaces by an alternating factor-ridge procedure
on *cross-fitted, forward-purged* folds, (b) solves for the Riesz representer of the
target functional matrix-free by conjugate gradients on the tangent space, (c) applies
a one-step debiasing correction, and (d) studentizes with both a White
(heteroskedasticity-robust) and a within-period (cross-sectional-dependence-robust)
standard error. The theory delivers a `sqrt(T+N)` central limit theorem with valid
confidence intervals.

## 2. Repository layout

```
dlrhcs/                package (NumPy/SciPy)
  design.py            design map A and its adjoint A*; H as the "ones" block
  factorridge.py       alternating factor-ridge ALS + warm start + ridge annealing
  folds.py             scattered cross-fitting folds + forward-exclusion window
  targets.py           target directions, tangent projector, matrix-free Riesz (CG)
  ranks.py             cross-fitted rank criterion + data-driven roadmap
  onestep.py           one-step debiasing + White/xs variances + IRF/LRM delta method
  pipeline.py          end-to-end feasible procedure (+ infeasible oracle mode)
  dgp.py               Monte Carlo DGP (iid baseline; hetero / decaying-xs variants)
  mc.py                Monte Carlo harness (checkpointed, resumable, parallel)
  experiments.py       theorem-justification experiments
  empirical.py         heterogeneous AR(2)/AR(1) pipeline (run_ar2 + targets + diagnostics
                       + rank/covariate robustness) and the data loaders
  covariates.py        metro covariate loaders (CBSA permits / population / GDP)
  unemp.py             monthly LAUS unemployment panel loader (NSA deseasonalization)
  diagnostics.py       residual adequacy, fit, and coefficient-heterogeneity diagnostics
  report.py            LaTeX table/figure helpers
configs/
  pilot.json           tiny config — smoke test, runs in minutes
  fast.json            reduced-cost full pass (for iteration)
  full.json            submission-scale config
scripts/
  zillow_abc.py        housing application: AR(2) specs A/B/C + full diagnostics
  unemp_abc.py         unemployment application: AR(1) specs A/B/C + full diagnostics
  sim_report.py        builds the simulation LaTeX tables + figure coordinates
  make_maps.py         renders the geographic heterogeneity choropleths (matplotlib)
  build_metro_panel.py BLS LAUS panel builder (raw flat files -> model-ready CSV)
data/                  model-ready data (committed); raw downloads are git-ignored
tests/test_core.py     spec section-15 unit checklist
run_all.py             one-command simulation reproduction (stage-by-stage)
requirements.txt       pinned dependencies
pyproject.toml         editable install (pip install -e .)
```

Generated results land in `outputs/` (git-ignored; regenerate with the commands below).

## 3. Installation

Python 3.10–3.13. Core dependencies are NumPy, SciPy, and joblib; matplotlib is needed
only for the heterogeneity maps (`scripts/make_maps.py`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`run_all.py` pins BLAS to a single thread automatically (so the parallel workers are
deterministic and contention-free). To do it by hand in an interactive session:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 4. Quick check (minutes)

```bash
python tests/test_core.py                               # unit checklist must pass
python run_all.py --config configs/pilot.json --stage all
```

`tests/test_core.py` is the specification's section-15 checklist: the `A`/`A*` adjoint
identity, the forward-exclusion index set on a hand-checked grid, ALS objective
monotonicity, tangent-projector idempotency, the Riesz representer identity, exact
recovery in the noiseless case, and the Gram-scale convention. All must pass before any
large run.

## 5. The oracle checkpoint

The single most important gate (specification section 12). The *infeasible* oracle
passes the true tangent spaces into the Riesz solve, isolating the influence-function +
martingale-CLT core from first-stage estimation error.

```bash
python run_all.py --config configs/full.json --stage oracle
```

Theory predicts essentially unbiased estimates and **coverage in [0.93, 0.96]**, with
the mean standard error matching the Monte Carlo standard deviation. Our validation run
gives **mean coverage 0.960** across the eight targets (lag and covariate coefficient,
each on entry / group / full / contrast). If the oracle does not land in the band on
your machine, stop and debug before running the feasible grid — a feasible failure
downstream of a passing oracle is a first-stage issue, not a core one.

## 6. Reproducing the simulation

Every stage is independent, resume-safe (per-replication JSONL checkpoints in
`outputs/sim/`), and uses all cores (`n_jobs: -1` in the config). Run in order:

| Command | What it computes | Output(s) |
|---|---|---|
| `--stage oracle` | infeasible oracle benchmark (Tp=N=100) | `outputs/sim/oracle_100.json` |
| `--stage grid` | feasible convergence over T=N in {50,100,200,400} | `outputs/sim/grid_*.json` |
| `--stage purge` | forward-exclusion window `q` sweep (q in {0..6}) | `outputs/sim/purge_*.json` |
| `--stage theorems` | rank-consistency, cross-sectional-dependence, IRF, debiased-vs-plugin | `outputs/theorems.json` |

```bash
python run_all.py --config configs/full.json --stage oracle
python run_all.py --config configs/full.json --stage grid
python run_all.py --config configs/full.json --stage purge
python run_all.py --config configs/full.json --stage theorems
python scripts/sim_report.py        # -> outputs/sim/tables/*.tex  (LaTeX tables + figure coords)
```

`scripts/sim_report.py` builds the simulation tables and figure coordinates, reporting
the **lag coefficient `a`** and the **covariate coefficient `b`** as distinct theory
objects throughout (main performance, target-type, debiased-vs-plugin,
oracle-vs-feasible, purge sensitivity, plus RMSE / coverage convergence and the
studentized QQ).

Useful flags — the grid's 400-cell is the long pole, so run it separately if needed:

```bash
python run_all.py --config configs/full.json --stage grid  --only 400   # one panel size
python run_all.py --config configs/full.json --stage purge --only 2     # one purge window
```

The grid is geometric (each step doubles `T+N`) so the `sqrt(T+N)` standard-error
contraction reads straight off the precision table. The main grid uses iid innovations
(the baseline for bias/RMSE/coverage); the cross-sectional dependence that distinguishes
the White from the cross-sectional standard error is exercised separately in the
`theorems` stage, under the decaying-mixing DGP that satisfies the paper's dependence
assumption (a pervasive common factor is excluded).

## 7. Reproducing the empirical applications

The two applications run directly from `scripts/` (not through `run_all.py`). Each
fits the heterogeneous dynamic panel in three specifications — **A** (full sample, no
covariates), **B** (restricted to the covariate window, no covariates), **C**
(covariate-augmented) — so that A→B isolates the sample-restriction effect and B→C
isolates the covariate effect.

```bash
set N_JOBS=4                         # Windows; or: export N_JOBS=4
python scripts/zillow_abc.py         # housing AR(2)  -> outputs/empirical/zillow_{A,B,C}.json + zillow_abc.json
python scripts/unemp_abc.py          # unemployment AR(1) -> outputs/empirical/unemp_{A,B,C}.json + unemp_abc.json
python scripts/make_maps.py          # geographic heterogeneity maps -> paper/figures/fig_emp_map_{housing,unemp}.pdf
```

Each run reports, per specification: the lag means and group/contrast targets with both
the White and the within-period cross-sectional standard error; cumulative persistence
(global and by group), the long-run multiplier, the companion spectral radius, and
impulse responses to h=12; plug-in vs debiased estimates; residual adequacy (lagged
autocorrelation, average cross-sectional residual correlation, residual first-singular-
value share); fit (RMSE, R² over a no-dynamics baseline); coefficient-surface
heterogeneity; the cross-fitted rank-selection candidate table; and r_H- and
covariate-forced-rank robustness sweeps.

## 8. Data

**The model-ready data is committed** (so a Data Editor can run on a clean checkout
without any downloads). Only the large raw/intermediate downloads are git-ignored; the
build scripts below regenerate the model-ready files from them.

### Housing — Zillow Home Value Index (`data/zillow/`)

- `zillow_metro_top.csv`, `zillow_metro_bottom.csv` — ZHVI by metro, top and bottom
  price tiers (Zillow Research, "ZHVI by tier", Metro & U.S.). Stacked as two units per
  metro; monthly log price growth after a per-series ADF stationarization.
- `metro_monthly_covariates_2000_present.csv` — CBSA monthly covariates for spec C:
  building-permit growth (Census Building Permits Survey), population growth (Census
  annual estimates, interpolated to monthly), and real-GDP growth (BEA annual
  metropolitan GDP, interpolated). Counties are aggregated to CBSAs with
  `cbsa_county_crosswalk_2023.csv` (2023 OMB delineation).
- `zillow-covariate.py` — the build script that downloads the raw permit/population/GDP
  sources and produces the covariate file. `README.md` documents the sources.

### Unemployment — BLS LAUS (`data/unemp/`)

- `unemployment_metro_model_panel_bls_only_name_matched.csv` — monthly,
  not-seasonally-adjusted metropolitan unemployment rate (BLS Local Area Unemployment
  Statistics), 2000–2026, carrying the modern CBSA code (`ces_cbsa_code`) used to match
  covariates. The loader (`dlrhcs/unemp.py`) deseasonalizes metro-by-metro by month-of-
  year means (level-preserving) and linearly interpolates short gaps. Covariates for
  spec C are CBSA population and real-GDP growth from the housing covariate file
  (employment is deliberately excluded — it is a labor-market identity with the
  unemployment rate). `scripts/build_metro_panel.py` rebuilds the panel from the LAUS
  flat files.

### Other

- `data/us_states_geo.json` — US-state GeoJSON cached by `scripts/make_maps.py` for the
  heterogeneity choropleths (committed so the maps render offline).

The content fingerprint recorded in each `outputs/empirical/*.json` pins the exact data
vintage used.

### Primary sources (cite these)

- **U.S. Bureau of Labor Statistics (2025).** Local Area Unemployment Statistics
  (LAUS): model-based *monthly*, not-seasonally-adjusted estimates of the unemployment
  rate for metropolitan statistical areas. Accessed June 2026. Public-domain flat files:
  <https://download.bls.gov/pub/time.series/la/>; program documentation:
  <https://www.bls.gov/lau/>.
- **Zillow Research (2024).** Zillow Home Value Index (ZHVI): a measure of the typical
  home value across a region and home type. Data:
  <https://www.zillow.com/research/data/>; methodology:
  <https://www.zillow.com/research/methodology-neural-zhvi-32128/>.
- **U.S. Census Bureau** — Building Permits Survey; population estimates. **U.S. Bureau
  of Economic Analysis** — metropolitan GDP. (Spec-C covariates only.)

## 9. Headline results

Regenerated by the commands above; indicative values from the submission run.

**Simulation.**
- Oracle coverage **0.960** (target band [0.93, 0.96]); mean s.e. ≈ Monte Carlo s.d.
- Feasible grid: coverage climbs to nominal and RMSE contracts at the `sqrt(T+N)` rate
  across {50,100,200,400} — shown separately for the lag and covariate coefficients.
- Forward-exclusion: coverage is at nominal for `q in {0,1,2,3}` and degrades only when
  the purge is so long (`q=6`) that too much training data is removed.

**Housing (Zillow, spec A — full sample, N=610 metro-tiers, T=315 months).**
- lag-1 mean **+1.250**, lag-2 mean **−0.419** (momentum then partial reversal).
- companion spectral radius **0.648** (< 1, stationary); cumulative persistence
  `a+b` ≈ **0.831**; long-run multiplier ≈ **5.9**.
- robust across A/B/C: building-permit, population, and GDP-growth covariates are
  statistically indistinguishable from zero, and the dynamics are unchanged.

**Unemployment (BLS LAUS, spec B — 2005–2024, N=315 metros, T=240 months).**
- heterogeneous **AR(1)** (the criterion drops the second lag at monthly frequency);
  idiosyncratic lag-1 persistence **+0.948**, companion radius **0.948** (stationary).
- robust across A/B/C: population and GDP-growth covariates are insignificant; the
  high- vs low-unemployment group contrast is small and not distinguishable from zero.

Both applications report every target with the White and the within-period
cross-sectional standard error, and the diagnostics in section 7.

## 10. Reproducibility notes

**Determinism.** Per-replication seeds are `SeedSequence([master_seed, rep])`; SVD signs
are canonicalized after every decomposition, so factors, Riesz weights, and the
downstream estimates are reproducible. Record `python --version`, `numpy.__version__`,
and the BLAS in your run log.

**Resumable & parallel.** Each simulation replication writes one JSONL line keyed by its
index, so a stopped grid resumes where it left off and can be split across machines by
replication range. Set `n_jobs` in the config (`-1` = all cores); the empirical scripts
read the `N_JOBS` environment variable. The joblib `loky` backend is used because
NumPy/OpenBLAS can deadlock under `fork` — do not change the backend.

**Runtime (rough, full settings).** Oracle and the small grid cells are quick; the
`(400,400)` grid cell at R=500 is the long pole (several core-hours, fully resumable).
The purge and theorems stages are a few hours each. The empirical scripts take tens of
minutes each; the pilot config finishes in minutes.

## 11. Design notes

- **Never SVD the outcome `Y`.** The low-rank structure lives in each coefficient
  surface, recovered only after removing the Hadamard design weighting — not in `Y`.
- **Matrix-free Riesz.** The feasible debiasing weights solve the tangent-space normal
  equations by conjugate gradients through `A`, `A*`, and the block projector; no dense
  representer basis is materialized, which keeps the large panels tractable.
- **Ridge annealing.** The lag block is weakly identified; a graduated ridge schedule
  lands the ALS in the correct global basin (validated by exact noiseless recovery).
- **Forward-purged cross-fitting.** Folds are scattered across cells and a
  forward-exclusion window removes the cells a fold's training set could leak into,
  which is what makes the cross-fitted residuals valid for a dynamic panel.

## 12. Citation and license

If you use this code, please cite the paper (citation to be added on publication). The
data sources are public: cite **Zillow** (ZHVI), the **U.S. Bureau of Labor Statistics**
(LAUS), the **U.S. Census Bureau** (Building Permits Survey; population estimates), and
the **U.S. Bureau of Economic Analysis** (metropolitan GDP) per their terms.

The code is released under the **MIT License** (see `LICENSE`) — permissive reuse with
attribution. The license covers the code only; the empirical data remain subject to their
providers' terms (Zillow, BLS, Census, BEA), as noted above.
