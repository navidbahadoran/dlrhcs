# dlrhcs — Replication Package

**Cross-Fitted Debiased Inference for Dynamic Panels with Low-Rank Heterogeneous
Coefficients.**

This repository contains the code that *generates* every number in the paper's
simulation and empirical sections from scratch. Nothing is transcribed by hand:
`run_all.py` is run on a clean machine and whatever it prints is what goes in the
tables. If the code's output ever disagreed with an earlier draft, the draft was
corrected, never the code.

The estimator, the cross-fitting scheme, the debiasing step, data-driven rank
selection, the Monte Carlo design, and the empirical application all follow
`IMPLEMENTATION_SPEC.md` section by section. Each simulation experiment is tied
to a specific theorem in `THEOREM_MAP.md`.

---

## Contents

1. [What the method does](#1-what-the-method-does)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick check (minutes)](#4-quick-check-minutes)
5. [The oracle checkpoint](#5-the-oracle-checkpoint)
6. [Reproducing each output](#6-reproducing-each-output)
7. [Data: download and preparation](#7-data-download-and-preparation)
8. [Headline results](#8-headline-results)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Design notes](#10-design-notes)
11. [Citation and license](#11-citation-and-license)

---

## 1. What the method does

The model is a heterogeneous dynamic panel in which every coefficient varies
across both unit *i* and time *t*, but the coefficient *surfaces* are low rank:

```
y_it = sum_m  Z^(m)_it * Gamma^(m)_it  +  h_it  +  u_it
```

Each coefficient surface `Gamma^(m)` (and the interactive nuisance `h`) is an
unknown low-rank `T x N` matrix. The target is not a single scalar but a smooth
functional of these surfaces — an individual entry, a (group) average
coefficient, a between-group contrast, or a derived dynamic functional such as an
impulse response, the long-run multiplier, or the companion spectral radius.

The estimator (a) fits the low-rank surfaces by an alternating factor-ridge
procedure on *cross-fitted, forward-purged* folds, (b) solves for the Riesz
representer of the target functional matrix-free by conjugate gradients on the
tangent space, (c) applies a one-step debiasing correction, and (d) studentizes
with both a White (heteroskedasticity-robust) and a within-period
(cross-sectional dependence-robust) standard error. The theory delivers a
`sqrt(T+N)` central limit theorem with valid confidence intervals.

## 2. Repository layout

```
dlrhcs/                package (pure NumPy/SciPy)
  design.py            design map A and its adjoint A*; H as the "ones" block
  factorridge.py       alternating factor-ridge ALS + warm start + ridge annealing
  folds.py             scattered cross-fitting folds + forward-exclusion window
  targets.py           target directions, tangent projector, matrix-free Riesz (CG)
  ranks.py             cross-fitted rank criterion + data-driven roadmap
  onestep.py           one-step debiasing + White/xs variances + IRF/LRM delta method
  pipeline.py          end-to-end feasible procedure (+ infeasible oracle mode)
  dgp.py               Monte Carlo DGP (iid baseline; hetero / xs variants)
  mc.py                Monte Carlo harness (checkpointed, resumable, parallel)
  experiments.py       theorem-justification experiments
  empirical.py         AR(2) empirical pipeline (Zillow; parked metro loader)
  report.py            writes the LaTeX tables
configs/
  pilot.json           tiny config — smoke test, runs in minutes
  full.json            submission-scale config
data/                  download instructions + a data-build script (NO data files)
tests/test_core.py     spec section-15 unit checklist (7 tests)
run_all.py             one-command reproduction (stage-by-stage)
requirements.txt       pinned dependencies
IMPLEMENTATION_SPEC.md the estimator specification, section by section
THEOREM_MAP.md         which experiment justifies which theorem
VALIDATION.md          validation record (unit tests + small-R experiment runs)
EMPIRICAL_FINDINGS.md  empirical application notes
```

Generated results land in `outputs/` (git-ignored; regenerate with `run_all.py`).
No data files and no paper sources are tracked — see sections 7 and 11.

## 3. Installation

Python 3.10 or 3.11. The only dependencies are NumPy, SciPy, and joblib.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`run_all.py` pins BLAS to a single thread automatically (so the parallel workers
are deterministic and contention-free). To do it by hand in an interactive
session:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 4. Quick check (minutes)

```bash
python tests/test_core.py                               # 7/7 must pass
python run_all.py --config configs/pilot.json --stage all
```

`tests/test_core.py` is the specification's section-15 checklist: the `A`/`A*`
adjoint identity, the forward-exclusion index set on a hand-checked grid, ALS
objective monotonicity, tangent-projector idempotency, the Riesz representer
identity, exact recovery in the noiseless case, and the Gram-scale convention.
All must pass before any large run.

## 5. The oracle checkpoint

The single most important gate (specification section 12). The *infeasible*
oracle passes the true tangent spaces into the Riesz solve, isolating the
influence-function + martingale-CLT core from first-stage estimation error.

```bash
python run_all.py --config configs/full.json --stage oracle
```

Theory predicts essentially unbiased estimates and **coverage in [0.93, 0.96]**,
with the mean standard error matching the Monte Carlo standard deviation. Our
validation run gave **mean coverage 0.952** across the eight targets. If the
oracle does not land in the band on your machine, stop and debug before running
the feasible grid — a feasible failure downstream of a passing oracle is a
first-stage issue, not a core one.

## 6. Reproducing each output

Every stage is independent, resume-safe, and writes named artifacts. Run them in
order, or all at once with `--stage all`.

| Command | What it computes | Output(s) | Paper object |
|---|---|---|---|
| `--stage oracle` | infeasible oracle benchmark | `outputs/sim/oracle_100.json` | section-12 coverage checkpoint |
| `--stage grid` | feasible convergence study over (T+,N) in {50,100,200,400} | `outputs/sim/grid_*.json` | convergence + precision tables |
| `--stage purge` | forward-exclusion window `q` sweep | `outputs/sim/purge_100.json` | purge-robustness table |
| `--stage theorems` | rank-consistency, IRF/LRM coverage, cross-sectional-dependence, contiguous-fold singularity, debiasing | `outputs/theorems.json` | theorem-justification figures/numbers |
| `--stage empirical` | Zillow AR(2) application | `outputs/empirical/zillow.json` | empirical section |
| `--stage tables` | renders LaTeX tables from the JSON above | `outputs/tables/tab_sim_*.tex` | tables pasted into the manuscript |

Useful flags:

```bash
# one panel size only (e.g. just the largest grid cell)
python run_all.py --config configs/full.json --stage grid  --only 400
# one purge window only
python run_all.py --config configs/full.json --stage purge --only 2
```

`configs/full.json` defines the exact grid, replication counts, tuning, and
seeds. The grid is geometric (each step doubles `T+N`) so the `sqrt(T+N)`
standard-error contraction can be read straight off the precision table. The
purge stage uses a deliberately harder DGP (`purge_dgp`, with `rho_y = 0.95`) to
stress the forward-exclusion window.

## 7. Data: download and preparation

No data files are committed (the sources are public but carry citation/red
istribution terms). Download them yourself and place them as described below;
the fingerprints printed in each `outputs/empirical/*.json` pin the exact vintage.

### Zillow (the paper's empirical application)

1. Go to <https://www.zillow.com/research/data/> → "Home Values".
2. Data Type = **ZHVI by tier**, Geography = **Metro & U.S.**, and download
   **both** tiers:
   - bottom tier: `Metro_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv`
   - top tier: `Metro_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv`
3. Save them as:
   - `data/zillow_metro_bottom.csv`
   - `data/zillow_metro_top.csv`
4. Run `python run_all.py --config configs/full.json --stage empirical`.

The loader stacks each metro's top- and bottom-tier series as two units in one
panel, balances and aligns, applies a minimal per-series stationarization (ADF
test: difference once only if a unit root — house-price levels are, so they
become log-returns — otherwise keep the level), and standardizes. The full
cleaning rule is documented in the paper's data appendix and in
`EMPIRICAL_FINDINGS.md`.

### Metro unemployment (optional, built but not used by default)

A second application is fully implemented and parked for later use; it is **not**
part of the default reproduction. To build the panel:

1. From the BLS LAUS flat files <https://download.bls.gov/pub/time.series/la/>
   download `la.data.60.Metro.txt` and `la.series` into `data/metro/`.
2. Run `python data/metro/build_metro_panel.py`, which writes
   `data/metro/metro_unemployment.csv` (383 metros, 1990–2025 annual averages).
3. Load it via `dlrhcs.empirical.load_metro`.

See `data/metro/README.md` for the cleaning rule and why it is parked (the series
are strongly common-factor-driven; the README explains the level-vs-difference
choice).

## 8. Headline results

Indicative numbers from validation runs; the exact values are regenerated by
`run_all.py --config configs/full.json`.

**Simulation.**
- Oracle coverage **0.952** (target band [0.93, 0.96]); mean s.e. ≈ Monte Carlo s.d.
- Feasible coverage approaches nominal as the panel grows; the standard error
  contracts at the `sqrt(T+N)` rate across the geometric grid.
- Data-driven rank selection: P(correct rank) rises from **0.60** at (50,50) to
  **1.00** by (200,200).
- The forward-exclusion window controls the leakage that a contiguous (non-purged)
  fold would suffer (the contiguous design is near-singular by construction).

**Empirical (Zillow, top vs bottom price tier).**
- lag-1 mean **+1.22**, lag-2 mean **−0.40** (momentum then reversion).
- companion spectral radius **0.63** (< 1, stationary); a+b ≈ **0.82**;
  long-run multiplier ≈ **5.6**.
- the top-vs-bottom tier contrast is estimated within a single panel via the
  paper's contrast functional, with both White and within-period standard errors.

## 9. Reproducibility notes

**Determinism.** Per-replication seeds are `SeedSequence([master_seed, rep])`;
SVD signs are canonicalized after every decomposition, so factors, Riesz weights,
and the downstream estimates are reproducible. Record `python --version`,
`numpy.__version__`, and the BLAS in your run log.

**Resumable & parallel.** Each replication writes one JSONL line keyed by its
index, so a stopped grid resumes where it left off and can be split across
machines by replication range. Set `n_jobs` in the config (`-1` = all cores). The
joblib `loky` backend is used because NumPy/OpenBLAS can deadlock under `fork` —
do not change the backend.

**Runtime (rough, per core, full settings).** A few seconds per replication at
(100,100), scaling with panel size; `R = 1000` at the largest grid cell is the
long pole (budget a few core-hours — it parallelizes linearly and is fully
resumable). The pilot config finishes in minutes.

**File integrity.** Every file in this repo is plain UTF-8 text; if you fork or
mirror it, a quick `git fsck` and a `python tests/test_core.py` confirm the code
is intact before a long run.

## 10. Design notes

- **Never SVD the outcome `Y`.** The low-rank structure lives in each coefficient
  surface, recovered only after removing the Hadamard design weighting — not in
  `Y` itself.
- **Matrix-free Riesz.** The feasible debiasing weights solve the tangent-space
  normal equations by conjugate gradients through `A`, `A*`, and the block
  projector. No dense representer basis is ever materialized, which is what keeps
  the large Zillow panel tractable.
- **Ridge annealing.** The lag block is weakly identified; a graduated ridge
  schedule lands the ALS in the correct global basin (validated by exact recovery
  in the noiseless case).
- **Forward-purged cross-fitting.** Folds are scattered across cells and a
  forward-exclusion window removes the cells a fold's training set could leak
  into, which is what makes the cross-fitted residuals valid for a dynamic panel.

## 11. Citation and license

If you use this code, please cite the paper (citation to be added on
publication). The data are not redistributed here: cite **Zillow** (ZHVI) and the
**U.S. Bureau of Labor Statistics** (LAUS) directly per their terms.

No license file is included yet — add one (e.g. MIT) before making the repository
public if you intend to permit reuse.
