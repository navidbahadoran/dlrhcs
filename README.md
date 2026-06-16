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
oracle does not land in