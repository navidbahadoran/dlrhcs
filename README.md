# DLRHCS Replication Package

**Cross-Fitted Debiased Inference for Dynamic Panels with Low-Rank Heterogeneous Coefficients**

This repository contains code for the estimator, Monte Carlo design, simulation reporting, and empirical pipelines used in the paper. The Monte Carlo pipeline has been reorganized around a fully specified, reproducible simulation design with three DGPs, theorem-aligned standard errors, finite-fold-floor cross-fitting, retained-training diagnostics, and resume-safe batch execution.

> **Current status.** The Monte Carlo code and reporting pipeline are the current focus of this replication package. The empirical scripts remain in the repository, but the empirical data construction, interpolation rules, seasonal adjustment, outlier diagnostics, and clustered/spatial inference reporting are still under revision. Treat the empirical commands as development tools until the empirical README section is updated.

---

## Contents

1. [Repository goals](#1-repository-goals)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick smoke checks](#4-quick-smoke-checks)
5. [Monte Carlo design](#5-monte-carlo-design)
6. [Fold rule and tuning defaults](#6-fold-rule-and-tuning-defaults)
7. [Running Monte Carlo simulations](#7-running-monte-carlo-simulations)
8. [Simulation outputs and tables](#8-simulation-outputs-and-tables)
9. [Rank-selection diagnostics](#9-rank-selection-diagnostics)
10. [Fold-retention diagnostics](#10-fold-retention-diagnostics)
11. [Resume-safe production runs](#11-resume-safe-production-runs)
12. [Empirical code status](#12-empirical-code-status)
13. [Reproducibility notes](#13-reproducibility-notes)
14. [Troubleshooting](#14-troubleshooting)
15. [Citation and license](#15-citation-and-license)

---

## 1. Repository goals

The package implements a low-rank heterogeneous dynamic-panel estimator with:

- rank-constrained first-stage least squares;
- scattered cross-fitting with forward space--time buffers;
- matrix-free Riesz/debiasing-weight construction;
- one-step debiased linear-target estimation;
- theorem-aligned diagonal and spatial-kernel standard errors in simulation;
- Monte Carlo DGPs with independent heteroskedastic errors, spatially dependent heteroskedastic errors, and predetermined covariates;
- rank-selection and retained-training-share diagnostics;
- resume-safe batch execution for large simulation grids.

The simulation code is designed so that the Monte Carlo section can report:

- true values;
- mean estimates;
- bias;
- RMSE;
- mean standard errors;
- empirical size;
- coverage;
- Monte Carlo standard errors for size and coverage;
- rank-selection frequencies;
- retained-training shares;
- fold-choice diagnostics.

---

## 2. Repository layout

```text
dlrhcs/
  design.py              Design map A and adjoint A*
  factorridge.py         Rank-constrained alternating least squares / factor-ridge fitting
  folds.py               Scattered folds and forward buffer construction
  targets.py             Target directions, tangent projections, Riesz weights
  ranks.py               Cross-fitted rank-selection criterion
  onestep.py             One-step estimator and standard-error routines
  pipeline.py            End-to-end estimator and finite-fold rule
  dgp.py                 Canonical Monte Carlo DGPs 1--3
  mc.py                  Monte Carlo replication and aggregation helpers
  empirical.py           Empirical application pipeline; currently under revision
  covariates.py          Empirical covariate loaders; currently under revision
  unemp.py               Unemployment data loader; currently under revision
  diagnostics.py         Diagnostic helpers
  spatial.py             Geographic distance tools for empirical spatial kernels
  report.py              Empirical reporting helpers

configs/
  pilot.json             Small/debug config
  fast.json              Medium/debug config
  full.json              Production config

scripts/
  run_mc_batches.py      Resume-safe Monte Carlo batch runner
  sim_report.py          Simulation CSV/LaTeX table generator
  stress_tests.py        Simulation stress-test runner; still under revision
  xs_stress.py           Spatial-dependence stress runner; still under revision
  fold_comparison.py     Fold/buffer comparison runner; still under revision
  zillow_abc.py          Housing empirical script; under revision
  unemp_abc.py           Unemployment empirical script; under revision
  spatial_kernel_se.py   Empirical spatial-kernel script; under revision

tests/
  test_core.py           Unit and smoke checks

outputs/
  sim/                   Simulation JSONL outputs and generated tables
```

Generated outputs are written under `outputs/`. Large output files are intended to be regenerated, not manually edited.

---

## 3. Installation

Use Python 3.10 or newer. On Windows, Git Bash or PowerShell both work.

### Git Bash / Unix-style shell

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the unit checks:

```bash
./.venv/Scripts/python.exe tests/test_core.py
```

For deterministic and stable parallel performance, it is often useful to restrict BLAS threading:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

In PowerShell:

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
```

---

## 4. Quick smoke checks

Run the core test suite:

```bash
./.venv/Scripts/python.exe tests/test_core.py
```

Run a tiny resume-safe Monte Carlo job:

```bash
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp1 --T 30 --N 30 --R-total 5 --batch-size 2 \
  --out-path outputs/sim/batch_smoke_dgp1_30.jsonl \
  --config configs/full.json --select false --fixed-ranks 1,1,1
```

Regenerate simulation tables:

```bash
./.venv/Scripts/python.exe scripts/sim_report.py
```

The table files are written to:

```text
outputs/sim/tables/tab_mc_performance.csv
outputs/sim/tables/tab_mc_performance.tex
outputs/sim/tables/tab_rank_frequency.csv
outputs/sim/tables/tab_rank_frequency.tex
outputs/sim/tables/tab_fold_retention.csv
outputs/sim/tables/tab_fold_retention.tex
```

---

## 5. Monte Carlo design

The simulation model has one lagged outcome and one scalar observed covariate:

\[
y_{it}
=
a_{it}y_{i,t-1}
+
\beta_{it}x_{it}
+
c_\xi(h_{it}+u_{it}),
\qquad
 i=1,\ldots,N,\quad t=-49,\ldots,0,1,\ldots,T.
\]

The initial condition is \(y_{i,-50}=0\). The first 50 periods are burn-in; estimation uses \(t=1,\ldots,T\).

### Coefficient matrices

The autoregression matrix \(A=(a_{it})\), slope matrix \(B=(\beta_{it})\), and interactive-effect matrix \(H=(h_{it})\) are rank one in the canonical DGP.

Raw autoregression coefficients are

\[
a_{it}^{raw}=\lambda_{a,i}f_{a,t}.
\]

Slope coefficients are

\[
\beta_{it}=\lambda_{b,i}f_{b,t}.
\]

For \(m\in\{a,b\}\),

\[
f_{m,t}=\mu_{f,m}+\kappa_{f,m}g_{m,t},
\]

where

\[
g_{m,t}
=
\rho_g g_{m,t-1}
+
(1-\rho_g^2)^{1/2}v_{m,t},
\qquad
v_{m,t}\overset{iid}{\sim}N(0,1),
\qquad
g_{m,-50}=0,
\]

with \(\rho_g=0.5\).

For the autoregression coefficients,

\[
\mu_{f,a}=0.5,
\qquad
\kappa_{f,a}=0.1,
\qquad
\lambda_{a,i}\overset{iid}{\sim}N(1,0.1^2).
\]

Stability is imposed by

\[
a_{it}=c_a a_{it}^{raw},
\qquad
c_a=\min\left\{1,\frac{0.85}{\max_{i,t}|a_{it}^{raw}|}\right\}.
\]

For the slope coefficients,

\[
\mu_{f,b}=0.6,
\qquad
\kappa_{f,b}=0.2,
\qquad
\lambda_{b,i}\overset{iid}{\sim}N(1,0.4^2).
\]

The interactive effect is

\[
h_{it}=c_h\lambda_{h,i}g_{h,t},
\qquad
\lambda_{h,i}\overset{iid}{\sim}N(0,1),
\]

where

\[
g_{h,t}
=
\rho_g g_{h,t-1}
+
(1-\rho_g^2)^{1/2}v_{h,t},
\qquad
v_{h,t}\overset{iid}{\sim}N(0,1),
\qquad
g_{h,-50}=0.
\]

The scale \(c_h\) is set to

\[
c_h=\left(\frac{0.3}{0.7}\right)^{1/2}\approx 0.655,
\]

so that the interactive effect contributes 30 percent of the variance of the composite disturbance in the population normalization.

### DGP 1: independent heteroskedastic errors

\[
u_{it}=\sigma_i\varepsilon_{it},
\qquad
\varepsilon_{it}\overset{iid}{\sim}N(0,1),
\qquad
\sigma_i^2\overset{iid}{\sim}U(0.5,1.5).
\]

### DGP 2: spatially dependent heteroskedastic errors

Units are placed on a one-dimensional lattice with distance

\[
d_N(i,j)=|i-j|.
\]

The idiosyncratic error vector is generated as

\[
u_t=D_\sigma R_N^{1/2}\varepsilon_t,
\]

where \(D_\sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_N)\) and

\[
(R_N)_{ij}=\rho_s^{|i-j|},
\qquad
\rho_s=0.5.
\]

The implementation uses an equivalent AR(1) recursive construction along the lattice, giving

\[
\operatorname{Cov}(u_{it},u_{jt}\mid\sigma_1,\ldots,\sigma_N)
=
\sigma_i\sigma_j\rho_s^{|i-j|}.
\]

### DGP 3: predetermined covariates

DGP 3 uses the same error process as DGP 2, but the covariate depends on lagged idiosyncratic shocks.

For DGP 1 and DGP 2,

\[
x_{it}
=
\rho_x x_{i,t-1}
+
\delta_x\lambda_{x,i}f_{x,t}
+
(1-\rho_x^2)^{1/2}e_{it}.
\]

For DGP 3,

\[
x_{it}
=
\rho_x x_{i,t-1}
+
\delta_x\lambda_{x,i}f_{x,t}
+
\eta_x u_{i,t-1}
+
(1-\rho_x^2)^{1/2}e_{it}.
\]

The parameter values are

\[
\rho_x=0.5,
\qquad
\delta_x=0.5,
\qquad
\eta_x=0.3,
\]

\[
\lambda_{x,i}\overset{iid}{\sim}N(0,1),
\qquad
e_{it}\overset{iid}{\sim}N(0,\sigma_{e,i}^2),
\qquad
\sigma_{e,i}^2\overset{iid}{\sim}U(0.5,1.5).
\]

The common factor in \(x_{it}\) follows

\[
f_{x,t}
=
\rho_{fx}f_{x,t-1}
+
(1-\rho_{fx}^2)^{1/2}v_{x,t},
\qquad
v_{x,t}\overset{iid}{\sim}N(0,1),
\qquad
f_{x,-50}=0,
\]

with \(\rho_{fx}=0.5\). The initial covariate is \(x_{i,-50}=0\).

### \(c_\xi\) calibration

The scalar \(c_\xi\) controls the overall signal-to-noise ratio. For each DGP and panel size, the code calibrates one fixed value of \(c_\xi\) using 100 deterministic calibration draws. The calibrated value solves

\[
\frac{1}{100}\sum_{k=1}^{100}PR_k^2(c_\xi)=0.5.
\]

That value is then held fixed across ordinary Monte Carlo replications for that DGP and panel size.

Each returned panel stores:

```text
c_a
c_h
c_xi
PR2_target
PR2_realized
PR2_calibration_mean
PR2_calibration_std
c_xi_calibration_draws
a_it_summary
beta_it_summary
max_abs_a_it
```

The estimator receives the model-scale matrices

\[
A_{it}=a_{it},
\qquad
B_{it}=\beta_{it},
\qquad
H_{it}=c_\xi h_{it},
\]

and the model-scale innovation

\[
\varepsilon_{it}^{model}=c_\xi u_{it}.
\]

---

## 6. Fold rule and tuning defaults

The asymptotic theory uses a slowly growing number of folds. The implementation uses a finite-sample floor rule:

\[
J_{TN}
=
\max\left\{
J_{\min},
\left\lceil c_J B_{TN}L^J_{TN}\right\rceil
\right\},
\]

where

\[
n_{\mathrm{eff}}=\frac{TN}{T+N},
\qquad
L^J_{TN}=\max\{1,\log\log n_{\mathrm{eff}}\}.
\]

For the lattice buffer used in simulation,

\[
B_{TN}=(q+1)(2r+1).
\]

The current production defaults are:

```text
J_min = 10
c_J = 1.0
kappa_c = 0.015
c_xi_calibration_draws = 100
q = 2
r = 0
```

The robustness grids include:

```text
J_min in {5, 6, 8, 10, 12}
kappa_c in {0.015, 0.03}
```

The finite fold count should be interpreted as the finite-sample floor of a growing theorem-admissible sequence, not as a separate fixed-\(J\) asymptotic theorem.

---

## 7. Running Monte Carlo simulations

### Fixed-rank performance runs

Fixed-rank runs estimate the model with the true rank vector \((1,1,1)\). These runs populate the main performance table.

Example:

```bash
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp1 --T 100 --N 100 --R-total 1000 --batch-size 25 \
  --out-path outputs/sim/grid_dgp1_100.jsonl \
  --config configs/full.json --select false --fixed-ranks 1,1,1
```

Run the main grid:

```bash
for dgp in dgp1 dgp2 dgp3; do
  for n in 50 100 200 400; do
    ./.venv/Scripts/python.exe scripts/run_mc_batches.py \
      --dgp-type "$dgp" --T "$n" --N "$n" --R-total 1000 --batch-size 25 \
      --out-path "outputs/sim/grid_${dgp}_${n}.jsonl" \
      --config configs/full.json --select false --fixed-ranks 1,1,1
  done
done
```

### Rank-selection runs

Rank-selection runs allow the cross-fitted rank criterion to choose the rank. The true rank is passed only for diagnostic scoring.

```bash
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp3 --T 100 --N 100 --R-total 1000 --batch-size 25 \
  --out-path outputs/sim/grid_rank_dgp3_100.jsonl \
  --config configs/full.json --select true --true-ranks 1,1,1 --rank-caps 1,1,1
```

Run the rank-selection grid:

```bash
for dgp in dgp1 dgp2 dgp3; do
  for n in 50 100 200; do
    ./.venv/Scripts/python.exe scripts/run_mc_batches.py \
      --dgp-type "$dgp" --T "$n" --N "$n" --R-total 1000 --batch-size 25 \
      --out-path "outputs/sim/grid_rank_${dgp}_${n}.jsonl" \
      --config configs/full.json --select true --true-ranks 1,1,1 --rank-caps 1,1,1
  done
done
```

### Fold-floor robustness runs

Use these to support the \(J_{\min}=10\) baseline.

```bash
for jmin in 5 6 8 10 12; do
  ./.venv/Scripts/python.exe scripts/run_mc_batches.py \
    --dgp-type dgp1 --T 100 --N 100 --R-total 1000 --batch-size 25 \
    --out-path "outputs/sim/fold_floor_dgp1_100_Jmin${jmin}.jsonl" \
    --config configs/full.json --select false --fixed-ranks 1,1,1 --J-min "$jmin"
done
```

---

## 8. Simulation outputs and tables

Regenerate all simulation tables with:

```bash
./.venv/Scripts/python.exe scripts/sim_report.py
```

The table files are:

```text
outputs/sim/tables/tab_mc_performance.csv
outputs/sim/tables/tab_mc_performance.tex

outputs/sim/tables/tab_rank_frequency.csv
outputs/sim/tables/tab_rank_frequency.tex

outputs/sim/tables/tab_fold_retention.csv
outputs/sim/tables/tab_fold_retention.tex
```

The main Monte Carlo performance table reports:

```text
DGP
target
T
N
true value
mean estimate
bias
RMSE
mean s.e.
empirical size
coverage
Monte Carlo s.e. for size/coverage
replications
```

DGP 1 uses the diagonal cell-heteroskedastic standard error. DGP 2 and DGP 3 use the Bartlett spatial-kernel standard error on the lattice.

The simulation tables do not report by-period clustered standard errors.

---

## 9. Rank-selection diagnostics

Rank-selection output is generated only when `--select true` is used. Fixed-rank runs are not treated as rank-selection evidence.

Rank-selection table columns include:

```text
DGP
T
N
J_min
kappa_c
retained_nonvalidation
P(correct rank)
P(underfit)
P(overfit)
modal selected rank
replications
```

Use `--true-ranks` to record the true rank for diagnostics. Use `--fixed-ranks` only when `--select false`.

Correct usage:

```bash
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp1 --T 100 --N 100 --R-total 1000 --batch-size 25 \
  --out-path outputs/sim/grid_rank_dgp1_100.jsonl \
  --config configs/full.json --select true --true-ranks 1,1,1 --rank-caps 1,1,1
```

Incorrect usage:

```bash
# This is rejected because fixed-ranks would fix the estimator and therefore cannot test rank selection.
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp1 --T 100 --N 100 --R-total 1000 --batch-size 25 \
  --out-path outputs/sim/grid_rank_dgp1_100.jsonl \
  --config configs/full.json --select true --fixed-ranks 1,1,1
```

---

## 10. Fold-retention diagnostics

For each fold \(j\), the code computes:

\[
\tau_{\mathrm{tot},j}
=
\frac{|\mathcal I_{-j}^{buf}|}{TN},
\]

and

\[
\tau_{\mathrm{nv},j}
=
\frac{|\mathcal I_{-j}^{buf}|}{TN-|\mathcal I_j|}.
\]

The reported retained share is

\[
\tau_{\mathrm{nv}}
=
\frac{1}{J}\sum_{j=1}^J\tau_{\mathrm{nv},j}.
\]

The code also reports

```text
retained_total
retained_nonvalidation
retained_nonvalidation_min
retained_nonvalidation_max
validation_fold_size_mean
validation_fold_share_mean
J_realized
J_rule_term
J_manual_override
```

The retained nonvalidation share is the main diagnostic used to support the fold floor.

---

## 11. Resume-safe production runs

`scripts/run_mc_batches.py` writes one JSON record per replication and can safely resume interrupted runs.

If a run stops, rerun the exact same command. Completed `rep_id`s are skipped.

Each run writes:

```text
outputs/sim/<name>.jsonl
outputs/sim/<name>.meta.json
```

The metadata sidecar records:

```text
dgp_type
T
N
R_total
completed_R
J_min
kappa_c
c_xi_calibration_draws
fixed_ranks or true_ranks
select mode
```

Before launching the full grid, run one production file first:

```bash
./.venv/Scripts/python.exe scripts/run_mc_batches.py \
  --dgp-type dgp1 --T 100 --N 100 --R-total 1000 --batch-size 25 \
  --out-path outputs/sim/grid_dgp1_100.jsonl \
  --config configs/full.json --select false --fixed-ranks 1,1,1
```

Then regenerate tables and inspect the output:

```bash
./.venv/Scripts/python.exe scripts/sim_report.py
```

---

## 12. Empirical code status

The empirical pipeline remains in the repository, but it is under revision and is not documented as final in this README.

Known empirical tasks still to be finalized include:

- removing or disabling interpolation unless explicitly used as a sensitivity option;
- verifying seasonal adjustment for unemployment and housing series;
- documenting raw data download and construction steps;
- defining housing top/bottom tiers clearly;
- resolving and documenting CBSA centroid matching;
- adding outlier diagnostics;
- adding companion-radius stability diagnostics;
- adding known cross-sectional cluster-score standard errors;
- revising empirical tables and text after the data pipeline is finalized.

Until those tasks are complete, do not treat empirical outputs as submission-final.

---

## 13. Reproducibility notes

### Randomness

Simulation replications use deterministic seeds. The \(c_\xi\) calibration is deterministic for a given DGP, panel size, and calibration configuration.

### Resume behavior

Batch outputs are JSONL files. Each line is one replication. If a run is interrupted, rerun the same command; existing replications are skipped.

### Manual fold override

Manual fold override is preserved for debugging, but production defaults use the finite-fold-floor rule.

### Rank-selection versus fixed-rank runs

Fixed-rank runs and rank-selection runs are conceptually different and should be kept in separate files.

- Use `--select false --fixed-ranks 1,1,1` for performance tables.
- Use `--select true --true-ranks 1,1,1 --rank-caps 1,1,1` for rank-selection frequency tables.

---

## 14. Troubleshooting

### No progress appears in the terminal

The batch runner may print only after a batch or replication depending on the current script version. Check whether output is being written:

```bash
wc -l outputs/sim/grid_dgp1_100.jsonl
```

or inspect the metadata:

```bash
cat outputs/sim/grid_dgp1_100.meta.json
```

If you stop the job, rerun the same command to resume.

### A table has missing fields

This usually means the JSONL file was generated before the schema update. Regenerate the simulation file under the current code.

### Rank-frequency table is empty

Rank-frequency rows are produced only for `--select true` runs. Fixed-rank runs are correctly excluded.

### DGP 1 shows spatial-kernel columns

This should not happen in the current schema. DGP 1 should report diagonal/white inference only. Regenerate the output file if it was created before the SE-routing update.

### Period-cluster columns appear in simulation tables

This indicates an old output or old reporting script. Current simulation tables should not report period-cluster standard errors.

---

## 15. Citation and license

If you use this code, please cite the paper once available. The code is released under the MIT License unless otherwise specified in `LICENSE`.

The empirical data sources remain subject to their providers' terms. The empirical section of this README will be updated after the empirical pipeline is finalized.
