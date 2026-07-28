# Housing Riesz Pilot Report

These pilots use only the three existing housing mean targets and deterministic first-N-MSA/date subsets. They do not run the full N=169 production panel.

## Solver-status convention

SciPy `cg` returns `info == 0` on successful convergence to the requested tolerance, `info > 0` when the requested tolerance is not achieved within the iteration limit, and `info < 0` for illegal input or breakdown. The housing diagnostics now define convergence using both solver status and the achieved relative residual; iteration count alone is not treated as failure.

## Diagnosis

The original smoke convergence fraction of 0 is a genuine Riesz nonconvergence result, not a status-accounting bug. In the N=20, usable-T=60 baseline pilot, all 30 solves hit maxiter=600 and the median achieved relative residual is 1.55e-3, far above the requested 1e-5. Larger geometries show the same pattern under maxiter=600, with median relative residuals 1.88e-2 for N=50,T=100 and 4.78e-2 for N=50,T=196.

Increasing only the Riesz maxiter to 2000 yields convergence fraction 1.0 in all three geometries. The maximum iterations used are 865, 1434, and 1304, respectively. Median achieved relative residuals are about 8e-6 to 9e-6, consistent with the requested 1e-5 tolerance. No nonfinite Riesz diagnostics were recorded.

Changing the Riesz ridge from 1e-6 to 1e-5 also converges faster, but materially changes estimates and standard errors. Relative to the matched baseline, ridge=1e-5 moves estimates by up to about 0.044 in the tiny pilot, 0.026 in the N=50,T=100 pilot, and 0.040 in the N=50 full-time pilot. That is a substantive regularization change, not merely a solver-completion change.

The first-stage diagnostics are stable across matched Riesz settings because the first-stage tuning and data are unchanged. Objective paths remain monotone. The first-stage sweep-cap rate is 1.0 in these pilots, but final relative objective decreases are small, about 1.5e-5 to 2.6e-5, so this should be reported as a sweep-cap/resource diagnostic rather than confused with the Riesz CG nonconvergence issue.

## Recommended production Riesz setting

The smallest defensible change is to keep the current Riesz tolerance and ridge, and increase only the Riesz maxiter:

```bash
python scripts/housing_all_homes.py --repo-root . --config configs/full.json --seed 2024 --n-jobs 1 --riesz-maxiter 2000
```

Run production only after the preflight passes. The corresponding no-estimation preflight command is:

```bash
python scripts/housing_all_homes.py --repo-root . --preflight --panel-id start_2010 --config configs/full.json --riesz-maxiter 2000
```

## Output files

- `tab_housing_riesz_pilots.csv`
- `tab_housing_riesz_pilots.tex`
- one `riesz_diagnostics.csv` and `riesz_summary.json` per pilot cell

Production estimation was not launched.
