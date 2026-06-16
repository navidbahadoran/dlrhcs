# Validation record

Build machine: NumPy 2.2 / SciPy 1.15 / Python 3.10, single-thread BLAS.
Small-scale validation runs; publication-scale numbers come from
`run_all.py --config configs/full.json`. See `THEOREM_MAP.md` for the full
result-to-experiment correspondence.

## Spec section-15 unit tests — 7/7 pass  (`python tests/test_core.py`)
adjoint identity (`prop:adjoint`); forward-exclusion indexing; ALS monotonicity;
tangent projector idempotent/self-adjoint; Riesz representer identity
(`prop:fold_riesz`); noiseless exact recovery; Gram-scale convention
(`prop:fold_transport_sufficient`).

## Theorem-justification experiments (small-R validation)

| Result | Experiment | Outcome |
|---|---|---|
| `thm:oracle_clt` | oracle MC (79,79), R=80 | mean coverage **0.952** (band [0.93,0.96]) |
| `thm:rank_consistency` | rank_consistency | P(r_hat=truth) **0.60 -> 1.00** from (50,50) to (120,120) |
| `cor:irf_body`/`thm:irf` | irf_lrm_coverage (oracle, 80,80) | IRF1/2/4 & LRM coverage 0.89–0.94 (R=18) |
| `thm:xs_dependence` | xs_coverage (70,70) | lag full mean White **0.83** vs xs **0.89** (R=18) |
| `lem:local_collinearity_singularity` | contiguous_fold_singular | min-eig scatter 1.6e-2 vs **contiguous ~0** (cond 3.7e12) |
| `thm:feasible` (debiasing) | debiasing_demo | one-step correction non-trivial (e.g. plug-in 0.320 -> debiased 0.200) |

All five experiments reproduce the qualitative prediction at small R; the trends
(coverage -> nominal, P(correct rank) -> 1, White-under/xs-correct, contiguous
singular) sharpen at full R.

## Empirical (real data, illustration of the feasible theorems)
The Zillow AR(2) application runs end-to-end: strong lag structure, tight valid
CIs, and stationary dynamics (companion radius 0.63). A metro-unemployment
application is implemented and parked (not in th