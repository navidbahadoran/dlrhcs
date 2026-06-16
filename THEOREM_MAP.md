# Theorem -> experiment map

Each paper result is justified by a specific, reproducible experiment in this
package. "Status" reflects validation runs on the build machine (small scale);
the publication-scale runs come from `run_all.py --config configs/full.json`.
Run the suite with `--stage theorems` (plus `oracle`, `grid`, `purge`).

| Paper label | Statement (short) | Experiment / test | Output | Validation |
|---|---|---|---|---|
| `prop:adjoint` | Adjoint & Gram identity `<A x,R>=<x,A* R>` | `tests/test_core.py::test_adjoint_identity` | — | PASS (machine precision) |
| `prop:fold_riesz` | Purged-training Riesz representer identity | `tests/test_core.py::test_riesz_identity` | — | PASS |
| `prop:fold_transport_sufficient` | Normalized design Gram transports across folds | `tests/test_core.py::test_gram_scale` | — | PASS |
| `lem:local_collinearity_singularity` | Contiguous folds -> singular information map | `experiments.contiguous_fold_singular` | `outputs/theorems.json` | min-eig: scatter 1.6e-2 vs contiguous ~0 (cond 3.7e12); held-out rows lose all training support |
| `thm:oracle_clt` | Benchmark CLT (true tangent spaces) | `mc.run_grid(oracle=True)` | `outputs/sim/oracle_*.json` | mean coverage 0.952 at (79,79), R=80 |
| `lem:rank_risk` / `thm:rank_consistency` | CV risk expansion; selector picks the truth, P->1 | `experiments.rank_consistency` | `outputs/theorems.json` | P(r_hat=truth): 0.60 (50,50) -> 1.00 (120,120) |
| `thm:first_stage` | Post-exclusion first-stage rates | oracle-vs-feasible gap in `mc.run_grid` | `outputs/sim/*` | feasible -> oracle as (T,N) grow |
| `thm:riesz_replacement` | Feasible debiasing-weight replacement | feasible vs oracle coverage | `outputs/sim/*` | gap shrinks with size |
| `thm:feasible` | Feasible studentized CLT, t -> N(0,1); debiasing removes bias | `mc.run_grid` (convergence) + `experiments.debiasing_demo` | `outputs/sim/grid_*.json`, `outputs/theorems.json` | one-step correction non-trivial (plug-in 0.320 -> debiased 0.200); coverage -> 0.95 |
| `cor:means_contrasts` / `thm:means` / `thm:contrast` | Group/full-mean & between-group contrast inference | mean & contrast targets in `mc.run_grid` | `outputs/sim/grid_*.json` | coverage near nominal in oracle; full grid for feasible |
| `prop:group_mean_scale` | sqrt(g_N) contraction of mean targets | `report.precision_table` | `outputs/tables/tab_sim_precision.tex` | SE ratio across the doubling grid |
| `cor:irf_body` / `thm:irf` / `lem:joint_clt` | Delta-method IRF & LRM CLT | `experiments.irf_lrm_coverage` | `outputs/theorems.json` | IRF1/2/4 & LRM coverage ~0.89-0.94 (R=18) |
| `ass:dependent` / `thm:xs_dependence` | xs s.e. valid under within-period dependence | `experiments.xs_coverage` | `outputs/theorems.json` | lag full mean: White 0.83 vs xs 0.89 (R=18) |
| `ass:purged_folds` (+ leakage) | Forward exclusion removes dynamic leakage | `mc.run_purge_sweep` | `outputs/sim/purge_*.json`, `tab_sim_purge.tex` | coverage anti-conservative at q=0, nominal at moderate q |
| `ass:dynamic`/`ass:signal`/`ass:image_no_collinearity`/`ass:strong_factor_dynamic` | Maintained assumptions hold in the DGP | `dgp.simulate` construction (+ coherence, stability checks) | — | enforced by construction; coherence O(1), companion radius < 1 |

## Empirical illustration (real-data evidence for the *feasible* theorems)

The Zillow AR(2) application (`dlrhcs/empirical.py`, `--stage
empirical`) shows the feasible machinery operating on a genuine large panel:
debiased estimates with a non-trivial one-step correction, **both** White
(`thm:feasible`) and within-period (`thm:xs_dependence`) standard errors,
stationary estimated dynamics (companion radius < 1), delta-method IRF/LRM
(`cor:irf_body`), and robustness across the nuisance rank. They are illustrations
of the theory in the wild, not matches to any pre-printed number.

## How to reproduce

```bash
python run_all.py --config configs/full.json --stage oracle      # thm:oracle_clt
python run_all.py --config configs/full.json --stage grid        # thm:feasible, means/contrasts
python run_all.py --config configs/full.json --stage purge       # forward-exclusion / leakage
python run_all.py --config configs/full.json --stage theorems    # rank consistency, IRF/LRM, xs, contiguous control
python run_all.py --config configs/full.json --stage empirical   # real-data illustration
python run_all.py --config configs/full.json --stage tables      # LaTeX tables
```
