#!/usr/bin/env python3
"""Fold-design comparison (referee add-on): validate that the scattered, space-time-
buffered cross-fitting is necessary.  At a fixed panel size, under the decaying
cross-sectional-dependence DGP, run the Monte Carlo with four fold configurations and
report coverage and RMSE of the lag full mean (the hardest target), plus the
conditioning of the buffered training information map (scattered vs contiguous):

  scatter, q=1, r=1   full space-time buffer  (the paper's design)
  scatter, q=1, r=0   temporal-only buffer
  scatter, q=0, r=0   no buffer
  contiguous, q=1     contiguous time blocks  (negative control)

Resume-safe (JSONL checkpoints in outputs/sim/).  Run from the repo root:
    python scripts/fold_comparison.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.mc import aggregate, run_grid                       # noqa: E402
from dlrhcs.experiments import contiguous_fold_singular         # noqa: E402
from dlrhcs.pipeline import Tuning                              # noqa: E402

SIM = os.path.join(ROOT, "outputs", "sim")
SEED = 2024
TP, N, R = 100, 100, 400      # fixed panel; raise R for the final run if compute allows


def _base_tuning():
    t = json.load(open(os.path.join(ROOT, "configs", "full.json")))["tuning"]
    t["ranks"] = tuple(t["ranks"])
    return t


def main():
    nj = int(os.environ.get("N_JOBS", "-1") or -1)
    base = _base_tuning()
    configs = [("scatter_q1_r1", "scatter", 1, 1),
               ("scatter_q1_r0", "scatter", 1, 0),
               ("scatter_q0_r0", "scatter", 0, 0),
               ("contiguous_q1", "contiguous", 1, 0)]
    dgp = dict(noise="xs")
    os.makedirs(SIM, exist_ok=True)
    rows = {}
    for name, scheme, q, r in configs:
        tun = Tuning(**{**base, "scheme": scheme, "q": q, "buffer_r": r})
        path = os.path.join(SIM, f"foldcmp_{name}.jsonl")
        print(f"[fold-cmp] {name}: scheme={scheme} q={q} r={r}", flush=True)
        try:
            run_grid(TP, N, R, tun, path, dgp_kwargs=dgp, master=SEED, n_jobs=nj)
            a = aggregate(path)
            lf, sf = a["lag_fmean"], a["slope_fmean"]
            rows[name] = dict(lag_cov=lf["cov"], lag_cov_xs=lf["cov_xs"], lag_rmse=lf["rmse"],
                              slope_cov=sf["cov"], retained=a["_meta"]["retained"])
        except Exception as e:                       # contiguous can be singular
            rows[name] = dict(failed=repr(e)[:120])
            print(f"   {name} FAILED (expected for a singular design): {repr(e)[:80]}", flush=True)
    cond = contiguous_fold_singular(TP, N, Tuning(**{**base, "q": 1, "buffer_r": 1}), master=SEED)
    json.dump(dict(panel=[TP, N], R=R, configs=rows, conditioning=cond),
              open(os.path.join(SIM, "fold_comparison.json"), "w"), indent=2, default=str)
    print(f"\n{'config':16}{'lag cov':>9}{'lag xs cov':>11}{'lag RMSE':>10}{'slope cov':>11}{'retained':>10}")
    for name, *_ in configs:
        r = rows[name]
        if "failed" in r:
            print(f"{name:16}{'-- singular / failed --':>51}")
        else:
            print(f"{name:16}{r['lag_cov']:>9.3f}{r['lag_cov_xs']:>11.3f}{r['lag_rmse']:>10.4f}"
                  f"{r['slope_cov']:>11.3f}{r['retained']:>10.3f}")
    print("\nTraining-map conditioning (scattered vs contiguous):")
    for k, v in cond.items():
        print(f"  {k:12s} min_eig={v['min_eig']:+.2e} cond={v['cond']:.2e} min_row_support={v['min_row_support']}")


if __name__ == "__main__":
    main()
