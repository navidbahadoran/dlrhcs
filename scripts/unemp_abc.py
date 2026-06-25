#!/usr/bin/env python3
"""Unemployment A/B/C specifications (Phase 2 robustness for the labor application).

  A. Main baseline       2000-2026, no covariate            (headline, full T)
  B. Restricted baseline 2001-2025, no covariate            (sample-restriction check)
  C. Covariate-augmented 2001-2025, CES payroll growth      (labor-demand check)

Self-contained monthly panel (NSA unemployment rate, deseasonalized via month-of-year
dummies; payroll covariate predetermined).  Writes outputs/empirical/unemp_abc.json
and prints the comparison.  Run from repo root:  python scripts/unemp_abc.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.unemp import load_unemp_panel                  # noqa: E402
from dlrhcs.empirical import run_ar2                       # noqa: E402
from dlrhcs.pipeline import Tuning                          # noqa: E402

PANEL = os.path.join(ROOT, "data", "unemp",
                     "unemployment_metro_model_panel_bls_only_name_matched.csv")
SEED = 13


def run_spec(label, start, end, with_cov, require_cov, n_jobs=1):
    d = load_unemp_panel(PANEL, start=start, end=end, require_cov=require_cov)
    Y = d["Y"]
    g = (d["mean_level"] > np.median(d["mean_level"])).astype(int)
    covars, covar_names = None, ()
    ranks = (1, 0, 3)                                 # AR(1): 2nd lag unidentified at monthly freq
    #                                                (near-integrated); matches annual selection (1,0,1)
    if with_cov:
        covars = [d["payroll"]]
        covar_names = ["payroll"]
        ranks = (1, 0, 1, 3)                          # AR(1) + payroll (rank1)
    tun = Tuning(ranks=ranks, q=1, J=6, ridge=0.5, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-5, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=n_jobs)
    r = run_ar2(Y, tun, groups=g, group_labels=("hi_unemp", "lo_unemp"),
                rng=np.random.default_rng(SEED), covars=covars, covar_names=covar_names)
    r["spec"] = label
    r["sample"] = f"{start}..{end}"
    r["N"], r["T"] = int(Y.shape[1]), int(Y.shape[0])
    return r


def main():
    nj = int(os.environ.get("N_JOBS", "1"))
    specs = {
        "A": run_spec("A_main", "2000-01", "2026-05", False, False, nj),
        "B": run_spec("B_restricted", "2001-01", "2025-12", False, True, nj),
        "C": run_spec("C_covariates", "2001-01", "2025-12", True, True, nj),
    }
    os.makedirs(os.path.join(ROOT, "outputs", "empirical"), exist_ok=True)
    json.dump(specs, open(os.path.join(ROOT, "outputs", "empirical", "unemp_abc.json"), "w"),
              indent=2, default=str)
    print(f"\n{'spec':14s}{'sample':20s}{'N':>5}{'T':>5}{'lag1':>8}{'lag2':>8}{'cum':>8}{'radius':>8}")
    for k in ("A", "B", "C"):
        r = specs[k]; t = r["targets"]; d = r["derived"]
        print(f"{r['spec']:14s}{r['sample']:20s}{r['N']:>5}{r['T']:>5}"
              f"{t['lag1_mean']['est']:>8.3f}{t['lag2_mean']['est']:>8.3f}"
              f"{d['cumulative_persistence']['est']:>8.3f}{d['companion_radius']:>8.3f}")
    t = specs["C"]["targets"]["payroll_mean"]
    print(f"\nSpec C payroll coefficient (mean): {t['est']:+.4f}  (White {t['se']:.4f}, cross-sec {t['se_xs']:.4f})")


if __name__ == "__main__":
    main()
