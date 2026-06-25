#!/usr/bin/env python3
"""Zillow A/B/C specifications (Phase 2 robustness for the housing application).

  A. Main baseline       2000-2026, no covariates           (headline, full T)
  B. Restricted baseline 2005-2024, no covariates           (sample-restriction check)
  C. Covariate-augmented 2005-2024, permits+population+GDP  (local-fundamentals check)

A vs B isolates the effect of the shorter sample; B vs C isolates the effect of the
covariates on the SAME sample.  Writes outputs/empirical/zillow_abc.json and prints
the comparison table.  Run from the repo root:  python scripts/zillow_abc.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.empirical import load_zillow, run_ar2          # noqa: E402
from dlrhcs.covariates import load_zillow_covariates       # noqa: E402
from dlrhcs.pipeline import Tuning                          # noqa: E402

DATA = os.path.join(ROOT, "data")
ZT = os.path.join(DATA, "zillow_metro_top.csv")
ZB = os.path.join(DATA, "zillow_metro_bottom.csv")
COV = os.path.join(DATA, "covariates", "metro_monthly_covariates_2000_present.csv")
XW = os.path.join(DATA, "covariates", "cbsa_county_crosswalk_2023.csv")
SEED = 7


def run_spec(label, start, end, with_cov, n_jobs=1):
    z = load_zillow(ZT, ZB, start=start, end=end)
    Y, tier = z["Y"], z["tier"]
    covars, covar_names, ranks = None, (), (1, 1, 1)
    if with_cov:
        mats, names, matched = load_zillow_covariates(list(z["col_region"]), z["months"], COV, XW)
        Y, tier = Y[:, matched], tier[matched]
        covars = [m[:, matched] for m in mats]
        covar_names = names
        ranks = (1, 1, 1, 1, 1, 1)             # lag1, lag2, 3 covariates, H (all rank-1)
    tun = Tuning(ranks=ranks, q=1, J=6, ridge=0.1, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-6, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=n_jobs)
    r = run_ar2(Y, tun, groups=tier, group_labels=("top", "bottom"),
                rng=np.random.default_rng(SEED), covars=covars, covar_names=covar_names)
    r["spec"] = label
    r["sample"] = f"{start or '2000-01'}..{end or 'latest'}"
    r["N"], r["T"] = int(Y.shape[1]), int(Y.shape[0])
    return r


def main():
    nj = int(os.environ.get("N_JOBS", "1"))
    specs = {
        "A": run_spec("A_main", None, None, False, nj),
        "B": run_spec("B_restricted", "2005-01", "2024-12", False, nj),
        "C": run_spec("C_covariates", "2005-01", "2024-12", True, nj),
    }
    os.makedirs(os.path.join(ROOT, "outputs", "empirical"), exist_ok=True)
    json.dump(specs, open(os.path.join(ROOT, "outputs", "empirical", "zillow_abc.json"), "w"),
              indent=2, default=str)
    print(f"\n{'spec':14s}{'sample':18s}{'N':>5}{'T':>5}{'lag1':>8}{'lag2':>8}{'cum':>8}{'radius':>8}")
    for k in ("A", "B", "C"):
        r = specs[k]; t = r["targets"]; d = r["derived"]
        print(f"{r['spec']:14s}{r['sample']:18s}{r['N']:>5}{r['T']:>5}"
              f"{t['lag1_mean']['est']:>8.3f}{t['lag2_mean']['est']:>8.3f}"
              f"{d['cumulative_persistence']['est']:>8.3f}{d['companion_radius']:>8.3f}")
    print("\nSpec C covariate coefficients (mean, both s.e.):")
    for nm in ("permits", "population", "gdp"):
        t = specs["C"]["targets"][nm + "_mean"]
        print(f"  {nm:11s}: {t['est']:+.4f}  (White {t['se']:.4f}, cross-sec {t['se_xs']:.4f})")


if __name__ == "__main__":
    main()
