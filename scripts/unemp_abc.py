#!/usr/bin/env python3
"""Unemployment A/B/C specifications (Phase 2 robustness for the labor application).

  A. Main baseline       2000-2026, no covariate                 (headline, full T)
  B. Restricted baseline 2005-2024, no covariate, matched metros (sample-restriction check)
  C. Covariate-augmented 2005-2024, population + real-GDP growth (local-fundamentals check)

Unemployment is a heterogeneous AR(1) -- the second lag is not identified at monthly
frequency (the rate is near-integrated), matching the annual selection (1,0,1).  The
rate is NSA, deseasonalized via month-of-year dummies (level-preserving).  Covariates
are CBSA population/GDP growth (NOT employment, which is a labor-market identity),
matched by ces_cbsa_code, predetermined, winsorized + standardized.  B and C use the
SAME covariate-matched metros so B->C isolates the covariate.  Run from repo root:
    python scripts/unemp_abc.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.unemp import load_unemp_panel                  # noqa: E402
from dlrhcs.covariates import load_cbsa_covariates         # noqa: E402
from dlrhcs.empirical import run_ar2                       # noqa: E402
from dlrhcs.pipeline import Tuning                          # noqa: E402

PANEL = os.path.join(ROOT, "data", "unemp",
                     "unemployment_metro_model_panel_bls_only_name_matched.csv")
COV = os.path.join(ROOT, "data", "zillow", "metro_monthly_covariates_2000_present.csv")
SEED = 13


def run_spec(label, start, end, mode, n_jobs=1):
    # mode: "full" | "matched_nocov" | "matched_cov"
    d = load_unemp_panel(PANEL, start=start, end=end, require_cov=False)
    Y, ml = d["Y"], d["mean_level"]
    covars, covar_names, ranks = None, (), (1, 0, 3)       # AR(1): lag2 dropped, H rank 3
    if mode in ("matched_nocov", "matched_cov"):
        mats, names, matched = load_cbsa_covariates(d["ces"], d["months"], COV)
        Y, ml = Y[:, matched], ml[matched]
        if mode == "matched_cov":
            covars = [m[:, matched] for m in mats]
            covar_names = names
            ranks = (1, 0, 1, 1, 3)                        # lag1, lag2(drop), pop, gdp, H
    g = (ml > np.median(ml)).astype(int)
    tun = Tuning(ranks=ranks, q=1, J=6, ridge=0.5, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-4, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=n_jobs)
    r = run_ar2(Y, tun, groups=g, group_labels=("hi_unemp", "lo_unemp"),
                rng=np.random.default_rng(SEED), covars=covars, covar_names=covar_names)
    r["spec"] = label
    r["sample"] = f"{start}..{end}"
    r["N"], r["T"] = int(Y.shape[1]), int(Y.shape[0])
    return r


def main():
    nj = int(os.environ.get("N_JOBS", "1"))
    plan = [("A", "A_main", "2000-01", "2026-05", "full"),
            ("B", "B_restricted", "2005-01", "2024-12", "matched_nocov"),
            ("C", "C_covariates", "2005-01", "2024-12", "matched_cov")]
    outdir = os.path.join(ROOT, "outputs", "empirical")
    os.makedirs(outdir, exist_ok=True)
    specs = {}
    for k, lab, a, b, mode in plan:
        r = run_spec(lab, a, b, mode, nj)
        specs[k] = r
        json.dump(r, open(os.path.join(outdir, f"unemp_{k}.json"), "w"), indent=2, default=str)
        d = r["derived"]; t = r["targets"]
        print(f"[done {k}] {lab} N={r['N']} T={r['T']} "
              f"lag1={t['lag1_mean']['est']:.3f} cum={d['cumulative_persistence']['est']:.3f} "
              f"radius={d['companion_radius']:.3f}", flush=True)
    json.dump(specs, open(os.path.join(outdir, "unemp_abc.json"), "w"), indent=2, default=str)
    print(f"\n{'spec':14s}{'sample':20s}{'N':>5}{'T':>5}{'lag1':>8}{'lag2':>8}{'cum':>8}{'radius':>8}")
    for k in ("A", "B", "C"):
        r = specs[k]; t = r["targets"]; d = r["derived"]
        print(f"{r['spec']:14s}{r['sample']:20s}{r['N']:>5}{r['T']:>5}"
              f"{t['lag1_mean']['est']:>8.3f}{t['lag2_mean']['est']:>8.3f}"
              f"{d['cumulative_persistence']['est']:>8.3f}{d['companion_radius']:>8.3f}")
    print("\nSpec C covariate coefficients (mean, both s.e.):")
    for nm in ("population", "gdp"):
        t = specs["C"]["targets"].get(nm + "_mean")
        if t:
            print(f"  {nm:11s}: {t['est']:+.4f}  (White {t['se']:.4f}, cross-sec {t['se_xs']:.4f})")


if __name__ == "__main__":
    main()
