#!/usr/bin/env python3
"""Zillow A/B/C specifications + full referee diagnostics (Phase 2, housing).

  A. Main baseline       2000-2026, no covariates           (headline, full T)
  B. Restricted baseline 2005-2024, no covariates           (sample-restriction check)
  C. Covariate-augmented 2005-2024, permits+population+GDP  (local-fundamentals check)

A vs B isolates the shorter sample; B vs C isolates the covariates.  Beyond the
headline targets, run_ar2 now also reports plugin-vs-debiased estimates, group and
global cumulative persistence, the long-run multiplier, IRFs to h=12, residual
adequacy diagnostics, fit (R^2 vs outcome and vs a no-dynamics baseline), coefficient
heterogeneity, and solver diagnostics.  This script additionally computes, on the
headline spec, the rank-selection candidate table and the r_H robustness sweep, and on
spec C the covariate forced-rank robustness.  Writes per-spec and combined JSON.
Run from the repo root:  python scripts/zillow_abc.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.empirical import (covariate_robustness, load_zillow,        # noqa: E402
                              rank_robustness, rank_selection_table, run_ar2)
from dlrhcs.covariates import load_zillow_covariates                    # noqa: E402
from dlrhcs.pipeline import Tuning                                       # noqa: E402

DATA = os.path.join(ROOT, "data")
ZT = os.path.join(DATA, "zillow", "zillow_metro_top.csv")
ZB = os.path.join(DATA, "zillow", "zillow_metro_bottom.csv")
COV = os.path.join(DATA, "zillow", "metro_monthly_covariates_2000_present.csv")
XW = os.path.join(DATA, "zillow", "cbsa_county_crosswalk_2023.csv")
SEED = 7
RBAR = (2, 2, 3)          # candidate-box caps (lag1, lag2, H) for rank selection
RH_SWEEP = [0, 1, 2]      # interactive-block ranks for the r_H robustness sweep


def run_spec(label, start, end, mode, n_jobs=1, extras=()):
    z = load_zillow(ZT, ZB, start=start, end=end)
    Y, tier = z["Y"], z["tier"]
    regions = list(z["col_region"])
    covars, covar_names, ranks = None, (), (1, 1, 1)
    if mode in ("matched_nocov", "matched_cov"):
        mats, names, matched = load_zillow_covariates(list(z["col_region"]), z["months"], COV, XW)
        Y, tier = Y[:, matched], tier[matched]
        regions = [regions[i] for i in range(len(regions)) if matched[i]]
        if mode == "matched_cov":
            covars = [m[:, matched] for m in mats]
            covar_names = names
            ranks = (1, 1, 1, 1, 1, 1)         # lag1, lag2, 3 covariates, H
    tun = Tuning(ranks=ranks, q=1, J=6, ridge=0.1, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-6, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=n_jobs)
    r = run_ar2(Y, tun, groups=tier, group_labels=("top", "bottom"),
                rng=np.random.default_rng(SEED), covars=covars, covar_names=covar_names)
    r["spec"] = label
    r["sample"] = f"{start or '2000-01'}..{end or 'latest'}"
    r["N"], r["T"] = int(Y.shape[1]), int(Y.shape[0])
    r["months"] = list(z["months"])[-r["Tp"]:]
    r["regions"] = regions
    r["tier"] = [int(x) for x in tier]
    r["data_summary"] = dict(fingerprint=z["fingerprint"], n_units=int(Y.shape[1]),
                             n_units_total=int(z["N"]), n_dropped=int(z["N"] - Y.shape[1]),
                             date_range=f"{z['months'][0]}..{z['months'][-1]}",
                             n_differenced=int(z.get("n_differenced", 0)))
    if "rank_select" in extras:
        sel = dataclasses.replace(tun, r_bar=RBAR)
        r["rank_selection"] = rank_selection_table(Y, sel, groups=tier,
                                                   group_labels=("top", "bottom"), top_k=8)
    if "rank_robust" in extras:
        r["rank_robustness"] = rank_robustness(Y, ranks, RH_SWEEP, tun, groups=tier,
                                               group_labels=("top", "bottom"),
                                               covars=covars, covar_names=covar_names)
    if "cov_robust" in extras:
        r["covariate_robustness"] = covariate_robustness(Y, ranks, tun, covars, covar_names,
                                                         groups=tier, group_labels=("top", "bottom"))
    return r


def main():
    nj = int(os.environ.get("N_JOBS", "1"))
    plan = [("A", "A_main", None, None, "full", ("rank_select", "rank_robust")),
            ("B", "B_restricted", "2005-01", "2024-12", "matched_nocov", ()),
            ("C", "C_covariates", "2005-01", "2024-12", "matched_cov", ("cov_robust",))]
    outdir = os.path.join(ROOT, "outputs", "empirical")
    os.makedirs(outdir, exist_ok=True)
    specs = {}
    for k, lab, a, b, mode, extras in plan:
        r = run_spec(lab, a, b, mode, nj, extras)
        specs[k] = r
        json.dump(r, open(os.path.join(outdir, f"zillow_{k}.json"), "w"), indent=2, default=str)
        d = r["derived"]; t = r["targets"]
        print(f"[done {k}] {lab} N={r['N']} T={r['T']} lag1={t['lag1_mean']['est']:.3f} "
              f"lag2={t['lag2_mean']['est']:.3f} cum={d['cumulative_persistence']['est']:.3f} "
              f"radius={d['companion_radius']:.3f} rmse={d['fit']['rmse']:.4f}", flush=True)
    json.dump(specs, open(os.path.join(outdir, "zillow_abc.json"), "w"), indent=2, default=str)
    print(f"\n{'spec':14s}{'sample':18s}{'N':>5}{'T':>5}{'lag1':>8}{'lag2':>8}{'cum':>8}{'radius':>8}")
    for k in ("A", "B", "C"):
        r = specs[k]; t = r["targets"]; d = r["derived"]
        print(f"{r['spec']:14s}{r['sample']:18s}{r['N']:>5}{r['T']:>5}"
              f"{t['lag1_mean']['est']:>8.3f}{t['lag2_mean']['est']:>8.3f}"
              f"{d['cumulative_persistence']['est']:>8.3f}{d['companion_radius']:>8.3f}")
    if "rank_selection" in specs["A"]:
        print("\nSpec A rank selection (top 5 by criterion):")
        for c in specs["A"]["rank_selection"]["candidates"][:5]:
            print(f"  rank={c['rank']} cv_loss={c['cv_loss']:.4f} eff_dim={c['eff_dim']:.0f} crit={c['criterion']:.4f}")
    print("\nSpec C covariate coefficients (mean, both s.e.):")
    for nm in ("permits", "population", "gdp"):
        t = specs["C"]["targets"][nm + "_mean"]
        print(f"  {nm:11s}: {t['est']:+.4f}  (White {t['se']:.4f}, cross-sec {t['se_xs']:.4f})")


if __name__ == "__main__":
    main()
