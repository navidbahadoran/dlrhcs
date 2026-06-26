#!/usr/bin/env python3
"""Unemployment A/B/C specifications + full referee diagnostics (Phase 2, labor).

  A. Main baseline       2000-2026, no covariate                 (longer-sample check)
  B. Restricted baseline 2005-2024, no covariate, matched metros (HEADLINE)
  C. Covariate-augmented 2005-2024, population + real-GDP growth (local-fundamentals)

Heterogeneous AR(1): the second lag is unidentified at monthly frequency (near-
integrated), so ranks are (1,0,3); the rate is NSA, deseasonalized via month-of-year
means (level-preserving).  Covariates are CBSA population/GDP growth (NOT employment,
a labor-market identity), matched by ces_cbsa_code, predetermined, winsorized +
standardized.  B and C use the SAME covariate-matched metros so B->C isolates the
covariate.  Beyond the headline targets, run_ar2 reports plugin-vs-debiased estimates,
group/global cumulative persistence, the long-run multiplier, IRFs to h=12, residual
adequacy, fit, heterogeneity, and solver diagnostics; this script adds the rank-
selection candidate table and r_H sweep on the headline (B) and covariate forced-rank
robustness on C.  Run from the repo root:  python scripts/unemp_abc.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.unemp import load_unemp_panel                              # noqa: E402
from dlrhcs.covariates import load_cbsa_covariates                     # noqa: E402
from dlrhcs.empirical import (covariate_robustness, data_fingerprint,  # noqa: E402
                              rank_robustness, rank_selection_table, run_ar2)
from dlrhcs.pipeline import Tuning                                      # noqa: E402

PANEL = os.path.join(ROOT, "data", "unemp",
                     "unemployment_metro_model_panel_bls_only_name_matched.csv")
COV = os.path.join(ROOT, "data", "zillow", "metro_monthly_covariates_2000_present.csv")
SEED = 13
RBAR = (2, 2, 4)          # candidate-box caps (lag1, lag2, H) for rank selection
RH_SWEEP = [2, 3, 4]      # interactive-block ranks for the r_H robustness sweep


def run_spec(label, start, end, mode, n_jobs=1, extras=()):
    d = load_unemp_panel(PANEL, start=start, end=end, require_cov=False)
    Y, ml = d["Y"], d["mean_level"]
    metros, ces = list(d["metros"]), list(d["ces"])
    n_total = int(Y.shape[1])
    covars, covar_names, ranks = None, (), (1, 0, 3)       # AR(1): lag2 dropped, H rank 3
    if mode in ("matched_nocov", "matched_cov"):
        mats, names, matched = load_cbsa_covariates(d["ces"], d["months"], COV)
        Y, ml = Y[:, matched], ml[matched]
        metros = [metros[i] for i in range(len(metros)) if matched[i]]
        ces = [ces[i] for i in range(len(ces)) if matched[i]]
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
    r["months"] = list(d["months"])[-r["Tp"]:]
    r["metros"] = metros
    r["ces"] = ces
    r["group"] = [int(x) for x in g]
    r["data_summary"] = dict(fingerprint=data_fingerprint(PANEL), n_units=int(Y.shape[1]),
                             n_units_total=n_total, n_dropped=int(n_total - Y.shape[1]),
                             date_range=f"{d['months'][0]}..{d['months'][-1]}")
    if "rank_select" in extras:
        sel = dataclasses.replace(tun, r_bar=RBAR)
        r["rank_selection"] = rank_selection_table(Y, sel, groups=g,
                                                   group_labels=("hi_unemp", "lo_unemp"), top_k=8)
    if "rank_robust" in extras:
        r["rank_robustness"] = rank_robustness(Y, ranks, RH_SWEEP, tun, groups=g,
                                               group_labels=("hi_unemp", "lo_unemp"),
                                               covars=covars, covar_names=covar_names)
    if "cov_robust" in extras:
        r["covariate_robustness"] = covariate_robustness(Y, ranks, tun, covars, covar_names,
                                                         groups=g, group_labels=("hi_unemp", "lo_unemp"))
    return r


def main():
    nj = int(os.environ.get("N_JOBS", "1"))
    plan = [("A", "A_main", "2000-01", "2026-05", "full", ()),
            ("B", "B_restricted", "2005-01", "2024-12", "matched_nocov", ("rank_select", "rank_robust")),
            ("C", "C_covariates", "2005-01", "2024-12", "matched_cov", ("cov_robust",))]
    outdir = os.path.join(ROOT, "outputs", "empirical")
    os.makedirs(outdir, exist_ok=True)
    specs = {}
    for k, lab, a, b, mode, extras in plan:
        r = run_spec(lab, a, b, mode, nj, extras)
        specs[k] = r
        json.dump(r, open(os.path.join(outdir, f"unemp_{k}.json"), "w"), indent=2, default=str)
        d = r["derived"]; t = r["targets"]
        print(f"[done {k}] {lab} N={r['N']} T={r['T']} lag1={t['lag1_mean']['est']:.3f} "
              f"cum={d['cumulative_persistence']['est']:.3f} radius={d['companion_radius']:.3f} "
              f"rmse={d['fit']['rmse']:.4f}", flush=True)
    json.dump(specs, open(os.path.join(outdir, "unemp_abc.json"), "w"), indent=2, default=str)
    print(f"\n{'spec':14s}{'sample':20s}{'N':>5}{'T':>5}{'lag1':>8}{'lag2':>8}{'cum':>8}{'radius':>8}")
    for k in ("A", "B", "C"):
        r = specs[k]; t = r["targets"]; d = r["derived"]
        print(f"{r['spec']:14s}{r['sample']:20s}{r['N']:>5}{r['T']:>5}"
              f"{t['lag1_mean']['est']:>8.3f}{t['lag2_mean']['est']:>8.3f}"
              f"{d['cumulative_persistence']['est']:>8.3f}{d['companion_radius']:>8.3f}")
    if "rank_selection" in specs["B"]:
        print("\nSpec B rank selection (top 5 by criterion):")
        for c in specs["B"]["rank_selection"]["candidates"][:5]:
            print(f"  rank={c['rank']} cv_loss={c['cv_loss']:.4f} eff_dim={c['eff_dim']:.0f} crit={c['criterion']:.4f}")
    print("\nSpec C covariate coefficients (mean, both s.e.):")
    for nm in ("population", "gdp"):
        t = specs["C"]["targets"].get(nm + "_mean")
        if t:
            print(f"  {nm:11s}: {t['est']:+.4f}  (White {t['se']:.4f}, cross-sec {t['se_xs']:.4f})")


if __name__ == "__main__":
    main()
