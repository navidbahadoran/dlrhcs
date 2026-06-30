#!/usr/bin/env python3
"""Geographic spatial-kernel (Conley 1999) standard errors for the unemployment
headline targets, using Census CBSA centroids as the metric.  This is the theorem-backed
dependence-robust object of thm:xs_dependence with an *explicit, credible* metric, so it
complements the by-period cluster sensitivity standard error reported in the headline
tables.  For each lag target it prints the diagonal, by-period cluster, and geographic
spatial-kernel standard errors at two admissible Bartlett bandwidths (km).

Prerequisite: ``python scripts/build_metro_coords.py`` (writes
``data/coords/cbsa_centroids.csv``).  Run from the repo root:
    python scripts/spatial_kernel_se.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.unemp import load_unemp_panel                      # noqa: E402
from dlrhcs.covariates import load_cbsa_covariates             # noqa: E402
from dlrhcs.empirical import run_ar2                           # noqa: E402
from dlrhcs.onestep import xs_se_geo                           # noqa: E402
from dlrhcs.spatial import haversine_matrix, load_centroids    # noqa: E402
from dlrhcs.pipeline import Tuning                             # noqa: E402

PANEL = os.path.join(ROOT, "data", "unemp",
                     "unemployment_metro_model_panel_bls_only_name_matched.csv")
COV = os.path.join(ROOT, "data", "zillow", "metro_monthly_covariates_2000_present.csv")
COORDS = os.path.join(ROOT, "data", "coords", "cbsa_centroids.csv")
BANDWIDTHS = [300.0, 600.0]          # km -- two admissible spatial-kernel radii
TARGETS = ["lag1_mean", "lag1_hi_unemp", "lag1_lo_unemp", "lag1_contrast"]
SEED = 13


def main():
    if not os.path.exists(COORDS):
        sys.exit(f"missing {COORDS}; run `python scripts/build_metro_coords.py` first")
    nj = int(os.environ.get("N_JOBS", "-1") or -1)
    # headline unemployment specification B (2005--2024, covariate-matched metros)
    d = load_unemp_panel(PANEL, start="2005-01", end="2024-12", require_cov=False)
    Y, ml, ces = d["Y"], d["mean_level"], list(d["ces"])
    _, _, matched = load_cbsa_covariates(d["ces"], d["months"], COV)
    Y, ml = Y[:, matched], ml[matched]
    ces = [ces[i] for i in range(len(ces)) if matched[i]]
    # restrict to coord-matched metros so the score field and the metric align
    lat, lon, cm = load_centroids(COORDS, ces)
    Y, ml, lat, lon = Y[:, cm], ml[cm], lat[cm], lon[cm]
    n_geo, n_full = int(cm.sum()), int(len(cm))
    g = (ml > np.median(ml)).astype(int)
    tun = Tuning(ranks=(1, 0, 1), q=1, J=6, ridge=0.5, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-4, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=nj)
    r = run_ar2(Y, tun, groups=g, group_labels=("hi_unemp", "lo_unemp"),
                rng=np.random.default_rng(SEED), return_onestep=True)
    res = r["onestep"]
    D = haversine_matrix(lat, lon)

    out = {"sample": "B 2005-2024", "N_coord_matched": n_geo, "N_panel": n_full,
           "bandwidths_km": BANDWIDTHS, "targets": {}}
    print(f"Unemployment headline (spec B); coord-matched metros {n_geo} of {n_full}.")
    print(f"{'target':30}{'est':>9}{'diag':>9}{'cluster':>10}"
          + "".join(f"{'geo '+str(int(b))+'km':>11}" for b in BANDWIDTHS))
    for nm in TARGETS:
        t = r["targets"][nm]
        geos = {b: float(xs_se_geo(res, nm, D, b)) for b in BANDWIDTHS}
        out["targets"][nm] = dict(est=t["est"], se_diag=t["se"], se_cluster=t["se_xs"],
                                  se_geo=geos)
        print(f"{nm:30}{t['est']:>9.3f}{t['se']:>9.3f}{t['se_xs']:>10.3f}"
              + "".join(f"{geos[b]:>11.3f}" for b in BANDWIDTHS))
    p = os.path.join(ROOT, "outputs", "empirical", "spatial_kernel_se.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
