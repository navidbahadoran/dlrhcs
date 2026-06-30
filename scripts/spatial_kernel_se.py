#!/usr/bin/env python3
"""Geographic spatial-kernel (Conley 1999) standard errors for the empirical headline
targets, using Census CBSA centroids as the metric -- the theorem-backed dependence-robust
object of thm:xs_dependence with an *explicit* metric, complementing the by-period
cluster sensitivity standard error.  For each lag target it prints the diagonal,
by-period cluster, and geographic spatial-kernel standard errors at two admissible
Bartlett bandwidths (km), for both the unemployment (spec B) and housing (spec A)
headlines.

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
from dlrhcs.unemp import load_unemp_panel                                   # noqa: E402
from dlrhcs.covariates import load_cbsa_covariates, cbsa_codes_for_zillow   # noqa: E402
from dlrhcs.empirical import run_ar2, load_zillow                          # noqa: E402
from dlrhcs.onestep import xs_se_geo                                       # noqa: E402
from dlrhcs.spatial import haversine_matrix, load_centroids               # noqa: E402
from dlrhcs.pipeline import Tuning                                        # noqa: E402

DATA = os.path.join(ROOT, "data")
UPANEL = os.path.join(DATA, "unemp",
                      "unemployment_metro_model_panel_bls_only_name_matched.csv")
UCOV = os.path.join(DATA, "zillow", "metro_monthly_covariates_2000_present.csv")
ZT = os.path.join(DATA, "zillow", "zillow_metro_top.csv")
ZB = os.path.join(DATA, "zillow", "zillow_metro_bottom.csv")
XW = os.path.join(DATA, "zillow", "cbsa_county_crosswalk_2023.csv")
COORDS = os.path.join(DATA, "coords", "cbsa_centroids.csv")
BANDWIDTHS = [300.0, 600.0]          # km -- two admissible spatial-kernel radii


def _report(label, r, res, lat, lon, targets, n_geo, n_full):
    D = haversine_matrix(lat, lon)
    print(f"\n{label}: coord-matched units {n_geo} of {n_full}.")
    print(f"{'target':28}{'est':>9}{'diag':>9}{'cluster':>10}"
          + "".join(f"{'geo '+str(int(b))+'km':>11}" for b in BANDWIDTHS))
    rows = {}
    for nm in targets:
        t = r["targets"][nm]
        geos = {b: float(xs_se_geo(res, nm, D, b)) for b in BANDWIDTHS}
        rows[nm] = dict(est=t["est"], se_diag=t["se"], se_cluster=t["se_xs"], se_geo=geos)
        print(f"{nm:28}{t['est']:>9.3f}{t['se']:>9.3f}{t['se_xs']:>10.3f}"
              + "".join(f"{geos[b]:>11.3f}" for b in BANDWIDTHS))
    return dict(N_coord_matched=n_geo, N_panel=n_full, bandwidths_km=BANDWIDTHS, targets=rows)


def run_unemp(nj):
    d = load_unemp_panel(UPANEL, start="2005-01", end="2024-12", require_cov=False)
    Y, ml, ces = d["Y"], d["mean_level"], list(d["ces"])
    _, _, matched = load_cbsa_covariates(d["ces"], d["months"], UCOV)
    Y, ml = Y[:, matched], ml[matched]
    ces = [ces[i] for i in range(len(ces)) if matched[i]]
    lat, lon, cm = load_centroids(COORDS, ces)
    n_full = int(len(cm))
    Y, ml, lat, lon = Y[:, cm], ml[cm], lat[cm], lon[cm]
    g = (ml > np.median(ml)).astype(int)
    tun = Tuning(ranks=(1, 0, 1), q=1, J=6, ridge=0.5, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-4, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=nj)
    r = run_ar2(Y, tun, groups=g, group_labels=("hi_unemp", "lo_unemp"),
                rng=np.random.default_rng(13), return_onestep=True)
    targets = ["lag1_mean", "lag1_hi_unemp", "lag1_lo_unemp", "lag1_contrast"]
    return _report("Unemployment headline (spec B)", r, r["onestep"], lat, lon,
                   targets, int(cm.sum()), n_full)


def run_housing(nj):
    z = load_zillow(ZT, ZB, start=None, end=None)
    Y, tier, regions = z["Y"], z["tier"], list(z["col_region"])
    codes = cbsa_codes_for_zillow(regions, XW)
    lat, lon, cm = load_centroids(COORDS, codes)
    n_full = int(len(cm))
    Y, tier, lat, lon = Y[:, cm], tier[cm], lat[cm], lon[cm]
    tun = Tuning(ranks=(1, 1, 1), q=1, J=6, ridge=0.1, n_restarts=2, n_sweeps=60,
                 riesz_tol=1e-5, riesz_ridge=1e-6, riesz_maxiter=600,
                 kappa_c=0.03, xs_kernel="cluster", n_jobs=nj)
    r = run_ar2(Y, tun, groups=tier, group_labels=("top", "bottom"),
                rng=np.random.default_rng(7), return_onestep=True)
    targets = ["lag1_mean", "lag2_mean", "lag1_top", "lag1_bottom", "lag1_contrast"]
    return _report("Housing headline (spec A)", r, r["onestep"], lat, lon,
                   targets, int(cm.sum()), n_full)


def main():
    if not os.path.exists(COORDS):
        sys.exit(f"missing {COORDS}; run `python scripts/build_metro_coords.py` first")
    nj = int(os.environ.get("N_JOBS", "-1") or -1)
    out = {"bandwidths_km": BANDWIDTHS,
           "unemployment": run_unemp(nj),
           "housing": run_housing(nj)}
    p = os.path.join(ROOT, "outputs", "empirical", "spatial_kernel_se.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
