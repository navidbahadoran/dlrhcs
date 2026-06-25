"""Metro covariate loading for the Zillow application (Phase 2, spec C).

Reads the consolidated metro monthly covariate file and the CBSA-county
crosswalk (both under ``data/covariates/``) and builds covariate matrices
aligned to a given list of Zillow metros and months.  All data transformations
-- county->CBSA aggregation via the 2023 delineation, annual->monthly
interpolation of GDP/population, and the Delta-log 12-month growth rates -- are
performed UPSTREAM by ``data/covariates/zillow-covariate.py``; this module only
matches metros to CBSAs, aligns the series, and winsorizes/standardizes them.

Each covariate enters the model PREDETERMINED (lagged one month) via
``build_ar2``; the matrices returned here are the contemporaneous monthly growth
series, aligned to ``months``.
"""
from __future__ import annotations

import csv
import re

import numpy as np

COVARIATES = ("permits_units_growth_12m", "population_growth_12m", "real_gdp_growth_1y")
COV_LABELS = {"permits_units_growth_12m": "permits",
              "population_growth_12m": "population",
              "real_gdp_growth_1y": "gdp"}


def _key(name: str):
    """(principal-city, state-abbrev) match key from a metro title."""
    city = re.split(r"[-/]", name.split(",")[0])[0].strip().lower()
    st = name.split(",")[-1].strip().split("-")[0].strip().lower() if "," in name else ""
    return (city, st)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _crosswalk(path):
    rows = list(csv.reader(open(path)))
    h = {c: i for i, c in enumerate(rows[0])}
    k2c = {}
    for r in rows[1:]:
        k2c.setdefault(_key(r[h["cbsa_name"]]), r[h["cbsa_code"]].zfill(5))
    return k2c


def _cov_table(path, covs):
    rows = list(csv.reader(open(path)))
    h = {c: i for i, c in enumerate(rows[0])}
    tab = {}
    for r in rows[1:]:
        tab[(r[h["cbsa_code"]], r[h["date"]][:7])] = {c: r[h[c]] for c in covs}
    return tab


def _winsorize_standardize(c):
    """Winsorize at the 1st/99th percentiles (the permit-growth series has extreme
    swings that ill-condition the first-stage SVD) and z-score over the non-missing
    entries, matching how the DGP standardizes its regressor."""
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        return c
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    out = np.clip(c, lo, hi)
    mu, sd = np.nanmean(out), np.nanstd(out)
    return (out - mu) / sd if sd > 0 else out - mu


def load_zillow_covariates(region_names, months, cov_path, xw_path, covs=COVARIATES,
                           standardize=True):
    """Build covariate matrices for the Zillow metros over ``months``.

    region_names : Zillow ``RegionName`` per panel column (duplicated across tiers).
    months       : target month labels ``"YYYY-MM"``.
    Returns ``(mats, names, matched)``: one (T x N) array per covariate (NaN where
    the metro has no complete match), short labels, and a per-column boolean mask
    that is True where ALL covariates are complete.
    """
    k2c = _crosswalk(xw_path)
    tab = _cov_table(cov_path, covs)
    T, N = len(months), len(region_names)
    mats = [np.full((T, N), np.nan) for _ in covs]
    matched = np.zeros(N, dtype=bool)
    for j, z in enumerate(region_names):
        code = k2c.get(_key(z))
        if not code:
            continue
        col = np.array([[_f(tab.get((code, ym), {}).get(c, "")) for c in covs]
                        for ym in months])
        if not np.isnan(col).any():
            for m in range(len(covs)):
                mats[m][:, j] = col[:, m]
            matched[j] = True
    if standardize:
        mats = [_winsorize_standardize(m) for m in mats]
    return mats, [COV_LABELS.get(c, c) for c in covs], matched


def load_cbsa_covariates(cbsa_codes, months, cov_path,
                         covs=("population_growth_12m", "real_gdp_growth_1y"),
                         labels=("population", "gdp"), standardize=True):
    """Covariate matrices matched DIRECTLY by CBSA code (for the unemployment
    panel, whose ces_cbsa_code is the modern CBSA).  Leading zeros are stripped on
    both sides.  Returns (mats, labels, matched) like load_zillow_covariates."""
    rows = list(csv.reader(open(cov_path)))
    h = {c: i for i, c in enumerate(rows[0])}
    tab = {}
    for r in rows[1:]:
        tab[(r[h["cbsa_code"]].lstrip("0"), r[h["date"]][:7])] = {c: r[h[c]] for c in covs}
    T, N = len(months), len(cbsa_codes)
    mats = [np.full((T, N), np.nan) for _ in covs]
    matched = np.zeros(N, dtype=bool)
    for j, code in enumerate(cbsa_codes):
        cc = str(code).lstrip("0")
        col = np.array([[_f(tab.get((cc, m), {}).get(c, "")) for c in covs] for m in months])
        if not np.isnan(col).any():
            for k in range(len(covs)):
                mats[k][:, j] = col[:, k]
            matched[j] = True
    if standardize:
        mats = [_winsorize_standardize(m) for m in mats]
    return mats, list(labels), matched
