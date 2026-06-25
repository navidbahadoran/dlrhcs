"""Monthly metro unemployment panel + CES payroll covariate (Phase 2, spec C).

Reads the self-contained name-matched LAUS+CES file: one row per metro-month with
the NSA unemployment rate and the 12-month CES payroll-employment growth.  Pivots
to metro x month matrices, dedups, linearly interpolates the sparse BLS
suppressions, removes a per-metro month-of-year seasonal-dummy pattern from the
(NSA) rate, and standardizes the payroll covariate.  The covariate enters the
model PREDETERMINED (one-month lag) via build_ar2.
"""
from __future__ import annotations

import csv

import numpy as np

MAX_GAP = 6


def _interp(vals):
    n = len(vals)
    idx = [i for i, v in enumerate(vals) if not np.isnan(v)]
    if not idx:
        return vals
    for i in range(idx[0]):
        vals[i] = vals[idx[0]]
    for i in range(idx[-1] + 1, n):
        vals[i] = vals[idx[-1]]
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            va, vb = vals[a], vals[b]
            for k in range(a + 1, b):
                vals[k] = va + (vb - va) * (k - a) / (b - a)
    return vals


def _ws(c):
    """Winsorize (1/99 pct) + standardize a covariate matrix over finite entries."""
    f = c[np.isfinite(c)]
    if f.size == 0:
        return c
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    out = np.clip(c, lo, hi)
    mu, sd = np.nanmean(out), np.nanstd(out)
    return (out - mu) / sd if sd > 0 else out - mu


def load_unemp_panel(path, start="2000-01", end="2026-12", deseasonalize=True, require_cov=True,
                     rate_col="unemployment_rate", cov_col="payroll_growth_12m"):
    """Return dict with Y (rate, deseasonalized), payroll (standardized covariate),
    months, metros -- all aligned (T x N)."""
    rows = list(csv.reader(open(path)))
    h = {c: i for i, c in enumerate(rows[0])}
    body = [r for r in rows[1:] if start <= r[h["date"]][:7] <= end]
    months = sorted({r[h["date"]][:7] for r in body})
    metros = sorted({r[h["cbsa_code"]] for r in body})
    mi = {m: i for i, m in enumerate(months)}
    ki = {k: j for j, k in enumerate(metros)}
    U = np.full((len(months), len(metros)), np.nan)
    P = np.full_like(U, np.nan)

    def f(r, c):
        v = r[h[c]].strip()
        try:
            return float(v)
        except ValueError:
            return np.nan
    for r in body:                                   # dedup: last write wins
        i, j = mi[r[h["date"]][:7]], ki[r[h["cbsa_code"]]]
        U[i, j] = f(r, rate_col)
        P[i, j] = f(r, cov_col)

    keep = np.isnan(U).sum(0) <= MAX_GAP                # complete unemployment
    if require_cov:                                  # require complete payroll only when used
        keep &= np.isnan(P).sum(0) <= MAX_GAP
    U, P = U[:, keep], P[:, keep]
    metros = [m for m, k in zip(metros, keep) if k]
    for j in range(U.shape[1]):                      # interpolate sparse gaps
        U[:, j] = _interp(list(U[:, j]))
        P[:, j] = _interp(list(P[:, j]))

    mean_level = U.mean(0)                           # average rate per metro (for grouping)
    if deseasonalize:                                # remove month-of-year seasonal effect,
        moy = np.array([int(m[5:7]) for m in months])# add back the grand mean so the LEVEL is kept
        grand = U.mean(0, keepdims=True)
        for mo in range(1, 13):
            mask = moy == mo
            U[mask] -= (U[mask].mean(0, keepdims=True) - grand)

    return dict(Y=U, payroll=_ws(P), months=months, metros=metros,
                mean_level=mean_level, T=int(U.shape[0]), N=int(U.shape[1]))
