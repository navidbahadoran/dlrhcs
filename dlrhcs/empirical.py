"""
Empirical applications (spec sec 13): the heterogeneous low-rank AR(2)

    y_tilde_{it} = a_{0,ti} y_tilde_{i,t-1} + b_{0,ti} y_tilde_{i,t-2}
                   + h_{0,ti} + u_{it}

estimated by the same cross-fitted debiased one-step.  Blocks are
``[lag1, lag2, ones]`` so ``B = 3`` and the default ranks are ``(1, 1, 2)``.

Two datasets:
  * Zillow            -- metro tier house-value panel (top vs bottom price tier).
  * State unemployment -- 51 U.S. state seasonally-adjusted unemployment rates.

Both panels are homogeneous (one variable across comparable units).  Data are
the RAW downloads; cleaning = balance + align; each series is made stationary by
the lightest step that works (per-series ADF: difference once only if a unit
root, else keep the level) and standardized.  No database-supplied or arbitrary
transforms.  See the data-cleaning appendix.

Every target is reported with both the White s.e. and the within-period
(cross-sectional) s.e.  Derived dynamic functionals (cumulative persistence,
long-run multiplier, horizon IRFs, companion spectral radius) use the delta
method on the joint covariance of the lag-1/lag-2 global means.

Provenance: record the data file and its content hash (``data_fingerprint``) in
the output so a Data Editor can confirm the exact vintage.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .design import build_blocks
from .onestep import (companion_p2, delta_se, irf_p2, joint_cov, lrm_p2)
from .pipeline import Tuning, estimate
from .targets import Target


# --------------------------------------------------------------------------- #
#  data loading
# --------------------------------------------------------------------------- #
def data_fingerprint(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def adf_tstat(y, p=4):
    """Augmented Dickey-Fuller t-statistic on the lagged level (self-contained).

    Regress Delta y_t on [const, y_{t-1}, Delta y_{t-1..t-p}] and return the
    t-stat on the y_{t-1} coefficient.  More negative => reject the unit root.
    """
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy)
    if n <= p + 8:
        return 0.0
    Y = dy[p:]
    rows = [np.ones(n - p), y[p:n]]
    for i in range(1, p + 1):
        rows.append(dy[p - i:n - i])
    X = np.column_stack(rows)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    s2 = resid @ resid / max(len(Y) - X.shape[1], 1)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    return float(beta[1] / se[1])


# Dickey-Fuller 5% critical value (constant, no trend)
ADF_CRIT_5 = -2.86


def is_unit_root(level, crit=ADF_CRIT_5):
    """True if the ADF test fails to reject a unit root (series non-stationary).

    Strictly positive series are tested in logs (multiplicative trends)."""
    z = np.log(level) if np.all(level > 0) else level
    return adf_tstat(z) > crit


def stationarize_panel(levels):
    """Clean -> minimal stationarization (spec sec 13, revised).

    Per series: difference once *only if* it has a unit root (ADF), otherwise
    keep the level.  Strictly positive unit-root series are log-differenced
    (returns -- the stationary transform for multiplicative trends); otherwise a
    plain first difference.  Stationary series are kept in levels.  All series
    are aligned to a common length (drop the first observation) and returned with
    a per-series record of the transform actually applied.  Standardization
    (caller) is location/scale only and leaves the AR dynamics unchanged.
    """
    levels = np.asarray(levels, float)
    T, N = levels.shape
    out = np.empty((T - 1, N))
    transforms = []
    for j in range(N):
        x = levels[:, j]
        if is_unit_root(x):
            if np.all(x > 0):
                out[:, j] = np.diff(np.log(x)); transforms.append("logdiff")
            else:
                out[:, j] = np.diff(x); transforms.append("diff")
        else:
            out[:, j] = x[1:]; transforms.append("level")
    return out, transforms


def load_fredqd(path, stationarize=True):
    """FRED-QD panel: parse the RAW levels, balance, then (default) apply the
    minimal per-series stationarization above and standardize.  The database's
    own tcode transforms are NOT applied -- they over-difference and erase the
    dynamics; we let an ADF test decide, series by series."""
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    body = [r for r in rows[3:] if r and r[0].strip()]   # skip header/factors/transform
    raw = np.array([[_to_float(v) for v in r[1:]] for r in body])   # RAW levels
    good = ~np.any(~np.isfinite(raw), axis=0)             # balanced (fully observed)
    raw = raw[:, good]
    names = [n for n, g in zip(header, good) if g]
    if stationarize:
        X, transforms = stationarize_panel(raw)
    else:
        X, transforms = raw, ["level"] * raw.shape[1]
    vol = X.std(0)
    X = (X - X.mean(0)) / X.std(0)                        # standardize (cleaning)
    return dict(Y=X, names=names, vol=vol, transforms=transforms,
                n_differenced=int(sum(t != "level" for t in transforms)),
                T=X.shape[0], N=X.shape[1],
                fingerprint=data_fingerprint(path), source="FRED-QD")


def load_zillow(path_top, path_bottom, stationarize=True):
    """Zillow ZHVI tier panel: parse RAW price levels, stack top & bottom tiers
    as units, balance, then (default) apply the minimal per-series
    stationarization and standardize.  House-price levels are a unit root, so in
    practice nearly every series is log-differenced (returns)."""
    import csv

    def read(path):
        with open(path) as fh:
            rows = list(csv.reader(fh))
        head = rows[0]
        date_cols = [k for k, c in enumerate(head) if _looks_like_date(c)]
        ids, mat = [], []
        for r in rows[1:]:
            if r[3] != "msa":
                continue
            ids.append(r[2])
            mat.append([_to_float(r[k]) for k in date_cols])
        return ids, np.array(mat)

    idt, top = read(path_top)
    idb, bot = read(path_bottom)
    common = [r for r in idt if r in set(idb)]
    it = {r: k for k, r in enumerate(idt)}
    ib = {r: k for k, r in enumerate(idb)}
    levels = np.vstack([np.array([top[it[r]] for r in common]),
                        np.array([bot[ib[r]] for r in common])]).T   # (T x 2*regions)
    good = ~np.any(~np.isfinite(levels), axis=0)
    levels = levels[:, good]
    tier = np.array([0] * len(common) + [1] * len(common))[good]
    if stationarize:
        X, transforms = stationarize_panel(levels)
    else:
        X, transforms = levels, ["level"] * levels.shape[1]
    vol = X.std(0)
    X = (X - X.mean(0)) / X.std(0)
    return dict(Y=X, tier=tier, vol=vol, transforms=transforms,
                n_differenced=int(sum(t != "level" for t in transforms)),
                regions=common, T=X.shape[0], N=X.shape[1],
                fingerprint=data_fingerprint(path_top)[:8] + data_fingerprint(path_bottom)[:8],
                source="Zillow")


def load_unemployment(path, stationarize=True):
    """State-unemployment panel: wide CSV (DATE + one column per state).  Parse
    the RAW rates, balance, apply the same minimal per-series stationarization,
    and standardize.  Grouping uses the pre-transform average rate."""
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    body = [r for r in rows[1:] if r and r[0].strip()]
    nc = len(header)
    raw = np.full((len(body), nc), np.nan)         # ragged-robust parse (pad short rows)
    for i, r in enumerate(body):
        for j in range(nc):
            if j + 1 < len(r):
                raw[i, j] = _to_float(r[j + 1])
    raw, good = _balanced_block(raw)               # longest contiguous balanced block
    names = [n for n, g in zip(header, good) if g]
    mean_level = raw.mean(0)                       # avg unemployment per state (for grouping)
    if stationarize:
        X, transforms = stationarize_panel(raw)
    else:
        X, transforms = raw, ["level"] * raw.shape[1]
    vol = X.std(0)
    X = (X - X.mean(0)) / X.std(0)
    return dict(Y=X, names=names, mean_level=mean_level, vol=vol,
                transforms=transforms,
                n_differenced=int(sum(t != "level" for t in transforms)),
                T=X.shape[0], N=X.shape[1],
                fingerprint=data_fingerprint(path), source="StateUnemployment")


def unemployment_groups(panel) -> np.ndarray:
    """Median split by average unemployment level (1 = high, 0 = low)."""
    ml = panel["mean_level"]
    return (ml > np.median(ml)).astype(int)


def load_metro(path, stationarize=True):
    """Metro-area unemployment panel: wide CSV (YEAR + one column per MSA) of
    BLS LAUS annual-average unemployment rates (period M13).  Built by
    ``data/metro/build_metro_panel.py`` from the raw ``la.data.60.Metro`` file;
    383 metros balanced over 1990-2025 (T=36).  Annual averages carry no
    seasonality (they ARE BLS's own M13 within-year mean), so no seasonal
    adjustment is needed -- this is the cleanest cut of the metro data.

    Parse the RAW rates, apply the same minimal per-series stationarization
    (per-series ADF: difference once only if a unit root, else keep the level),
    and standardize.  Grouping uses the pre-transform mean rate.  Unlike the
    state panel (T/N~12, which over-fit the interactive block), this panel has
    N>>T, the well-conditioned regime for the low-rank time factors."""
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    raw = np.array([[_to_float(v) for v in r[1:]]
                    for r in rows[1:] if r and r[0].strip()])
    mean_level = raw.mean(0)                        # avg unemployment per metro
    if stationarize:
        X, transforms = stationarize_panel(raw)
    else:
        X, transforms = raw, ["level"] * raw.shape[1]
    vol = X.std(0)
    X = (X - X.mean(0)) / X.std(0)
    return dict(Y=X, names=header, mean_level=mean_level, vol=vol,
                transforms=transforms,
                n_differenced=int(sum(t != "level" for t in transforms)),
                T=X.shape[0], N=X.shape[1],
                fingerprint=data_fingerprint(path), source="MetroUnemployment")


def metro_groups(panel) -> np.ndarray:
    """Median split by average unemployment level (1 = high, 0 = low)."""
    ml = panel["mean_level"]
    return (ml > np.median(ml)).astype(int)


def panel_diagnostic(Y, n_factors=2) -> Dict:
    """Go/no-go checklist for a candidate empirical panel (homogeneity is by
    construction).  Reports own-AR(1), idiosyncratic AR(1) after removing
    ``n_factors`` common factors, the first-two-PC variance share, and size."""
    Y = np.asarray(Y, float)
    T, N = Y.shape

    def mean_ar1(M):
        return float(np.mean([np.dot(M[:-1, j], M[1:, j]) /
                              max(np.dot(M[:-1, j], M[:-1, j]), 1e-12)
                              for j in range(M.shape[1])]))
    Yc = Y - Y.mean(0)
    U, sv, Vt = np.linalg.svd(Yc, full_matrices=False)
    var = sv ** 2 / np.sum(sv ** 2)
    low = (U[:, :n_factors] * sv[:n_factors]) @ Vt[:n_factors]
    return dict(T=T, N=N, mean_AR1=mean_ar1(Y),
                idiosyncratic_AR1=mean_ar1(Yc - low),
                pc_share_2=float(var[:2].sum()))


def _looks_like_date(s: str) -> bool:
    s = s.strip()
    return len(s) >= 6 and (s[:4].isdigit() and ("-" in s or "/" in s))


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def _balanced_block(raw, col_thresh=0.8):
    """Return the largest balanced rectangle: drop sparse columns, then take the
    longest run of *consecutive* fully-observed rows (AR needs time-contiguity,
    so interior gaps are handled by truncation, not row deletion)."""
    raw = np.asarray(raw, float)
    finite = np.isfinite(raw)
    col_ok = finite.mean(0) >= col_thresh
    sub = raw[:, col_ok]
    rows_ok = np.all(np.isfinite(sub), axis=1)
    best_len = best_start = cur = start = 0
    for t, ok in enumerate(rows_ok):
        if ok:
            if cur == 0:
                start = t
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, start
        else:
            cur = 0
    return sub[best_start:best_start + best_len], col_ok


# --------------------------------------------------------------------------- #
#  AR(2) panel construction
# --------------------------------------------------------------------------- #
def build_ar2(Ymat: np.ndarray):
    """From a (T x N) stationary series matrix build (Y_eff, [lag1, lag2])."""
    T, N = Ymat.shape
    Y = Ymat[2:]                                # effective sample t = 3..T
    lag1 = Ymat[1:-1]
    lag2 = Ymat[0:-2]
    return Y, [lag1, lag2]


# --------------------------------------------------------------------------- #
#  targets
# --------------------------------------------------------------------------- #
def _cellmean_dir(blocks, block, W):
    D = [np.zeros_like(zb) for zb in blocks]
    D[block] = W
    return D


def ar2_targets(blocks, Tp, N, groups=None, group_labels=("g0", "g1")):
    """Global-mean lag1/lag2, optional group means and a between-group contrast."""
    Wall = np.full((Tp, N), 1.0 / (Tp * N))
    targets = [
        Target("lag1_mean", 0, _cellmean_dir(blocks, 0, Wall)),
        Target("lag2_mean", 1, _cellmean_dir(blocks, 1, Wall)),
    ]
    if groups is not None:
        g0 = np.where(groups == 0)[0]
        g1 = np.where(groups == 1)[0]
        W0 = np.zeros((Tp, N)); W0[:, g0] = 1.0 / (Tp * len(g0))
        W1 = np.zeros((Tp, N)); W1[:, g1] = 1.0 / (Tp * len(g1))
        targets += [
            Target(f"lag1_{group_labels[0]}", 0, _cellmean_dir(blocks, 0, W0)),
            Target(f"lag1_{group_labels[1]}", 0, _cellmean_dir(blocks, 0, W1)),
            Target("lag1_contrast", 0, _cellmean_dir(blocks, 0, W0 - W1)),
        ]
    return targets


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
def run_ar2(Ymat, tuning: Tuning, groups=None, group_labels=("g0", "g1"),
            rng=None):
    """Estimate the heterogeneous AR(2) and all targets; add derived dynamics."""
    if rng is None:
        rng = np.random.default_rng(0)
    Y, Z = build_ar2(Ymat)
    Tp, N = Y.shape
    blocks = build_blocks(Z)
    targets = ar2_targets(blocks, Tp, N, groups=groups, group_labels=group_labels)
    res = estimate(Y, Z, targets, tuning, P=2, rng=rng)

    table = {}
    for tg in targets:
        table[tg.name] = dict(est=res.estimates[tg.name], se=res.se[tg.name],
                              se_xs=res.se_xs[tg.name], ci=res.ci[tg.name],
                              ci_xs=res.ci_xs[tg.name])

    # Derived dynamics from the lag1/lag2 global means via the delta method.
    # The joint covariance is the White (sandwich) form -- this is exactly the
    # studentizer of thm:irf / lem:joint_clt, which is stated under the baseline
    # (the paper does not define a cross-sectional dependence-robust delta-method
    # covariance for IRF/LRM).  The scalar lag means above additionally carry a
    # cross-sectional (cluster) s.e.; the derived functionals follow thm:irf.
    a = res.estimates["lag1_mean"]; b = res.estimates["lag2_mean"]
    Sig = joint_cov(res.onestep, ["lag1_mean", "lag2_mean"])
    cum = a + b
    cum_se = float(np.sqrt(max(np.array([1.0, 1.0]) @ Sig @ np.array([1.0, 1.0]), 0)))
    lrm_val, lrm_g = lrm_p2(a, b)
    lrm_se = delta_se(res.onestep, ["lag1_mean", "lag2_mean"], lrm_g)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion_p2(a, b)))))
    irfs = {}
    for h in (1, 2, 4, 8):
        val, g = irf_p2(a, b, h)
        irfs[h] = dict(est=val,
                       se=delta_se(res.onestep, ["lag1_mean", "lag2_mean"], g))

    derived = dict(cumulative_persistence=dict(est=cum, se=cum_se),
                   long_run_multiplier=dict(est=lrm_val, se=lrm_se),
                   companion_radius=radius, irf=irfs)
    return dict(targets=table, derived=derived, ranks=res.ranks,
                q=res.q, J=res.J, Tp=Tp, N=N)


def fred_volatility_groups(panel) -> np.ndarray:
    """Median split of series by *pre-standardization* sample volatility
    (1 = high, 0 = low).  Standardized series all have unit variance, so the
    split must use the raw transformed-series volatility."""
    vol = panel.get("vol")
    if vol is None:
        vol = panel["Y"].std(0)
    return (vol > np.median(vol)).astype(int)


def rank_robustness(Ymat, rH_list, base_tuning, groups=None,
                    group_labels=("g0", "g1"), seed=2024):
    """Re-estimate at several nuisance ranks r_H; report the headline targets
    so the empirical conclusions can be shown robust to r_H (spec sec 13)."""
    import dataclasses
    out = {}
    for rH in rH_list:
        t = dataclasses.replace(base_tuning, ranks=(1, 1, int(rH)))
        r = run_ar2(Ymat, t, groups=groups, group_labels=group_labels,
                    rng=np.random.default_rng(seed))
        out[int(rH)] = r
    return out
