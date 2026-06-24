"""
Empirical applications (spec sec 13): the heterogeneous low-rank AR(2)

    y_tilde_{it} = a_{0,ti} y_tilde_{i,t-1} + b_{0,ti} y_tilde_{i,t-2}
                   + h_{0,ti} + u_{it}

estimated by the same cross-fitted debiased one-step.  Blocks are
``[lag1, lag2, ones]`` so ``B = 3`` and the default ranks are ``(1, 1, 2)``.

Datasets:
  * Zillow (``load_zillow``) -- metro-tier house-value panel (top vs bottom price
    tier).  This is the SHIPPED empirical application run by ``run_all.py``.
  * Metro unemployment (``load_metro``) -- BLS LAUS metro-area annual-average
    unemployment rates.  OPTIONAL / not in the default run; build it with
    ``data/metro/build_metro_panel.py`` and see the README (section 7) and
    ``data/metro/README.md`` for download + preparation.

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
from .factorridge import fit_factor_ridge
from .onestep import (companion_p2, delta_se, irf_p2, joint_cov, joint_cov_xs,
                      lrm_p2)
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


def _looks_like_date(s: str) -> bool:
    s = s.strip()
    return len(s) >= 6 and (s[:4].isdigit() and ("-" in s or "/" in s))


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- #
#  AR(2) panel construction
# --------------------------------------------------------------------------- #
def build_ar2(Ymat: np.ndarray, covars=None):
    """From a (T x N) stationary series build ``(Y_eff, [lag1, lag2, *covars])``.

    ``covars`` is an optional list of (T x N) covariate matrices aligned to
    ``Ymat`` by (time, unit).  Each enters as a PREDETERMINED (once-lagged)
    regressor ``X_{i,t-1}`` -- aligned with ``lag1`` -- so weak exogeneity
    (a:exog) is preserved.  The returned design list is ordered
    ``[lag1, lag2, covar_1, ..., covar_M]``; ``build_blocks`` then appends the
    interactive H block, giving ``B = M + 3`` coefficient blocks in total."""
    T, N = Ymat.shape
    Y = Ymat[2:]                                # effective sample t = 3..T
    lag1 = Ymat[1:-1]
    lag2 = Ymat[0:-2]
    Z = [lag1, lag2]
    for c in (covars or []):
        c = np.asarray(c, dtype=float)
        if c.shape != Ymat.shape:
            raise ValueError("each covariate must match Ymat shape (T, N)")
        Z.append(c[1:-1])                       # X_{i,t-1}: predetermined, aligned to lag1
    return Y, Z


# --------------------------------------------------------------------------- #
#  targets
# --------------------------------------------------------------------------- #
def _cellmean_dir(blocks, block, W):
    D = [np.zeros_like(zb) for zb in blocks]
    D[block] = W
    return D


def ar2_targets(blocks, Tp, N, groups=None, group_labels=("g0", "g1"),
                covar_names=()):
    """Global-mean lag1/lag2 (+ optional group means/contrast), plus a global-mean
    target for each covariate coefficient block.  Covariate blocks occupy indices
    ``2, 3, ..., 1+len(covar_names)`` (block order ``[lag1, lag2, covars..., H]``)."""
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
    for m, nm in enumerate(covar_names):                 # covariate coefficient means
        targets.append(Target(f"{nm}_mean", 2 + m, _cellmean_dir(blocks, 2 + m, Wall)))
    return targets


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
def run_ar2(Ymat, tuning: Tuning, groups=None, group_labels=("g0", "g1"),
            rng=None, covars=None, covar_names=()):
    """Estimate the heterogeneous AR(2) and all targets; add derived dynamics.

    ``covars`` (list of T x N matrices) and ``covar_names`` add predetermined
    covariate regressor blocks; each gets a global-mean coefficient target named
    ``f"{name}_mean"``.  The lag-1/lag-2 dynamics (companion radius, IRF) are
    unchanged -- the covariate blocks enter the model but the dynamic summaries
    remain functions of the autoregressive coefficients."""
    if rng is None:
        rng = np.random.default_rng(0)
    Y, Z = build_ar2(Ymat, covars)
    Tp, N = Y.shape
    blocks = build_blocks(Z)
    targets = ar2_targets(blocks, Tp, N, groups=groups, group_labels=group_labels,
                          covar_names=covar_names)
    res = estimate(Y, Z, targets, tuning, P=2, rng=rng)

    table = {}
    for tg in targets:
        table[tg.name] = dict(est=res.estimates[tg.name], se=res.se[tg.name],
                              se_xs=res.se_xs[tg.name], ci=res.ci[tg.name],
                              ci_xs=res.ci_xs[tg.name])

    # Derived dynamics from the lag1/lag2 global means via the delta method.
    # CUMULATIVE PERSISTENCE a+b is a LINEAR combination of the lag-mean targets,
    # so it is an a:target-regular target covered by thm:xs_dependence: it carries
    # BOTH the White delta-method s.e. AND the within-period cross-sectional
    # (cluster) s.e., exactly like the scalar lag means.  LRM and the horizon IRFs
    # are NONLINEAR transforms; the paper defers their dependence-robust inference
    # to the optional response-path bootstrap (app:irf_bootstrap), so they are
    # reported with the baseline White delta-method studentizer only.
    a = res.estimates["lag1_mean"]; b = res.estimates["lag2_mean"]
    Sig = joint_cov(res.onestep, ["lag1_mean", "lag2_mean"])
    Sig_xs = joint_cov_xs(res.onestep, ["lag1_mean", "lag2_mean"],
                          kernel=tuning.xs_kernel, bandwidth=tuning.xs_bandwidth)
    one = np.array([1.0, 1.0])
    cum = a + b
    cum_se = float(np.sqrt(max(one @ Sig @ one, 0)))            # White
    cum_se_xs = float(np.sqrt(max(one @ Sig_xs @ one, 0)))      # cross-sectional
    lrm_val, lrm_g = lrm_p2(a, b)
    lrm_se = delta_se(res.onestep, ["lag1_mean", "lag2_mean"], lrm_g)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion_p2(a, b)))))
    irfs = {}
    for h in (1, 2, 4, 8):
        val, g = irf_p2(a, b, h)
        irfs[h] = dict(est=val,
                       se=delta_se(res.onestep, ["lag1_mean", "lag2_mean"], g))

    derived = dict(cumulative_persistence=dict(est=cum, se=cum_se, se_xs=cum_se_xs),
                   long_run_multiplier=dict(est=lrm_val, se=lrm_se),
                   companion_radius=radius, irf=irfs)

    # Per-cell heterogeneity of the lag-1 surface, for the reproducible histogram
    # figure.  One full-sample first-stage fit at the selected ranks recovers the
    # estimated coefficient surface a_{ti}; we store its histogram (shared bins),
    # split by group, so the figure renders without shipping the whole T x N grid.
    fit = fit_factor_ridge(Y, blocks, res.ranks, mask=None, ridge=tuning.ridge,
                           n_sweeps=tuning.n_sweeps, tol=tuning.tol,
                           n_restarts=tuning.n_restarts, rng=rng)
    a_surf = np.asarray(fit.surfaces[0], dtype=float)
    flat = a_surf.ravel()
    lo, hi = float(np.percentile(flat, 0.5)), float(np.percentile(flat, 99.5))
    edges = np.linspace(lo, hi, 41)
    hist = dict(edges=edges.tolist(),
                counts_all=np.histogram(flat, bins=edges)[0].tolist(),
                mean=float(flat.mean()),
                q05=float(np.percentile(flat, 5)),
                q50=float(np.percentile(flat, 50)),
                q95=float(np.percentile(flat, 95)))
    if groups is not None:
        g = np.asarray(groups).astype(int)            # length-N unit labels
        hist["labels"] = list(group_labels)
        hist["counts_g0"] = np.histogram(a_surf[:, g == 0].ravel(),
                                         bins=edges)[0].tolist()
        hist["counts_g1"] = np.histogram(a_surf[:, g == 1].ravel(),
                                         bins=edges)[0].tolist()
    derived["coef_hist"] = hist

    return dict(targets=table, derived=derived, ranks=res.ranks,
                q=res.q, J=res.J, Tp=Tp, N=N)


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
