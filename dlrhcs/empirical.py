"""
Empirical applications: the heterogeneous low-rank dynamic panel applied to two real
metropolitan panels, run via ``scripts/zillow_abc.py`` and ``scripts/unemp_abc.py``.

    y_tilde_{it} = a_{ti} y_tilde_{i,t-1} (+ b_{ti} y_tilde_{i,t-2})
                   (+ sum_m beta^m_{ti} X^m_{i,t-1}) + h_{ti} + u_{it}

estimated by the cross-fitted debiased one-step.  Blocks are
``[lag1, lag2, covariates..., ones]``; :func:`run_ar2` threads optional predetermined
covariate blocks and reports, per specification, the lag / group / contrast targets
(each with a White and a within-period cross-sectional s.e.), cumulative persistence
(global and by group), the long-run multiplier, horizon IRFs, the companion spectral
radius, and a diagnostics battery (plug-in vs debiased, residual adequacy, fit,
coefficient heterogeneity), plus rank- and covariate-robustness sweeps.

Datasets:
  * Housing (:func:`load_zillow`) -- Zillow ZHVI metro-tier panel (top vs bottom price
    tier), monthly log price growth; heterogeneous AR(2).
  * Unemployment (:func:`dlrhcs.unemp.load_unemp_panel`) -- BLS LAUS monthly metro
    unemployment rate, not-seasonally-adjusted and deseasonalized; heterogeneous AR(1).

Covariates for specification C are loaded by :mod:`dlrhcs.covariates`.  Each application
is run in three specifications -- A (full sample), B (covariate window, no covariates),
C (covariate-augmented) -- so that A->B isolates the sample-restriction effect and
B->C the covariate effect.  :func:`homogeneous_benchmark` gives the pooled
common-coefficient comparison.  Provenance: each output records the data file's content
hash (``data_fingerprint``) so a Data Editor can confirm the exact vintage.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .design import A, build_blocks
from .diagnostics import (heterogeneity_stats, no_dynamics_resid_var,
                          residual_diagnostics)
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


def load_zillow(path_top, path_bottom, stationarize=True, start=None, end=None):
    """Zillow ZHVI tier panel: parse RAW price levels, stack top & bottom tiers
    as units, balance, then (default) apply the minimal per-series
    stationarization and standardize.  House-price levels are a unit root, so in
    practice nearly every series is log-differenced (returns).

    ``start``/``end`` (``"YYYY-MM"``) optionally window the monthly columns -- used
    for the restricted-sample specifications B and C.  The return dict adds
    ``months`` (the effective post-differencing month labels) and ``col_region``
    (the metro RegionName for each panel column), so covariate matrices can be
    aligned to the panel by :func:`dlrhcs.covariates.load_zillow_covariates`."""
    import csv

    def read(path):
        with open(path) as fh:
            rows = list(csv.reader(fh))
        head = rows[0]
        dcols = [(k, head[k][:7]) for k in range(len(head)) if _looks_like_date(head[k])]
        if start:
            dcols = [(k, ym) for k, ym in dcols if ym >= start]
        if end:
            dcols = [(k, ym) for k, ym in dcols if ym <= end]
        cols = [k for k, _ in dcols]
        labels = [ym for _, ym in dcols]
        ids, mat = [], []
        for r in rows[1:]:
            if r[3] != "msa":
                continue
            ids.append(r[2])
            mat.append([_to_float(r[k]) for k in cols])
        return ids, np.array(mat), labels

    idt, top, labels = read(path_top)
    idb, bot, _ = read(path_bottom)
    common = [r for r in idt if r in set(idb)]
    it = {r: k for k, r in enumerate(idt)}
    ib = {r: k for k, r in enumerate(idb)}
    levels = np.vstack([np.array([top[it[r]] for r in common]),
                        np.array([bot[ib[r]] for r in common])]).T   # (T x 2*regions)
    good = ~np.any(~np.isfinite(levels), axis=0)
    levels = levels[:, good]
    col_region = np.array(common + common)[good]                     # RegionName per column
    tier = np.array([0] * len(common) + [1] * len(common))[good]
    if stationarize:
        X, transforms = stationarize_panel(levels)
        months = labels[1:]                          # differencing drops the first month
    else:
        X, transforms = levels, ["level"] * levels.shape[1]
        months = labels
    vol = X.std(0)
    X = (X - X.mean(0)) / X.std(0)
    return dict(Y=X, tier=tier, vol=vol, transforms=transforms,
                n_differenced=int(sum(t != "level" for t in transforms)),
                regions=common, col_region=col_region, months=months,
                T=X.shape[0], N=X.shape[1],
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
            Target(f"lag2_{group_labels[0]}", 1, _cellmean_dir(blocks, 1, W0)),
            Target(f"lag2_{group_labels[1]}", 1, _cellmean_dir(blocks, 1, W1)),
        ]
    for m, nm in enumerate(covar_names):                 # covariate coefficient means
        targets.append(Target(f"{nm}_mean", 2 + m, _cellmean_dir(blocks, 2 + m, Wall)))
    return targets


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
def run_ar2(Ymat, tuning: Tuning, groups=None, group_labels=("g0", "g1"),
            rng=None, covars=None, covar_names=(), return_onestep=False):
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
                              ci_xs=res.ci_xs[tg.name],
                              plugin=res.onestep.plugins.get(tg.name))

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
    for h in (1, 2, 4, 8, 12):
        val, g = irf_p2(a, b, h)
        irfs[h] = dict(est=val,
                       se=delta_se(res.onestep, ["lag1_mean", "lag2_mean"], g))

    derived = dict(cumulative_persistence=dict(est=cum, se=cum_se, se_xs=cum_se_xs),
                   long_run_multiplier=dict(est=lrm_val, se=lrm_se),
                   companion_radius=radius, irf=irfs)

    # Per-group cumulative persistence a+b (linear, so both s.e.'s are valid).  For
    # the AR(1) unemployment fit b==0, so the group cumulative equals the group lag-1.
    if groups is not None:
        gcum = {}
        for lab in group_labels:
            ag = res.estimates[f"lag1_{lab}"]; bg = res.estimates[f"lag2_{lab}"]
            Sg = joint_cov(res.onestep, [f"lag1_{lab}", f"lag2_{lab}"])
            Sg_xs = joint_cov_xs(res.onestep, [f"lag1_{lab}", f"lag2_{lab}"],
                                 kernel=tuning.xs_kernel, bandwidth=tuning.xs_bandwidth)
            gcum[lab] = dict(est=ag + bg,
                             se=float(np.sqrt(max(one @ Sg @ one, 0))),
                             se_xs=float(np.sqrt(max(one @ Sg_xs @ one, 0))))
        derived["cumulative_persistence_by_group"] = gcum

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

    # Coefficient time-series and per-unit averages, for the educative figures.
    # a_surf is the lag-1 surface a_{ti}; surfaces[1] is the lag-2 surface b_{ti}
    # (a (Tp x N) zero matrix when the second lag has rank 0, e.g. the AR(1)
    # unemployment fit).  a_t,b_t average across units within each period; a_i,b_i
    # average across periods within each unit.  Month and region labels are attached
    # by the calling script, which holds the panel's index.
    b_surf = (np.asarray(fit.surfaces[1], dtype=float)
              if len(fit.surfaces) > 1 else np.zeros_like(a_surf))
    a_t, b_t = a_surf.mean(axis=1), b_surf.mean(axis=1)
    a_i, b_i = a_surf.mean(axis=0), b_surf.mean(axis=0)
    derived["coef_path"] = dict(a_t=a_t.tolist(), b_t=b_t.tolist(),
                                cum_t=(a_t + b_t).tolist())
    derived["coef_by_unit"] = dict(a_i=a_i.tolist(), cum_i=(a_i + b_i).tolist())

    # Residual adequacy, goodness of fit, and coefficient-surface heterogeneity,
    # all from the full-sample first-stage residual matrix (one SVD + one cheap
    # H-only baseline fit; no extra debiasing).  Plus solver/fold diagnostics from
    # the one-step result, and the panel's effective dimensions.
    Rfull = Y - A(fit.surfaces, blocks)
    rd = residual_diagnostics(Rfull, float(np.var(Y)))
    derived["residual_diag"] = rd
    derived["heterogeneity"] = heterogeneity_stats(a_surf, groups, group_labels)
    fitkw = dict(ridge=tuning.ridge, n_sweeps=tuning.n_sweeps, tol=tuning.tol,
                 n_restarts=tuning.n_restarts, rng=rng)
    base_var = no_dynamics_resid_var(Y, blocks, res.ranks, fitkw)
    derived["fit"] = dict(rmse=rd["rmse"], r2_vs_outcome=rd["r2_vs_outcome"],
                          r2_vs_nodynamics=float(1.0 - rd["resid_var"] / base_var)
                          if base_var > 0 else 0.0)
    rz = res.onestep.riesz_diag
    conv = [c for tg in rz.values() for c in tg.get("converged", [])]
    cgit = [c for tg in rz.values() for c in tg.get("cg_iters", [])]
    meig = [c for tg in rz.values() for c in tg.get("min_eig", [])]
    robj = list(getattr(fit, "restart_objs", []) or [])
    restart_disp = (float((max(robj) - min(robj)) / (1.0 + abs(min(robj))))
                    if len(robj) > 1 else 0.0)
    derived["solver"] = dict(monotone=bool(res.diagnostics.get("monotone", True)),
                             retained=float(res.diagnostics.get("retained", float("nan"))),
                             cg_converged_frac=float(np.mean(conv)) if conv else 1.0,
                             cg_iters_mean=float(np.mean(cgit)) if cgit else 0.0,
                             min_eig_mean=float(np.mean(meig)) if meig else 0.0,
                             obj_rel_improve_final=float(getattr(fit, "obj_rel_improve", 0.0)),
                             n_restarts=len(robj),
                             restart_obj_dispersion=restart_disp)
    derived["data"] = dict(T=int(Ymat.shape[0]), Tp=int(Tp), N=int(N))

    out = dict(targets=table, derived=derived, ranks=res.ranks,
               q=res.q, J=res.J, Tp=Tp, N=N)
    if return_onestep:
        out["onestep"] = res.onestep          # in-memory only (not JSON-serialised)
    return out


def rank_robustness(Ymat, base_ranks, rH_values, base_tuning, groups=None,
                    group_labels=("g0", "g1"), covars=None, covar_names=(), seed=2024):
    """Re-estimate at several interactive-block ranks r_H (the LAST block), holding
    the lag and covariate ranks fixed at ``base_ranks``; report the headline dynamic
    summaries so the empirical conclusions can be shown robust to the nuisance rank
    (spec sec 13).  Works for AR(1)/AR(2) and for covariate-augmented designs."""
    import dataclasses
    out = {}
    for rH in rH_values:
        ranks = tuple(list(base_ranks[:-1]) + [int(rH)])
        t = dataclasses.replace(base_tuning, ranks=ranks)
        r = run_ar2(Ymat, t, groups=groups, group_labels=group_labels,
                    rng=np.random.default_rng(seed), covars=covars,
                    covar_names=covar_names)
        out[int(rH)] = dict(ranks=list(ranks),
                            lag1_mean=r["targets"]["lag1_mean"]["est"],
                            lag1_se=r["targets"]["lag1_mean"]["se"],
                            cumulative=r["derived"]["cumulative_persistence"]["est"],
                            radius=r["derived"]["companion_radius"])
    return out


def covariate_robustness(Ymat, base_ranks, base_tuning, covars, covar_names,
                         groups=None, group_labels=("g0", "g1"), seed=2024):
    """Forced-rank robustness for each covariate (spec sec 13, point 10).  For each
    covariate block in turn, re-estimate with that block's rank forced to 0 -- i.e.
    the covariate is present in the design but its coefficient is pinned at zero
    (effectively dropped) -- and report the resulting lag dynamics, so the autoregressive
    conclusions can be shown invariant to whether the covariate is retained.  Covariate
    blocks occupy indices ``2 .. 1+len(covar_names)`` in ``base_ranks``
    ``[lag1, lag2, cov1, ..., H]``."""
    import dataclasses
    out = {}
    for m, nm in enumerate(covar_names):
        ranks = list(base_ranks); ranks[2 + m] = 0
        t = dataclasses.replace(base_tuning, ranks=tuple(ranks))
        r = run_ar2(Ymat, t, groups=groups, group_labels=group_labels,
                    rng=np.random.default_rng(seed), covars=covars,
                    covar_names=covar_names)
        out[nm] = dict(dropped=nm, ranks=ranks,
                       lag1_mean=r["targets"]["lag1_mean"]["est"],
                       cumulative=r["derived"]["cumulative_persistence"]["est"],
                       radius=r["derived"]["companion_radius"])
    return out


def rank_selection_table(Ymat, base_tuning, covars=None, covar_names=(),
                         groups=None, group_labels=("g0", "g1"), top_k=8, seed=2024):
    """Run the cross-fitted rank criterion over the roadmap candidate box and return
    the selected rank vector plus the top-k candidates ranked by the criterion
    (spec sec 13, point 1): each row is (rank, CV loss, effective dim, criterion)."""
    import dataclasses
    Y, Z = build_ar2(Ymat, covars)
    blocks = build_blocks(Z)
    targets = ar2_targets(blocks, Y.shape[0], Y.shape[1], groups=groups,
                          group_labels=group_labels, covar_names=covar_names)
    t = dataclasses.replace(base_tuning, ranks=None, select=True, use_roadmap=True)
    res = estimate(Y, Z, targets, t, P=2, rng=np.random.default_rng(seed))
    rt = res.diagnostics.get("rank_table", [])
    rt = sorted(rt, key=lambda row: row[3])[:top_k]
    return dict(selected=list(res.ranks),
                candidates=[dict(rank=row[0], cv_loss=row[1], eff_dim=row[2],
                                 criterion=row[3]) for row in rt])


def homogeneous_benchmark(Ymat, P):
    """Pooled two-way fixed-effects AR(P): the homogeneous (common-coefficient)
    benchmark.  Returns dict(coef, cum, rmse, r2) -- the single common lag vector, its
    sum, the in-sample RMSE, and R^2 over the within-transformed outcome -- used to
    quantify what the heterogeneous low-rank estimator adds over a model that forces one
    coefficient for every cell.  The two-way within transform removes additive unit and
    time effects (the homogeneous analogue of the interactive block)."""
    Y = np.asarray(Ymat, dtype=float)
    T, N = Y.shape
    yt = Y[P:]
    Xl = [Y[P - l - 1:T - l - 1] for l in range(P)]

    def _w(M):
        return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()

    yw = _w(yt).ravel()
    Xw = np.column_stack([_w(x).ravel() for x in Xl])
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    resid = yw - Xw @ beta
    return dict(coef=[float(b) for b in beta], cum=float(beta.sum()),
                rmse=float(np.sqrt(np.mean(resid ** 2))),
                r2=float(1.0 - resid.var() / yw.var()))
