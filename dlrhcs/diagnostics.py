"""Empirical model diagnostics for the heterogeneous low-rank AR(p) fits.

Pure, fit-agnostic helpers used by :func:`dlrhcs.empirical.run_ar2` to report

  * residual adequacy  -- mean, sd, RMSE, temporal autocorrelation by lag, the
    average cross-sectional residual correlation, and the residual first-singular-
    value share (leftover common-factor strength);
  * goodness of fit    -- R^2 against the outcome and against a no-dynamics
    baseline that keeps only the interactive block;
  * heterogeneity      -- mean / sd / 5-50-95 percentiles of the estimated lag-1
    coefficient surface, overall and by group.

All operate on the FULL-sample first-stage residual matrix, so they add one SVD and
(for the no-dynamics R^2) one cheap H-only fit -- no extra debiasing.
"""
from __future__ import annotations

import numpy as np

from .design import A, build_blocks
from .factorridge import fit_factor_ridge


def _autocorr(R, max_lag):
    """Mean over units of the lag-k temporal autocorrelation of the residuals."""
    Rc = R - R.mean(axis=0, keepdims=True)
    denom = (Rc ** 2).sum(axis=0)
    out = []
    for k in range(1, max_lag + 1):
        num = (Rc[k:] * Rc[:-k]).sum(axis=0)
        ac = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        out.append(float(np.mean(ac)))
    return out


def _avg_xs_resid_corr(R):
    """Average pairwise cross-sectional correlation of the unit residual series --
    a Pesaran-style read-out of leftover contemporaneous dependence.  Zero if the
    interactive block has absorbed the common comovement."""
    Rc = R - R.mean(axis=0, keepdims=True)
    sd = Rc.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        return 0.0
    Z = Rc[:, keep] / sd[keep]
    C = (Z.T @ Z) / Z.shape[0]
    n = C.shape[0]
    return float((C.sum() - np.trace(C)) / (n * (n - 1)))


def residual_diagnostics(R, y_var, max_lag=6):
    """Residual adequacy battery from the (Tp x N) residual matrix ``R``."""
    R = np.asarray(R, dtype=float)
    s = np.linalg.svd(R, compute_uv=False)
    sv2 = s ** 2
    rv = float(R.var())
    return dict(resid_mean=float(R.mean()),
                resid_sd=float(R.std()),
                rmse=float(np.sqrt(np.mean(R ** 2))),
                resid_var=rv,
                r2_vs_outcome=float(1.0 - rv / y_var) if y_var > 0 else 0.0,
                autocorr=_autocorr(R, max_lag),
                avg_xs_resid_corr=_avg_xs_resid_corr(R),
                first_sv_share=float(sv2[0] / sv2.sum()) if sv2.sum() > 0 else 0.0,
                n_lags=int(max_lag))


def _qstats(x):
    x = np.asarray(x, dtype=float).ravel()
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q05=float(np.percentile(x, 5)),
                q50=float(np.percentile(x, 50)),
                q95=float(np.percentile(x, 95)))


def heterogeneity_stats(a_surf, groups=None, group_labels=("g0", "g1")):
    """Dispersion of the estimated lag-1 surface, overall and by group."""
    a_surf = np.asarray(a_surf, dtype=float)
    out = dict(overall=_qstats(a_surf))
    if groups is not None:
        g = np.asarray(groups).astype(int)
        out["by_group"] = {group_labels[0]: _qstats(a_surf[:, g == 0]),
                           group_labels[1]: _qstats(a_surf[:, g == 1])}
    return out


def no_dynamics_resid_var(Y, blocks, ranks, fit_kwargs):
    """Residual variance of a no-dynamics baseline that zeroes every lag/covariate
    rank and keeps only the interactive block (last block).  Used for the R^2
    improvement of the full dynamic model over 'common factor only'."""
    base = tuple([0] * (len(ranks) - 1) + [int(ranks[-1])])
    fit = fit_factor_ridge(Y, blocks, base, **fit_kwargs)
    Rb = np.asarray(Y, dtype=float) - A(fit.surfaces, blocks)
    return float(Rb.var())
