"""
Rank selection and the data-driven roadmap (spec sec 7, app:roadmap).

Cross-fitted prediction criterion over a candidate box ``R``:

    L_hat(r)   = (1/TpN) sum_j || Pi_j { Y - A(Theta_hat^0_{-j}(r)) } ||_F^2
    d(r)       = sum_b r_b (Tp + N - r_b)
    r_hat      = argmin_r  L_hat(r) + kappa_TN * d(r) / (Tp*N).

Roadmap Steps 0-4 produce, from one working-rank full-sample fit, the
persistence rho_hat, the forward window q, the fold count J, the candidate box,
and the penalty kappa_TN.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .design import A, build_blocks
from .factorridge import fit_factor_ridge
from .folds import make_folds, retained_share


# --------------------------------------------------------------------------- #
#  prediction criterion
# --------------------------------------------------------------------------- #
def cv_loss(Y, blocks, folds, ranks, fit_kwargs) -> float:
    Tp, N = Y.shape
    tot = 0.0
    for fd in folds:
        fit = fit_factor_ridge(Y, blocks, ranks, mask=fd.train, **fit_kwargs)
        R = (Y - A(fit.surfaces, blocks))
        tot += float(np.sum((R[fd.val]) ** 2))
    return tot / (Tp * N)


def effective_dim(ranks, Tp, N) -> float:
    return float(sum(r * (Tp + N - r) for r in ranks))


def select_ranks(Y, blocks, candidates, folds, kappa, fit_kwargs):
    """Return (r_hat, table) minimizing CV loss + penalty over candidates."""
    Tp, N = Y.shape
    rows = []
    best = None
    for r in candidates:
        L = cv_loss(Y, blocks, folds, r, fit_kwargs)
        d = effective_dim(r, Tp, N)
        crit = L + kappa * d / (Tp * N)
        rows.append((tuple(r), L, d, crit))
        key = (crit, d, tuple(r))
        if best is None or key < best[0]:
            best = (key, tuple(r))
    return best[1], rows


# --------------------------------------------------------------------------- #
#  roadmap
# --------------------------------------------------------------------------- #
@dataclass
class Roadmap:
    rho_hat: float
    sigma2_hat: float
    q: int
    J: int
    candidates: List[tuple]
    kappa: float
    r_work: tuple


def _persistence_p1(A_surface, Tp, N):
    """rho_hat for P=1: companion modulus = |a|; horizon-averaged max."""
    H_TN = max(1, int(np.ceil(np.log(Tp * N))))
    a = np.abs(A_surface)
    best = 0.0
    # geometric-mean of |a| products along time, per unit, for horizons 1..H_TN
    for h in range(1, H_TN + 1):
        if h > Tp:
            break
        # rolling product of h consecutive |a| along time, then ^(1/h)
        logabs = np.log(np.maximum(a, 1e-12))
        csum = np.cumsum(logabs, axis=0)
        prod = csum[h - 1:] - np.vstack([np.zeros((1, N)), csum[:-h]])[: csum.shape[0] - h + 1]
        mod = np.exp(prod / h)
        best = max(best, float(np.max(mod)))
    return min(0.99, best)


def roadmap(Y, Z_list, P=1, r_work=None, kappa_c=1.0, tau_tr=0.45,
            tau_sv=0.15, fit_kwargs=None):
    """Run roadmap Steps 0-4; returns a :class:`Roadmap`."""
    fit_kwargs = fit_kwargs or {}
    blocks = build_blocks(Z_list)
    Tp, N = Y.shape
    B = len(blocks)
    if r_work is None:
        r_work = tuple([2] * B)

    # Step 0: working fit -> persistence, residual scale
    fit = fit_factor_ridge(Y, blocks, r_work, mask=None, **fit_kwargs)
    sigma2 = float(np.var(Y - A(fit.surfaces, blocks)))
    rho = _persistence_p1(fit.surfaces[0], Tp, N)

    # Step 1: window q_TN = ceil(log(TN)/|log rho_hat|)  (app:roadmap Step 1).
    # For strongly persistent panels this is large; per para:capped_window the
    # paper sanctions capping q at a MODERATE value and letting the stability
    # margin + a moderate J control the residual same-unit feedback, with the
    # remaining O(q/J) leakage measured by the forward-exclusion-window sweep
    # (the `purge` stage).  The cap below is that device, not an oversight: it
    # trades a vanishing-leakage guarantee for finite-sample data retention, and
    # the purge sweep is its empirical defence.  (The shipped MC configs use a
    # fixed q anyway; this cap only binds when the data-driven roadmap is on.)
    q = int(np.ceil(np.log(Tp * N) / max(abs(np.log(rho)), 1e-6)))
    q = int(min(q, 8))

    # Step 2: folds
    J = 6
    for Jc in (6, 8, 10, 12):
        if retained_share(Jc, q) >= tau_tr:
            J = Jc
            break

    # Step 3: candidate box from singular-value screening of working fit
    rbar = []
    for b in range(B):
        s = fit.svals[b]
        s = s[s > 0]
        if len(s) <= 1:
            rbar.append(1)
            continue
        thresh = tau_sv * s[0]
        rb = int(np.sum(s > thresh))
        rbar.append(max(1, rb))
    candidates = [tuple(c) for c in itertools.product(
        *[range(1, rb + 2) for rb in rbar])]

    # Step 4: penalty kappa_TN = c_kappa * sigma^2 * ell^2_TN * loglog(TN)
    # (app:roadmap Step 4).  The design-localization factor ell_TN is, by
    # assumption, an O(1) (slowly growing) bound on the *normalized* design
    # |Z^(m)_ti| <= C_Z ell_TN; for the standardized regressors used here it is a
    # constant, so it is absorbed into the free tuning constant c_kappa (kappa_c).
    # (Using a literal max|Z|^2 would conflate the design SCALE with the
    # localization and over-penalize -- it collapses P(r_hat = r_0) to ~0 in the
    # MC -- so ell^2 is kept in c_kappa.)  The loglog scale keeps the selector
    # consistent (verified: P(correct rank) -> 1 in experiments.rank_consistency).
    kappa = kappa_c * sigma2 * np.log(np.log(Tp * N))

    return Roadmap(rho_hat=rho, sigma2_hat=sigma2, q=q, J=J,
                   candidates=candidates, kappa=float(kappa), r_work=r_work)
