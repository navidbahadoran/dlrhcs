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


def _pmap(fn, items, n_jobs=1):
    """Map ``fn`` over ``items``; parallel via joblib loky when ``n_jobs != 1``."""
    items = list(items)
    if n_jobs and n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(x) for x in items)
        except Exception:
            pass
    return [fn(x) for x in items]


# --------------------------------------------------------------------------- #
#  prediction criterion
# --------------------------------------------------------------------------- #
def effective_dim(ranks, Tp, N) -> float:
    return float(sum(r * (Tp + N - r) for r in ranks))


def rank_penalty(sigma2, Tp, N, J, kappa_c=1.0, q=8.0, L_TN=None) -> float:
    """Explicit rank penalty kappa_TN (eq:explicit_rank_penalty; operational
    residual-scaled form of app:roadmap Step 4):

        kappa_TN = c_kappa * sigma^2 * b_TN^2 * ell_TN^2 * zeta_TN^{-1/2}
                 = c_kappa * sigma^2 * b_TN * ell_TN / sqrt(a_TN),

    with (sec:assumptions / eq:explicit_rank_penalty)
        a_TN    = 1/T + 1/N,
        ell_TN  = sqrt(log(TN * J)),                 ell_TN^2 = log(TN J),
        b_TN    = (TN)^{1/q} * L_TN,   q = 8 + eta,  L_TN slowly diverging,
        zeta_TN = a_TN * b_TN^2 * ell_TN^2.

    Defaults: q = 8 (the minimal-moment boundary, the most conservative b_TN),
    and L_TN = max(1, loglog(TN)) (an arbitrarily slowly diverging envelope).

    c_kappa is the free tuning constant.  IMPORTANT: with this exact b_TN the raw
    per-rank penalty kappa * (T+N)/(TN) = c_kappa * zeta_TN^{1/2} is ~2.8 sigma^2
    at c_kappa = 1, which OVER-penalizes the weakly-identified lag block (verified:
    P(r_hat = r_0) = 0 at c_kappa = 1 on the baseline DGP).  The configs therefore
    use a calibrated c_kappa ~ 0.03, which matches the validated per-rank scale.
    Report the selected ranks over a c_kappa grid (e.g. {0.02, 0.03, 0.05}) as a
    sensitivity check.
    """
    a = 1.0 / Tp + 1.0 / N
    ell = np.sqrt(np.log(Tp * N * max(J, 1)))
    if L_TN is None:
        L_TN = max(1.0, np.log(np.log(Tp * N)))
    b = (Tp * N) ** (1.0 / q) * L_TN
    return float(kappa_c * sigma2 * b * ell / np.sqrt(a))


def select_ranks(Y, blocks, candidates, folds, kappa, fit_kwargs, n_jobs=1):
    """Return (r_hat, table) minimizing CV loss + penalty over candidates.

    The candidate x fold first-stage fits are independent and dominate the cost
    (|candidates| * |folds| factor-ridge fits), so they run in parallel over
    ``n_jobs`` cores.  Each fit gets its own seed derived from the base rng and
    the (candidate, fold) indices, so the selected rank is reproducible and does
    NOT depend on n_jobs.
    """
    Tp, N = Y.shape
    cand = [tuple(r) for r in candidates]
    base = fit_kwargs.get("rng", None)
    seed0 = int(base.integers(2 ** 31)) if base is not None else 0
    fk = {k: v for k, v in fit_kwargs.items() if k != "rng"}
    tasks = [(ci, fi) for ci in range(len(cand)) for fi in range(len(folds))]

    def _fit(task):
        ci, fi = task
        fd = folds[fi]
        rng = np.random.default_rng(np.random.SeedSequence([seed0, ci, fi]))
        fit = fit_factor_ridge(Y, blocks, cand[ci], mask=fd.train, rng=rng, **fk)
        Rr = Y - A(fit.surfaces, blocks)
        return ci, float(np.sum(Rr[fd.val] ** 2))

    loss = [0.0] * len(cand)
    for ci, l in _pmap(_fit, tasks, n_jobs):
        loss[ci] += l

    rows, best = [], None
    for ci, r in enumerate(cand):
        L = loss[ci] / (Tp * N)
        d = effective_dim(r, Tp, N)
        crit = L + kappa * d / (Tp * N)
        rows.append((r, L, d, crit))
        key = (crit, d, r)
        if best is None or key < best[0]:
            best = (key, r)
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


def _persistence(surfaces, P, Tp, N):
    """General (P>=1) roadmap persistence (app:roadmap Step 0): the max over
    cells and horizons of the per-cell companion-PRODUCT modulus
    ``||C_{t+h,i} ... C_{t+1,i}||^{1/h}``, where C_{ti} is the P x P companion of
    the first P lag surfaces (top row = lag coefficients, ones on the subdiagonal).
    For P=1 this collapses to the geometric mean of |a| (the fast path); for P>=2
    it uses the full companion product, not a single-lag or L1 shortcut."""
    if P == 1:
        return _persistence_p1(surfaces[0], Tp, N)
    # Paper's exact persistence (app:roadmap, .tex:6474-6487):
    #   rho_hat = min{ 0.99, max_{1<=h<=H} max_{t,i}
    #                  ||C_{t,i} C_{t-1,i} ... C_{t-h+1,i}||_op^{1/h} },
    # the max over cells AND horizons of the per-cell companion-product operator
    # norm.  Every length-h window is covered (range(Tp), h up to min(H, Tp-t)),
    # so the last window is included.  NOTE: the companion is non-normal, so
    # ||C||_2 can exceed 1 transiently even for a stable system; the h=1 term then
    # pushes rho_hat toward the 0.99 cap for AR(2).  That is a property of this
    # formula (it yields a conservative, i.e. larger, exclusion window q via
    # Step 1), not a bug.
    H = min(max(1, int(np.ceil(np.log(Tp * N)))), Tp)
    C = np.zeros((Tp, N, P, P))
    for p in range(P):
        C[:, :, 0, p] = surfaces[p]                 # top row = lag coefficients
    for p in range(1, P):
        C[:, :, p, p - 1] = 1.0                     # subdiagonal ones
    best = 0.0
    for t in range(Tp):                             # every window, batched over units
        M = np.broadcast_to(np.eye(P), (N, P, P)).copy()
        for h in range(1, min(H, Tp - t) + 1):
            M = np.matmul(C[t + h - 1], M)          # C_{t+h-1} ... C_t  (h consecutive)
            nrm = np.linalg.norm(M, ord=2, axis=(1, 2))   # operator norm per unit
            best = max(best, float(np.max(nrm ** (1.0 / h))))
    return min(0.99, best)


def roadmap(Y, Z_list, P=1, r_work=None, r_bar=None, kappa_c=1.0, tau_tr=0.45,
            fit_kwargs=None, r_buffer=0):
    """Run roadmap Steps 0-4; returns a :class:`Roadmap`.

    ``r_bar`` are the FIXED rank caps that define the candidate box
    (eq:fixed_candidate_box); if ``None`` they default to the generous working
    rank ``r_work``.
    """
    fit_kwargs = fit_kwargs or {}
    blocks = build_blocks(Z_list)
    Tp, N = Y.shape
    B = len(blocks)
    if r_work is None:
        r_work = tuple([2] * B)

    # Step 0: working fit -> persistence, residual scale
    fit = fit_factor_ridge(Y, blocks, r_work, mask=None, **fit_kwargs)
    sigma2 = float(np.var(Y - A(fit.surfaces, blocks)))
    rho = _persistence(fit.surfaces, P, Tp, N)

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

    # Step 2: folds.  Retention is computed for the FULL space-time buffer volume
    # (q+1)(2 r_buffer + 1), so the chosen J provisions training for the spatial
    # radius actually in force (a:folds), not just the forward window.
    J = 6
    for Jc in (6, 8, 10, 12):
        if retained_share(Jc, q, r_buffer) >= tau_tr:
            J = Jc
            break

    # Step 3: FIXED deterministic candidate box prod_m {0,...,rbar_m}
    # (eq:fixed_candidate_box).  The caps rbar are fixed inputs chosen >= the
    # true ranks (Assumption a:signal); zero ranks are admitted so the selector
    # can drop an absent block.  The preliminary SVD is used only to ORDER the
    # candidate fits, never to PRUNE the box -- and since the full box is
    # searched, ordering is moot here.  Default caps = the generous working rank.
    rbar = tuple(r_bar) if r_bar is not None else tuple(r_work)
    candidates = [tuple(c) for c in itertools.product(
        *[range(0, rb + 1) for rb in rbar])]

    # Step 4: explicit penalty kappa_TN = c_kappa sigma^2 b^2 ell^2 zeta^{-1/2}
    # (eq:explicit_rank_penalty), computed by the shared helper.
    kappa = rank_penalty(sigma2, Tp, N, J, kappa_c)

    return Roadmap(rho_hat=rho, sigma2_hat=sigma2, q=q, J=J,
                   candidates=candidates, kappa=float(kappa), r_work=r_work)
