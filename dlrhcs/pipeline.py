"""
Full feasible pipeline (spec sec 11).

    estimate(Y, Z_list, P, K, targets, tuning) -> estimates, ses, intervals, diag

Steps: build designs; (optionally) run the roadmap for q/J/box/kappa; build the
scattered purged folds; (optionally) select ranks by the cross-fitted criterion;
refit Theta_hat^0_{-j} on each purged fold and form residuals; solve the feasible
Riesz weights per target; one-step debias; studentize (White + xs); intervals.

Set ``oracle=True`` and pass ``true_U/true_V`` to run the infeasible oracle
benchmark (true tangent spaces in the Riesz solve) -- the spec sec 12 checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .design import A, build_blocks
from .factorridge import fit_factor_ridge
from .folds import make_folds
from .onestep import FoldFit, OneStepResult, one_step, white_se, xs_se
from .ranks import roadmap, select_ranks
from .targets import Target


@dataclass
class Tuning:
    ranks: Optional[tuple] = None       # fixed ranks; else select / roadmap
    q: Optional[int] = None
    J: Optional[int] = None
    ridge: float = 0.02
    n_sweeps: int = 80
    n_restarts: int = 4
    tol: float = 1e-8
    scheme: str = "scatter"
    select: bool = False                # run rank selection over roadmap box
    use_roadmap: bool = False           # derive q/J/box/kappa from data
    kappa_c: float = 1.0
    alpha_level: float = 0.05
    riesz_ridge: float = 1e-8
    riesz_tol: float = 1e-10
    riesz_maxiter: int = 2000
    xs_bandwidth: Optional[int] = None   # spatial-kernel xs s.e. bandwidth (None=auto)
    xs_kernel: str = "bartlett"          # "bartlett" (spatial, sim) | "cluster" (empirical)
    n_jobs: int = 1                      # cores for rank selection (single-panel use)


@dataclass
class EstimateResult:
    estimates: Dict[str, float]
    se: Dict[str, float]
    se_xs: Dict[str, float]
    ci: Dict[str, tuple]
    ci_xs: Dict[str, tuple]
    ranks: tuple
    q: int
    J: int
    onestep: OneStepResult
    diagnostics: Dict = field(default_factory=dict)


def estimate(Y, Z_list, targets: Sequence[Target], tuning: Tuning,
             P=1, rng=None, foldid=None,
             oracle=False, true_U=None, true_V=None) -> EstimateResult:
    if rng is None:
        rng = np.random.default_rng(0)
    Y = np.asarray(Y, dtype=float)
    Tp, N = Y.shape
    blocks = build_blocks(Z_list)
    B = len(blocks)

    fit_kwargs = dict(ridge=tuning.ridge, n_sweeps=tuning.n_sweeps,
                      tol=tuning.tol, n_restarts=tuning.n_restarts, rng=rng)

    # ---- q, J, ranks, kappa --------------------------------------------------
    ranks, q, J, kappa, candidates = tuning.ranks, tuning.q, tuning.J, None, None
    if tuning.use_roadmap or tuning.select:
        rm = roadmap(Y, Z_list, P=P, kappa_c=tuning.kappa_c, fit_kwargs=fit_kwargs)
        q = q if q is not None else rm.q
        J = J if J is not None else rm.J
        kappa, candidates = rm.kappa, rm.candidates
    if q is None:
        q = 3
    if J is None:
        J = 6

    folds = make_folds(Tp, N, J, q, P=P, rng=rng,
                       scheme=tuning.scheme, foldid=foldid)

    if ranks is None:
        if tuning.select and candidates:
            ranks, _ = select_ranks(Y, blocks, candidates, folds, kappa, fit_kwargs,
                                    n_jobs=tuning.n_jobs)
        else:
            ranks = tuple([1] * B)

    # ---- per-fold purged fits ------------------------------------------------
    foldfits: List[FoldFit] = []
    mono_ok = True
    for fd in folds:
        fit = fit_factor_ridge(Y, blocks, ranks, mask=fd.train, **fit_kwargs)
        mono_ok = mono_ok and fit.monotone
        resid = Y - A(fit.surfaces, blocks)
        if oracle:
            U, V = true_U, true_V
        else:
            U, V = fit.U, fit.V
        foldfits.append(FoldFit(surfaces=fit.surfaces, U=U, V=V, residual=resid,
                                train=fd.train, val=fd.val, p=fd.p, alpha=fd.alpha))

    # ---- one-step + variances ------------------------------------------------
    res = one_step(blocks, foldfits, targets,
                   riesz_kwargs=dict(ridge=tuning.riesz_ridge, tol=tuning.riesz_tol, maxiter=tuning.riesz_maxiter))
    z = norm.ppf(1 - tuning.alpha_level / 2)
    se, se_xs, ci, ci_xs = {}, {}, {}, {}
    for tg in targets:
        s = white_se(res, tg.name)
        sx = xs_se(res, tg.name, bandwidth=tuning.xs_bandwidth, kernel=tuning.xs_kernel)
        e = res.estimates[tg.name]
        se[tg.name], se_xs[tg.name] = s, sx
        ci[tg.name] = (e - z * s, e + z * s)
        ci_xs[tg.name] = (e - z * sx, e + z * sx)

    diag = dict(monotone=mono_ok, ranks=ranks, q=q, J=J,
                retained=float(np.mean([fd.n_pur for fd in folds]) / (Tp * N)))
    return EstimateResult(estimates=res.estimates, se=se, se_xs=se_xs,
                          ci=ci, ci_xs=ci_xs, ranks=tuple(ranks), q=q, J=J,
                          onestep=res, diagnostics=diag)
