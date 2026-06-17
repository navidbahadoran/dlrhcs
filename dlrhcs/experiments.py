"""
Theorem-justification experiments.  Each function targets a specific result in
the manuscript and returns a small dict of summary statistics; the full-scale
versions are driven by run_all.py.  These are the experiments that turn "the
code runs" into "the code demonstrates the theorems".

  debiasing_demo            one-step correction removes first-order bias   (thm:feasible)
  rank_consistency          selector picks the true rank, P -> 1           (thm:rank_consistency)
  irf_lrm_coverage          delta-method IRF / LRM intervals cover         (cor:irf_body, thm:irf)
  xs_coverage               xs s.e. restores coverage under within-period
                            dependence while White under-covers            (thm:xs_dependence)
  contiguous_fold_singular  contiguous folds make the information map
                            singular (motivates scattered folds)           (lem:local_collinearity_singularity)
"""
from __future__ import annotations

import numpy as np

from .design import build_blocks
from .dgp import simulate
from .folds import make_folds, assign_folds
from .factorridge import fit_factor_ridge
from .mc import standard_targets, true_value
from .pipeline import Tuning, estimate
from .ranks import select_ranks, effective_dim, cv_loss
from .targets import mean_direction, project_tangent, riesz_weights, Target
from .onestep import irf_p1, lrm_p1
from scipy.stats import norm

Z95 = norm.ppf(0.975)


# --------------------------------------------------------------------------- #
#  main message 1: debiasing removes first-order bias (thm:feasible)
# --------------------------------------------------------------------------- #
def debiasing_demo(Tp, N, R, tuning: Tuning, master=2024, oracle=False,
                   target="lag_entry"):
    """Show the one-step correction removes the regularization (shrinkage) bias.

    The lag-ENTRY target is heavily shrunk by the factor ridge, so the plug-in is
    biased toward zero and a CI built around it under-covers; the debiased one is
    centered and covers.  Reports plug-in vs debiased bias and the coverage of a
    CI centred at each (same studentizer)."""
    pin, deb, cov_plug, cov_deb = [], [], [], []
    for rep in range(R):
        sa = np.random.default_rng(np.random.SeedSequence([master, rep]))
        sb = np.random.default_rng(np.random.SeedSequence([master + 1, rep]))
        p = simulate(Tp, N, sa)
        bl = build_blocks(p.Z)
        tgs, ctx = standard_targets(bl, Tp, N)
        r = estimate(p.Y, p.Z, tgs, tuning, rng=sb, oracle=oracle,
                     true_U=p.U, true_V=p.V)
        tg = [t for t in tgs if t.name == target][0]
        tv = true_value(p, tg, ctx)
        plug = r.onestep.plugins[target]; deb_est = r.estimates[target]
        se = r.se[target]
        pin.append(plug - tv); deb.append(deb_est - tv)
        cov_plug.append(int(plug - Z95 * se <= tv <= plug + Z95 * se))
        lo, hi = r.ci[target]; cov_deb.append(int(lo <= tv <= hi))
    pin, deb = np.array(pin), np.array(deb)
    return dict(R=R, target=target,
                plugin_bias=float(pin.mean()), plugin_absbias=float(np.abs(pin).mean()),
                debiased_bias=float(deb.mean()), debiased_absbias=float(np.abs(deb).mean()),
                plugin_cov=float(np.mean(cov_plug)), debiased_cov=float(np.mean(cov_deb)))


# --------------------------------------------------------------------------- #
#  thm:rank_consistency -- selector picks the truth with prob -> 1
# --------------------------------------------------------------------------- #
def rank_consistency(grid, R, tuning: Tuning, master=2024, candidates=None,
                     kappa_c=1.0):
    """For each (Tp,N) in grid, fraction of reps with r_hat == true rank (1,1,1)."""
    if candidates is None:
        candidates = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 2),
                      (0, 1, 1), (1, 1, 0)]
    true_rank = (1, 1, 1)
    out = {}
    fit_kwargs = dict(ridge=tuning.ridge, n_sweeps=tuning.n_sweeps,
                      n_restarts=tuning.n_restarts)
    for (Tp, N) in grid:
        hits = 0
        for rep in range(R):
            sa = np.random.default_rng(np.random.SeedSequence([master, Tp, rep]))
            p = simulate(Tp, N, sa)
            bl = build_blocks(p.Z)
            folds = make_folds(Tp, N, tuning.J, tuning.q, rng=sa)
            sig2 = float(np.var(p.Y))
            # kappa = c_k * sig^2 * ell^2 * loglog(TN); ell_TN = O(1) for the
            # standardized design, absorbed into c_k (a literal max|Z|^2 conflates
            # design scale with localization and over-penalizes -> P(correct)~0).
            kappa = kappa_c * sig2 * np.log(np.log(Tp * N))
            rhat, _ = select_ranks(p.Y, bl, candidates, folds, kappa, fit_kwargs)
            hits += int(tuple(rhat) == true_rank)
        out[(Tp, N)] = dict(R=R, p_correct=hits / R, true_rank=true_rank)
    return out


# --------------------------------------------------------------------------- #
#  cor:irf_body / thm:irf -- delta-method IRF and LRM intervals cover
# --------------------------------------------------------------------------- #
def irf_lrm_coverage(Tp, N, R, tuning: Tuning, horizons=(1, 2, 4), master=2024,
                     oracle=False):
    """Coverage of delta-method IRF(h) and LRM intervals for the lag full mean."""
    acc = {f"irf{h}": [] for h in horizons}; acc["lrm"] = []
    for rep in range(R):
        sa = np.random.default_rng(np.random.SeedSequence([master, rep]))
        sb = np.random.default_rng(np.random.SeedSequence([master + 1, rep]))
        p = simulate(Tp, N, sa)
        bl = build_blocks(p.Z)
        tgs, ctx = standard_targets(bl, Tp, N)
        r = estimate(p.Y, p.Z, tgs, tuning, rng=sb, oracle=oracle,
                     true_U=p.U, true_V=p.V)
        a_hat = r.estimates["lag_fmean"]; se_a = r.se["lag_fmean"]
        a_true = float(p.surfaces[0][ctx["t0"]] @ ctx["wf"])
        for h in horizons:
            val, g = irf_p1(a_hat, h); se = abs(g) * se_a
            tval = irf_p1(a_true, h)[0]
            acc[f"irf{h}"].append(int(val - Z95 * se <= tval <= val + Z95 * se))
        m, g = lrm_p1(a_hat); se = abs(g) * se_a; tval = lrm_p1(a_true)[0]
        acc["lrm"].append(int(m - Z95 * se <= tval <= m + Z95 * se))
    return {k: dict(R=R, cov=float(np.mean(v))) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
#  thm:xs_dependence -- xs s.e. restores coverage under within-period dependence
# --------------------------------------------------------------------------- #
def xs_coverage(Tp, N, R, tuning: Tuning, master=2024):
    """Under the 'xs' DGP, compare White vs xs coverage for the full-mean targets."""
    names = ["lag_fmean", "slope_fmean"]
    cw = {n: [] for n in names}; cx = {n: [] for n in names}
    for rep in range(R):
        sa = np.random.default_rng(np.random.SeedSequence([master, rep]))
        sb = np.random.default_rng(np.random.SeedSequence([master + 1, rep]))
        p = simulate(Tp, N, sa, noise="xs")
        bl = build_blocks(p.Z)
        tgs, ctx = standard_targets(bl, Tp, N)
        r = estimate(p.Y, p.Z, tgs, tuning, rng=sb)
        for n in names:
            tg = [t for t in tgs if t.name == n][0]; tv = true_value(p, tg, ctx)
            lo, hi = r.ci[n]; cw[n].append(int(lo <= tv <= hi))
            lox, hix = r.ci_xs[n]; cx[n].append(int(lox <= tv <= hix))
    return {n: dict(R=R, white_cov=float(np.mean(cw[n])),
                    xs_cov=float(np.mean(cx[n]))) for n in names}


# --------------------------------------------------------------------------- #
#  lem:local_collinearity_singularity -- contiguous folds => singular info map
# --------------------------------------------------------------------------- #
def _tangent_basis_block(U, V, Tp, N):
    """Orthonormal basis surfaces of the rank-r tangent space at U Sigma V'."""
    r = U.shape[1]
    if r == 0:
        return []
    Qperp = np.eye(Tp) - U @ U.T
    # column space complement of U (Tp - r vectors)
    Wc, _ = np.linalg.qr(Qperp)
    Wc = Wc[:, :max(Tp - r, 0)]
    basis = []
    eN = np.eye(N)
    for a in range(r):                       # U[:,a] x e_n'  (r*N vectors)
        for n in range(N):
            basis.append(np.outer(U[:, a], eN[n]))
    for c in range(Wc.shape[1]):             # u_perp_c x V[:,b]'  ((Tp-r)*r)
        for b in range(r):
            basis.append(np.outer(Wc[:, c], V[:, b]))
    return basis


def contiguous_fold_singular(Tp, N, tuning: Tuning, master=2024):
    """Smallest eigenvalue of the local information map on the tangent space for
    scattered vs contiguous folds.  Contiguous time-block folds leave a held-out
    block's dates with no training support (min per-row support 0), so the map is
    (near) singular -- exactly lem:local_collinearity_singularity.  We use a small
    panel so the map can be assembled densely on an explicit tangent basis."""
    from .design import A, A_adjoint
    from .targets import project_tangent
    sa = np.random.default_rng(np.random.SeedSequence([master]))
    p = simulate(Tp, N, sa)
    bl = build_blocks(p.Z)
    fit = fit_factor_ridge(p.Y, bl, (1, 1, 1), n_restarts=tuning.n_restarts,
                           n_sweeps=tuning.n_sweeps)
    # explicit tangent basis (tuple-of-surfaces) for the three blocks
    blocks_basis = []
    for b, (Ub, Vb) in enumerate(zip(fit.U, fit.V)):
        for surf in _tangent_basis_block(Ub, Vb, Tp, N):
            vec = [np.zeros((Tp, N)) for _ in bl]; vec[b] = surf
            blocks_basis.append(vec)
    out = {}
    for scheme in ("scatter", "contiguous"):
        foldid = assign_folds(Tp, N, tuning.J, rng=np.random.default_rng(0),
                              scheme=scheme)
        folds = make_folds(Tp, N, tuning.J, tuning.q, foldid=foldid)
        fd = folds[0]
        amask = fd.alpha * fd.train.astype(float)
        # assemble G_hat in the tangent basis: G x = alpha P_T A* Pi A P_T x
        d = len(blocks_basis)
        G = np.zeros((d, d))
        images = []
        for xb in blocks_basis:
            Px = project_tangent(xb, fit.U, fit.V)
            R = amask * A(Px, bl)
            adj = project_tangent(A_adjoint(R, bl), fit.U, fit.V)
            images.append(adj)
        for i in range(d):
            for j in range(i, d):
                v = sum(float(np.vdot(images[i][k], blocks_basis[j][k]))
                        for k in range(len(bl)))
                G[i, j] = G[j, i] = v
        eig = np.linalg.eigvalsh((G + G.T) / 2)
        row_support = fd.train.sum(axis=1)
        out[scheme] = dict(min_eig=float(eig[0]), max_eig=float(eig[-1]),
                           cond=float(eig[-1] / max(eig[0], 1e-12)),
                           min_row_support=int(row_support.min()))
    return out
