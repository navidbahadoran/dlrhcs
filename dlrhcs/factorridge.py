"""Alternating factor-ridge ALS (spec sec 4) + truncated-SVD warm start
(spec sec 5) + ridge annealing (graduated optimization) to escape the weakly
identified lag-block stationary points.

We fit, on a training mask S (a purged fold), at ranks r = (r_1..r_M, r_H):

    Q = 0.5 * sum_{(t,i) in S}[ y_it - sum_b Z[b]_ti fv_{t,b}'lam_{i,b} ]^2
        + 0.5 * rho * sum_b ( ||F_b||^2 + ||Lam_b||^2 ),   Gamma^(b)=F_b Lam_b'.

Row/column updates are batched closed-form ridge solves; the objective is
monotone non-increasing at fixed ridge.  Warm start: per-cell min-norm ridge
(never an SVD of Y) then per-block truncated SVD.  Ridge annealing starts at a
large ridge (smooth landscape -> global basin) and anneals to the target ridge.

This is exactly the code validated by tests/test_core.py and the oracle MC
checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .design import A


def _canonical_sign(U, V):
    """Fix SVD sign ambiguity: largest-|.| entry of each U column made positive."""
    if U.shape[1] == 0:
        return U, V
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs, V * signs


def block_svd(surface, r):
    """Rank-r singular spaces of a surface, sign-canonicalized."""
    if r == 0:
        Tp, N = surface.shape
        return np.zeros((Tp, 0)), np.zeros(0), np.zeros((N, 0))
    U, s, Vt = np.linalg.svd(surface, full_matrices=False)
    U, V = _canonical_sign(U[:, :r], Vt[:r].T)
    return U, s[:r], V


@dataclass
class FitResult:
    surfaces: List[np.ndarray]
    F: List[np.ndarray]
    Lam: List[np.ndarray]
    U: List[np.ndarray]
    V: List[np.ndarray]
    svals: List[np.ndarray]
    obj_path: np.ndarray
    objective: float
    n_sweeps: int
    monotone: bool = field(default=True)
    restart_objs: List[float] = field(default_factory=list)
    obj_rel_improve: float = field(default=0.0)


def _softimpute_block(obs, mask, r, iters=4):
    """Rank-r soft-impute on observed (training) cells only -> seed factors."""
    G = np.where(mask, obs, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        low = (U[:, :r] * s[:r]) @ Vt[:r]
        G = np.where(mask, obs, low)
    return G


def warm_start(Y, blocks, ranks, mask, tau=1e-3, soft_iters=4):
    """Per-cell linear surface recovery + per-block truncated SVD (spec sec 5)."""
    S = np.zeros_like(Y)
    for zb in blocks:
        S += zb * zb
    denom = S + tau
    F, Lam = [], []
    for zb, r in zip(blocks, ranks):
        gamma_lin = (zb * Y) / denom
        G = _softimpute_block(gamma_lin, mask, r, iters=soft_iters)
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        U, V = _canonical_sign(U[:, :r], Vt[:r].T)
        sr = np.sqrt(np.maximum(s[:r], 0.0))
        F.append(U * sr)
        Lam.append(V * sr)
    return F, Lam


def _col_slices(ranks):
    out, off = [], 0
    for r in ranks:
        out.append(slice(off, off + r))
        off += r
    return out, off


def _scaleZ(blocks, ranks, Rtot):
    """Per-cell design scaling tensor (Tp, N, Rtot): column k uses Z[block(k)]."""
    Tp, N = blocks[0].shape
    sc = np.empty((Tp, N, Rtot))
    off = 0
    for zb, r in zip(blocks, ranks):
        sc[:, :, off:off + r] = zb[:, :, None]
        off += r
    return sc


def _objective(Y, blocks, surfaces, mask, ridge, F, Lam):
    R = (Y - A(surfaces, blocks)) * mask
    val = 0.5 * float(np.sum(R * R))
    for Fb, Lb in zip(F, Lam):
        val += 0.5 * ridge * (float(np.sum(Fb * Fb)) + float(np.sum(Lb * Lb)))
    return val


def _als_loop(Y, blocks, ranks, mask, ridge, n_sweeps, tol, F0, Lam0):
    Tp, N = Y.shape
    slices, Rtot = _col_slices(ranks)
    sc = _scaleZ(blocks, ranks, Rtot)
    Ridge = ridge * np.eye(Rtot)
    Fmat = np.concatenate(F0, axis=1).copy()
    Lmat = np.concatenate(Lam0, axis=1).copy()
    Ym = Y * mask
    m3 = mask[:, :, None]

    def surfaces_from(Fm, Lm):
        return [Fm[:, sl] @ Lm[:, sl].T for sl in slices]

    obj_path = []
    prev = np.inf
    sweeps_done = 0
    for sweep in range(n_sweeps):
        D = sc * Lmat[None, :, :]
        Dm = D * m3
        At = np.einsum('tik,til->tkl', Dm, D, optimize=True) + Ridge
        bt = np.einsum('tik,ti->tk', Dm, Ym, optimize=True)
        Fmat = np.linalg.solve(At, bt[:, :, None])[:, :, 0]
        C = sc * Fmat[:, None, :]
        Cm = C * m3
        Ai = np.einsum('tik,til->ikl', Cm, C, optimize=True) + Ridge
        bi = np.einsum('tik,ti->ik', Cm, Ym, optimize=True)
        Lmat = np.linalg.solve(Ai, bi[:, :, None])[:, :, 0]

        surfaces = surfaces_from(Fmat, Lmat)
        Fb = [Fmat[:, sl] for sl in slices]
        Lb = [Lmat[:, sl] for sl in slices]
        cur = _objective(Y, blocks, surfaces, mask, ridge, Fb, Lb)
        obj_path.append(cur)
        sweeps_done = sweep + 1
        if sweep > 0 and (prev - cur) <= tol * max(1.0, abs(prev)):
            break
        prev = cur
    Fb = [Fmat[:, sl].copy() for sl in slices]
    Lb = [Lmat[:, sl].copy() for sl in slices]
    return surfaces_from(Fmat, Lmat), Fb, Lb, np.asarray(obj_path), sweeps_done


def _ridge_schedule(ridge, n_anneal):
    """Geometric schedule from a large ridge down to the target (graduated opt)."""
    if n_anneal <= 1:
        return [ridge]
    return list(np.geomspace(max(1.0, ridge * 50.0), ridge, n_anneal))


def _annealed_als(Y, blocks, ranks, mask, ridge, n_sweeps, tol, F0, Lam0, n_anneal):
    """Run ALS over a decreasing ridge schedule; the final level uses ``ridge``."""
    schedule = _ridge_schedule(ridge, n_anneal)
    F, Lam = F0, Lam0
    final = None
    for level, rg in enumerate(schedule):
        nsw = n_sweeps if level == len(schedule) - 1 else max(15, n_sweeps // 3)
        surfaces, F, Lam, path, ns = _als_loop(
            Y, blocks, ranks, mask, rg, nsw, tol, F, Lam)
        final = (surfaces, F, Lam, path, ns)
    return final


def fit_factor_ridge(Y, blocks, ranks, mask=None, ridge=0.02, n_sweeps=80,
                     tol=1e-8, n_restarts=4, rng=None, warm=True, perturb=0.1,
                     n_anneal=8):
    """Fit the alternating factor-ridge model; keep the lowest-objective restart.

    ranks    : per-block ranks (length B = M+1, last is the H block).
    ridge    : factor-ridge constant rho (default 0.02).
    n_anneal : ridge-annealing levels (graduated optimization); 1 disables it.
    """
    Y = np.asarray(Y, dtype=float)
    Tp, N = Y.shape
    if mask is None:
        mask = np.ones((Tp, N), dtype=bool)
    if rng is None:
        rng = np.random.default_rng(0)
    if warm:
        F0, Lam0 = warm_start(Y, blocks, ranks, mask)
    else:
        F0 = [rng.standard_normal((Tp, r)) * 0.1 for r in ranks]
        Lam0 = [rng.standard_normal((N, r)) * 0.1 for r in ranks]
    best = None
    restart_objs = []
    for restart in range(max(1, n_restarts)):
        if restart == 0:
            Fi, Li = [f.copy() for f in F0], [l.copy() for l in Lam0]
        else:
            Fi = [f + perturb * rng.standard_normal(f.shape) for f in F0]
            Li = [l + perturb * rng.standard_normal(l.shape) for l in Lam0]
        surfaces, Fb, Lb, path, ns = _annealed_als(
            Y, blocks, ranks, mask, ridge, n_sweeps, tol, Fi, Li, n_anneal)
        obj = float(path[-1]) if len(path) else np.inf
        restart_objs.append(obj)
        monotone = bool(np.all(np.diff(path) <= 1e-9 * (1 + np.abs(path[:-1]))))
        if best is None or obj < best[0]:
            best = (obj, surfaces, Fb, Lb, path, ns, monotone)
    obj, surfaces, Fb, Lb, path, ns, monotone = best
    # final-sweep relative objective improvement (how close the iterate is to a fixed point)
    rel_improve = (float(abs(path[-2] - path[-1]) / (1.0 + abs(path[-1])))
                   if len(path) > 1 else 0.0)
    U, V, svals = [], [], []
    for surf, r in zip(surfaces, ranks):
        Ub, sb, Vb = block_svd(surf, r)
        U.append(Ub); V.append(Vb); svals.append(sb)
    return FitResult(surfaces=surfaces, F=Fb, Lam=Lb, U=U, V=V, svals=svals,
                     obj_path=path, objective=obj, n_sweeps=ns, monotone=monotone,
                     restart_objs=restart_objs, obj_rel_improve=rel_improve)
