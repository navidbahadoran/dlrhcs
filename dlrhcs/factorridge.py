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
import hashlib
import json
from typing import Dict, List, Optional, Sequence

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
    warm_start_objective: float = field(default=float("nan"))
    random_restart_objs: List[float] = field(default_factory=list)
    best_objective: float = field(default=float("nan"))
    stopped_before_sweep_cap: bool = field(default=True)
    converged: bool = field(default=True)
    max_iteration_hit: bool = field(default=False)
    relative_restart_improvement: float = field(default=0.0)
    restart_improvement_gt_1e_6: bool = field(default=False)
    restart_improvement_gt_1e_4: bool = field(default=False)
    final_relative_objective_decrease: float = field(default=0.0)
    stationarity_residual: float = field(default=0.0)
    obj_rel_improve: float = field(default=0.0)
    restart_diagnostics: List[Dict[str, object]] = field(default_factory=list)
    selected_restart_label: str = field(default="")
    selected_restart_index: int = field(default=0)
    total_sweeps_from_initialization: int = field(default=0)
    convergence_sweep: Optional[int] = field(default=None)
    stopping_reason: str = field(default="")
    final_level_sweeps: int = field(default=0)


_DEFAULT_TRACE_CHECKPOINTS = (0, 1, 10, 25, 50, 75, 100, 125, 150, 200, 300, 400, 600, 800)


def stable_restart_seed(global_seed, fold_id, restart_type, restart_index, model_id="factor_ridge"):
    """Stable restart seed independent of sweep cap, output path, and execution order."""
    payload = {
        "global_seed": int(global_seed),
        "fold_id": int(fold_id),
        "model_id": str(model_id),
        "restart_index": int(restart_index),
        "restart_type": str(restart_type),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=False)


def _canonical_array_bytes(arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    return str(a.shape).encode("ascii") + b"\0" + a.tobytes(order="C")


def initialization_hash(F, Lam):
    """SHA-256 hash of numerical initialization arrays in canonical dtype/order."""
    h = hashlib.sha256()
    for label, mats in (("F", F), ("Lam", Lam)):
        h.update(label.encode("ascii"))
        h.update(str(len(mats)).encode("ascii"))
        for mat in mats:
            h.update(_canonical_array_bytes(mat))
    return h.hexdigest()


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


def _surfaces_from_factors(F, Lam):
    return [Fb @ Lb.T for Fb, Lb in zip(F, Lam)]


def _als_loop(Y, blocks, ranks, mask, ridge, n_sweeps, tol, F0, Lam0,
              *, sweep_offset=0, trace_checkpoints: Optional[Sequence[int]] = None):
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
    trace = {}
    checkpoint_set = set(int(x) for x in trace_checkpoints) if trace_checkpoints else set()
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
        absolute_sweep = int(sweep_offset + sweeps_done)
        if absolute_sweep in checkpoint_set:
            trace[absolute_sweep] = float(cur)
        if sweep > 0 and (prev - cur) <= tol * max(1.0, abs(prev)):
            break
        prev = cur
    Fb = [Fmat[:, sl].copy() for sl in slices]
    Lb = [Lmat[:, sl].copy() for sl in slices]
    return surfaces_from(Fmat, Lmat), Fb, Lb, np.asarray(obj_path), sweeps_done, trace


def _ridge_schedule(ridge, n_anneal):
    """Geometric schedule from a large ridge down to the target (graduated opt)."""
    if n_anneal <= 1:
        return [ridge]
    return list(np.geomspace(max(1.0, ridge * 50.0), ridge, n_anneal))


def _annealed_als(Y, blocks, ranks, mask, ridge, n_sweeps, tol, F0, Lam0, n_anneal,
                  *, trace_checkpoints: Optional[Sequence[int]] = None):
    """Run ALS over a decreasing ridge schedule; the final level uses ``ridge``."""
    schedule = _ridge_schedule(ridge, n_anneal)
    F, Lam = F0, Lam0
    final = None
    total_sweeps = 0
    trace = {}
    for level, rg in enumerate(schedule):
        nsw = n_sweeps if level == len(schedule) - 1 else max(15, n_sweeps // 3)
        surfaces, F, Lam, path, ns, level_trace = _als_loop(
            Y, blocks, ranks, mask, rg, nsw, tol, F, Lam,
            sweep_offset=total_sweeps, trace_checkpoints=trace_checkpoints)
        total_sweeps += int(ns)
        trace.update(level_trace)
        final = (surfaces, F, Lam, path, ns, total_sweeps, trace)
    return final


def fit_factor_ridge(Y, blocks, ranks, mask=None, ridge=0.02, n_sweeps=80,
                     tol=1e-8, n_restarts=4, rng=None, warm=True, perturb=0.1,
                     n_anneal=8, global_seed=None, fold_id=None,
                     model_id="factor_ridge", trace_checkpoints: Optional[Sequence[int]] = None):
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
    warm_obj = _objective(Y, blocks, _surfaces_from_factors(F0, Lam0), mask,
                          ridge, F0, Lam0)
    trace_checkpoints = tuple(_DEFAULT_TRACE_CHECKPOINTS if trace_checkpoints is None else trace_checkpoints)
    best = None
    restart_objs = []
    restart_diags = []
    for restart in range(max(1, n_restarts)):
        restart_type = "warm_start" if restart == 0 else "random_restart"
        restart_label = "warm_start" if restart == 0 else f"random_{restart}"
        init_seed = None
        if restart == 0:
            Fi, Li = [f.copy() for f in F0], [l.copy() for l in Lam0]
            if global_seed is not None and fold_id is not None:
                init_seed = stable_restart_seed(global_seed, fold_id, restart_type, restart, model_id)
        else:
            if global_seed is not None and fold_id is not None:
                init_seed = stable_restart_seed(global_seed, fold_id, restart_type, restart, model_id)
                rng_i = np.random.default_rng(np.random.SeedSequence(init_seed))
            else:
                rng_i = rng
            Fi = [f + perturb * rng_i.standard_normal(f.shape) for f in F0]
            Li = [l + perturb * rng_i.standard_normal(l.shape) for l in Lam0]
        init_hash = initialization_hash(Fi, Li)
        init_obj = _objective(Y, blocks, _surfaces_from_factors(Fi, Li), mask,
                              ridge, Fi, Li)
        trace = {0: float(init_obj)} if 0 in set(trace_checkpoints) else {}
        surfaces, Fb, Lb, path, ns, total_ns, anneal_trace = _annealed_als(
            Y, blocks, ranks, mask, ridge, n_sweeps, tol, Fi, Li, n_anneal,
            trace_checkpoints=trace_checkpoints)
        trace.update(anneal_trace)
        obj = float(path[-1]) if len(path) else np.inf
        restart_objs.append(obj)
        monotone = bool(np.all(np.diff(path) <= 1e-9 * (1 + np.abs(path[:-1]))))
        max_hit = bool(ns >= n_sweeps)
        stopped_before_cap = bool(not max_hit)
        convergence_sweep = int(total_ns) if stopped_before_cap else None
        stopping_reason = "relative_objective_tolerance" if stopped_before_cap else "sweep_cap"
        diag = dict(
            initialization_seed=int(init_seed) if init_seed is not None else None,
            initialization_hash=init_hash,
            restart_label=restart_label,
            restart_index=int(restart),
            restart_type=restart_type,
            initial_objective=float(init_obj),
            final_objective=float(obj),
            total_sweeps_from_initialization=int(total_ns),
            final_level_sweeps=int(ns),
            convergence_sweep=convergence_sweep,
            stopping_reason=stopping_reason,
            converged=stopped_before_cap,
            stopped_before_sweep_cap=stopped_before_cap,
            max_iteration_hit=max_hit,
            objective_trace_checkpoints={str(int(k)): float(v) for k, v in sorted(trace.items())},
            n_anneal=int(n_anneal),
        )
        restart_diags.append(diag)
        if best is None or obj < best[0]:
            best = (obj, surfaces, Fb, Lb, path, ns, monotone, restart, total_ns, diag)
    obj, surfaces, Fb, Lb, path, ns, monotone, best_restart, total_ns, best_diag = best
    # Final-sweep relative objective decrease: a numerical-stability proxy.
    # Hitting the sweep cap is recorded separately and is not, by itself, a
    # convergence failure.
    rel_improve = (float(abs(path[-2] - path[-1]) / (1.0 + abs(path[-1])))
                   if len(path) > 1 else 0.0)
    baseline = restart_objs[0] if restart_objs else obj
    restart_improve = float(max(0.0, (baseline - obj) / (1.0 + abs(baseline))))
    max_hit = bool(ns >= n_sweeps)
    stopped_before_cap = bool(not max_hit)
    U, V, svals = [], [], []
    for surf, r in zip(surfaces, ranks):
        Ub, sb, Vb = block_svd(surf, r)
        U.append(Ub); V.append(Vb); svals.append(sb)
    return FitResult(surfaces=surfaces, F=Fb, Lam=Lb, U=U, V=V, svals=svals,
                     obj_path=path, objective=obj, n_sweeps=ns, monotone=monotone,
                     restart_objs=restart_objs,
                     warm_start_objective=float(warm_obj),
                     random_restart_objs=restart_objs[1:],
                     best_objective=float(obj),
                     stopped_before_sweep_cap=stopped_before_cap,
                     converged=stopped_before_cap,
                     max_iteration_hit=max_hit,
                     relative_restart_improvement=restart_improve,
                     restart_improvement_gt_1e_6=bool(restart_improve > 1e-6),
                     restart_improvement_gt_1e_4=bool(restart_improve > 1e-4),
                     final_relative_objective_decrease=rel_improve,
                     stationarity_residual=rel_improve,
                     obj_rel_improve=rel_improve,
                     restart_diagnostics=restart_diags,
                     selected_restart_label=str(best_diag.get("restart_label", "")),
                     selected_restart_index=int(best_restart),
                     total_sweeps_from_initialization=int(total_ns),
                     convergence_sweep=best_diag.get("convergence_sweep"),
                     stopping_reason=str(best_diag.get("stopping_reason", "")),
                     final_level_sweeps=int(ns))
