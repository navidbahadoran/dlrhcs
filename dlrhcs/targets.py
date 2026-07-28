"""
Targets, the local tangent space, and the feasible debiasing (Riesz) weights
(spec sec 8).

A linear target is ``phi_nu(Theta) = <D_nu, Theta>`` for a *direction* ``D_nu``
(a tuple of surfaces, mostly zero).  The plug-in value is just the inner product
with the fitted surfaces.

The tangent space of the rank-``r`` manifold at ``Gamma = U Sigma V'`` is

    T = { U B' + A V' },   P_T(X) = U U'X + X V V' - U U'X V V'.

``T_0 = T_1 x ... x T_M x T_H`` and ``P_{T_0}`` applies blockwise.

Feasible weights (eq:feasible_fold_gram), with the purged-training mask
``Pi^pur_{-j}`` and fold scale ``alpha_j``:

    G_hat = alpha_j * P_T A* Pi^pur A P_T            (local information map)
    q_hat = G_hat^+ P_T D_nu                         (Riesz solve)
    Psi_hat = A(q_hat)                               (Tp x N observation weights)

We solve the Riesz equation **matrix-free** by conjugate gradients: ``G_hat``
acts on a tuple of surfaces through cheap primitives (``A``, ``A*``, the block
projector ``P_T``), so we never materialize the ``O(sum r_b (Tp+N))``-dimensional
tangent basis.  This is both memory-light (essential for the large Zillow panel)
and fast.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .design import (A, A_adjoint, theta_dot, theta_flatten, theta_unflatten,
                     zeros_like_theta)


# --------------------------------------------------------------------------- #
#  Directions D_nu
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    name: str
    block: int                 # which coefficient block the target reads
    direction: List[np.ndarray]
    kind: str = "linear"       # 'linear' | 'irf' | 'lrm' (smooth -> delta method)


def _zero_dirs(blocks):
    return [np.zeros_like(zb) for zb in blocks]


def entry_direction(blocks, block, t, i):
    D = _zero_dirs(blocks)
    D[block][t, i] = 1.0
    return D


def mean_direction(blocks, block, t, weights):
    """theta = e_t' Gamma^(block) pi ; weights is a length-N unit-weight vector."""
    D = _zero_dirs(blocks)
    D[block][t, :] = np.asarray(weights, dtype=float)
    return D


def group_weights(N, members):
    w = np.zeros(N)
    w[members] = 1.0 / max(len(members), 1)
    return w


def make_target(blocks, name, block, kind="entry", t=0, i=0,
                weights=None, weights2=None):
    """Convenience builder for the standard MC targets."""
    if kind == "entry":
        return Target(name, block, entry_direction(blocks, block, t, i))
    if kind == "mean":
        return Target(name, block, mean_direction(blocks, block, t, weights))
    if kind == "contrast":
        d1 = mean_direction(blocks, block, t, weights)
        d2 = mean_direction(blocks, block, t, weights2)
        return Target(name, block, [a - b for a, b in zip(d1, d2)])
    raise ValueError(f"unknown kind {kind}")


# --------------------------------------------------------------------------- #
#  Tangent projector
# --------------------------------------------------------------------------- #
def project_block(X, U, V):
    """P_T(X) for a single block; U (Tp x r), V (N x r)."""
    if U.shape[1] == 0:
        return np.zeros_like(X)
    UtX = U.T @ X                      # r x N
    XV = X @ V                         # Tp x r
    UUtX = U @ UtX                     # Tp x N
    XVVt = XV @ V.T                    # Tp x N
    UUtXVVt = (U @ (UtX @ V)) @ V.T    # Tp x N
    return UUtX + XVVt - UUtXVVt


def project_tangent(theta, U_list, V_list):
    return [project_block(Xb, Ub, Vb)
            for Xb, Ub, Vb in zip(theta, U_list, V_list)]


# --------------------------------------------------------------------------- #
#  Matrix-free feasible Riesz weights
# --------------------------------------------------------------------------- #
@dataclass
class RieszResult:
    Psi: np.ndarray          # (Tp, N) observation-space weights
    q: List[np.ndarray]      # tangent-space representer (tuple of surfaces)
    cg_iters: int
    converged: bool
    min_eig_proxy: float     # Rayleigh quotient of the solution (diagnostic)
    solver_name: str = "scipy.sparse.linalg.cg"
    convergence_info_code: int = 0
    maxiter: int = 0
    requested_tolerance: float = 0.0
    achieved_absolute_residual: float = 0.0
    achieved_relative_residual: float = 0.0
    rhs_norm: float = 0.0
    solution_norm: float = 0.0
    maximum_absolute_solution_entry: float = 0.0
    riesz_ridge: float = 0.0
    scaling_value: float = 1.0
    cached_scale: bool = False
    elapsed_seconds: float = 0.0
    contains_nonfinite: bool = False


def cg_converged_from_status(info, achieved_relative_residual, requested_tolerance,
                             contains_nonfinite=False) -> bool:
    """Interpret SciPy CG status using solver status and achieved residual.

    SciPy's ``cg`` returns ``info == 0`` on successful convergence, ``info > 0``
    when tolerance is not achieved within the iteration limit, and ``info < 0``
    for illegal input or breakdown.  A solve that reaches the final allowed
    callback iteration can still be converged if SciPy reports success and the
    residual criterion is met.
    """
    try:
        rel = float(achieved_relative_residual)
        tol = float(requested_tolerance)
    except Exception:
        return False
    return bool(int(info) == 0 and not contains_nonfinite and
                np.isfinite(rel) and np.isfinite(tol) and rel <= tol)


class RieszFoldSolver:
    """Fold-level matrix-free Riesz solver.

    The purged training mask, tangent spaces, ridge-free normal operator, and
    LinearOperator objects are target-independent within a fold.  This solver
    keeps those objects together so multiple target right-hand sides can reuse
    the fold setup while each target still runs its own CG solve.

    By default ``solve`` preserves the historical RHS-initialized ridge scaling
    exactly.  ``use_cached_scale=True`` enables a fold-level scale cache for
    experiments where a common fold scale is desired.
    """

    def __init__(self, blocks, U_list, V_list, train_mask, alpha):
        self.blocks = blocks
        self.U_list = U_list
        self.V_list = V_list
        self.amask = alpha * train_mask.astype(float)
        self.n = int(sum(np.asarray(zb).size for zb in blocks))
        self._operator_cache = {}
        self._fold_scale = None
        self._active_scale = 1.0

    def project(self, theta):
        return project_tangent(theta, self.U_list, self.V_list)

    def _G0_apply(self, vec):
        Px0 = self.project(theta_unflatten(vec, self.blocks))
        adj0 = A_adjoint(self.amask * A(Px0, self.blocks), self.blocks)
        return theta_flatten(self.project(adj0))

    def _power_scale_from_seed(self, seed):
        v = seed / max(float(np.linalg.norm(seed)), 1e-12)
        for _ in range(4):
            g = self._G0_apply(v)
            nrm = float(np.linalg.norm(g))
            if nrm < 1e-30:
                break
            v = g / nrm
        return max(float(v @ self._G0_apply(v)), 1e-12)

    def fold_scale(self):
        """Return an optional fold-level operator scale.

        This is not used by the default pipeline because the legacy numerical
        ridge scale is initialized from each target RHS.  It is available as a
        controlled switch without changing the old ``riesz_weights`` behavior.
        """
        if self._fold_scale is None:
            self._fold_scale = self._power_scale_from_seed(np.ones(self.n))
        return self._fold_scale

    def linear_operator(self, ridge):
        key = float(ridge)
        if key in self._operator_cache:
            return self._operator_cache[key]

        def matvec(vec):
            x = theta_unflatten(vec, self.blocks)
            Px = self.project(x)
            AX = A(Px, self.blocks)
            R = self.amask * AX
            adj = A_adjoint(R, self.blocks)
            out = self.project(adj)
            if ridge:
                scale = self._active_scale
                out = [o + ridge * scale * p for o, p in zip(out, Px)]
            return theta_flatten(out)

        G = LinearOperator((self.n, self.n), matvec=matvec, dtype=float)
        self._operator_cache[key] = G
        return G

    def solve(self, direction, ridge=1e-8, tol=1e-10, maxiter=2000,
              use_cached_scale=False):
        t0 = time.perf_counter()
        rhs_theta = self.project(direction)
        rhs = theta_flatten(rhs_theta)
        scale = self.fold_scale() if use_cached_scale else self._power_scale_from_seed(rhs)
        self._active_scale = scale
        G = self.linear_operator(ridge)
        counter = {"k": 0}

        def cb(_):
            counter["k"] += 1

        q_vec, info = cg(G, rhs, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
        q = theta_unflatten(q_vec, self.blocks)
        q = self.project(q)
        Psi = A(q, self.blocks)
        Gq = theta_unflatten(G.matvec(q_vec), self.blocks)
        residual_vec = rhs - G.matvec(q_vec)
        abs_resid = float(np.linalg.norm(residual_vec))
        rhs_norm = float(np.linalg.norm(rhs))
        rel_resid = abs_resid / max(rhs_norm, 1e-30)
        sol_norm = float(np.linalg.norm(q_vec))
        max_abs_sol = float(np.max(np.abs(q_vec))) if q_vec.size else 0.0
        contains_nonfinite = not (
            np.all(np.isfinite(q_vec)) and
            np.isfinite(abs_resid) and np.isfinite(rel_resid) and
            np.isfinite(rhs_norm) and np.isfinite(scale)
        )
        num = theta_dot(q, Gq)
        den = max(theta_dot(q, q), 1e-30)
        return RieszResult(Psi=Psi, q=q, cg_iters=counter["k"],
                           converged=cg_converged_from_status(info, rel_resid, tol, contains_nonfinite),
                           min_eig_proxy=num / den,
                           convergence_info_code=int(info),
                           maxiter=int(maxiter),
                           requested_tolerance=float(tol),
                           achieved_absolute_residual=abs_resid,
                           achieved_relative_residual=rel_resid,
                           rhs_norm=rhs_norm,
                           solution_norm=sol_norm,
                           maximum_absolute_solution_entry=max_abs_sol,
                           riesz_ridge=float(ridge),
                           scaling_value=float(scale),
                           cached_scale=bool(use_cached_scale),
                           elapsed_seconds=float(time.perf_counter() - t0),
                           contains_nonfinite=bool(contains_nonfinite))


def riesz_weights(direction, blocks, U_list, V_list, train_mask, alpha,
                  ridge=1e-8, tol=1e-10, maxiter=2000,
                  use_cached_scale=False):
    """Solve the feasible Riesz equation on the tangent space, matrix-free.

    ``U_list/V_list`` are the (estimated or, in the oracle, true) singular
    spaces defining the tangent space ``T``.  ``train_mask`` is ``Pi^pur_{-j}``
    and ``alpha`` is ``alpha_j``.
    """
    solver = RieszFoldSolver(blocks, U_list, V_list, train_mask, alpha)
    return solver.solve(direction, ridge=ridge, tol=tol, maxiter=maxiter,
                        use_cached_scale=use_cached_scale)


# Backward-compatible class alias for code that imported the first cache name.
RieszSolver = RieszFoldSolver
