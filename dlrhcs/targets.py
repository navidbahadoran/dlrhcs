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


def riesz_weights(direction, blocks, U_list, V_list, train_mask, alpha,
                  ridge=1e-8, tol=1e-10, maxiter=2000):
    """Solve the feasible Riesz equation on the tangent space, matrix-free.

    ``U_list/V_list`` are the (estimated or, in the oracle, true) singular
    spaces defining the tangent space ``T``.  ``train_mask`` is ``Pi^pur_{-j}``
    and ``alpha`` is ``alpha_j``.
    """
    Tp, N = blocks[0].shape
    amask = alpha * train_mask.astype(float)
    rhs_theta = project_tangent(direction, U_list, V_list)
    rhs = theta_flatten(rhs_theta)
    scale = max(float(np.max(np.abs(rhs))), 1e-12)

    def matvec(vec):
        x = theta_unflatten(vec, blocks)
        Px = project_tangent(x, U_list, V_list)
        AX = A(Px, blocks)                       # Tp x N
        R = amask * AX                            # alpha * Pi^pur A P_T x
        adj = A_adjoint(R, blocks)
        out = project_tangent(adj, U_list, V_list)
        if ridge:
            out = [o + ridge * scale * p for o, p in zip(out, Px)]
        return theta_flatten(out)

    n = rhs.size
    G = LinearOperator((n, n), matvec=matvec, dtype=float)
    counter = {"k": 0}

    def cb(_):
        counter["k"] += 1

    q_vec, info = cg(G, rhs, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
    q = theta_unflatten(q_vec, blocks)
    q = project_tangent(q, U_list, V_list)       # clean numerical drift
    Psi = A(q, blocks)
    # diagnostic: Rayleigh quotient q' G q / q'q (should be > 0, ~ smallest eig)
    Gq = theta_unflatten(matvec(q_vec), blocks)
    num = theta_dot(q, Gq)
    den = max(theta_dot(q, q), 1e-30)
    return RieszResult(Psi=Psi, q=q, cg_iters=counter["k"],
                       converged=(info == 0), min_eig_proxy=num / den)
