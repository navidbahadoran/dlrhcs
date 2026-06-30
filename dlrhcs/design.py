"""
Design map A and its adjoint A* -- the two primitives the whole package calls.

Model (eq:model, spec sec 1):

    y_{it} = sum_{m=1..M} Z^(m)_{ti} * Gamma^(m)_{ti}  +  H_{ti}  +  u_{it}

We treat the interactive nuisance H as one extra "block" whose design is the
all-ones matrix.  So internally there are B = M + 1 blocks; block ``M`` is H with
``Z[M] = 1``.  A parameter ``Theta`` is a list of ``B`` surfaces, each ``Tp x N``;
``Theta[m]`` is ``Gamma^(m)`` for ``m < M`` and ``Theta[M]`` is ``H``.

``A`` is *linear in Theta given the designs Z*; its adjoint maps a residual
matrix ``R`` (Tp x N) to the tuple ``(Z^(m) (.) R)_m``.  Implement nothing else
in terms of raw outcome SVDs -- the low-rank structure lives in each surface.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

Surface = np.ndarray            # (Tp, N)
Theta = List[np.ndarray]        # length B = M+1, last entry is H


# --------------------------------------------------------------------------- #
#  Block designs
# --------------------------------------------------------------------------- #
def build_blocks(Z_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Append the all-ones H-design to the M coefficient designs.

    Parameters
    ----------
    Z_list : sequence of (Tp, N) arrays
        The M coefficient designs (lagged outcomes and/or regressors).

    Returns
    -------
    blocks : list of B = M+1 arrays; ``blocks[M]`` is ``ones((Tp, N))``.
    """
    Z_list = [np.asarray(z, dtype=float) for z in Z_list]
    Tp, N = Z_list[0].shape
    for z in Z_list:
        if z.shape != (Tp, N):
            raise ValueError("all designs must share shape (Tp, N)")
    return list(Z_list) + [np.ones((Tp, N))]


# --------------------------------------------------------------------------- #
#  A and A*
# --------------------------------------------------------------------------- #
def A(theta: Theta, blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Design map: fitted outcome  sum_b Z[b] (.) Theta[b]  (Tp x N)."""
    out = np.zeros_like(blocks[0])
    for zb, gb in zip(blocks, theta):
        out += zb * gb
    return out


def A_adjoint(R: np.ndarray, blocks: Sequence[np.ndarray]) -> Theta:
    """Adjoint of A: residual matrix R (Tp x N) -> tuple (Z[b] (.) R)_b."""
    return [zb * R for zb in blocks]


# --------------------------------------------------------------------------- #
#  Tiny helpers for the tuple-of-surfaces vector space
# --------------------------------------------------------------------------- #
def zeros_like_theta(blocks: Sequence[np.ndarray]) -> Theta:
    return [np.zeros_like(zb) for zb in blocks]


def theta_dot(x: Theta, y: Theta) -> float:
    """Frobenius inner product over the tuple of surfaces."""
    return float(sum(np.vdot(xb, yb).real for xb, yb in zip(x, y)))


def theta_flatten(x: Theta) -> np.ndarray:
    return np.concatenate([xb.ravel() for xb in x])


def theta_unflatten(vec: np.ndarray, blocks: Sequence[np.ndarray]) -> Theta:
    out, off = [], 0
    for zb in blocks:
        n = zb.size
        out.append(vec[off:off + n].reshape(zb.shape))
        off += n
    return out
