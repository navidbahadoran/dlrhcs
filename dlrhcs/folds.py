"""
Scattered cross-fitting folds with a forward exclusion window (spec sec 6).

The signature piece of the method.  Two rules:

* **Scattered folds.**  Every effective cell ``(t, i)`` is assigned to a fold
  ``sigma(t, i) in {0..J-1}``, scattered over the time-unit grid (deterministic
  checkerboard or a fixed-seed draw), *not* contiguous time blocks -- a time
  block would leave its own dates with no training support.

* **Forward exclusion window.**  The training set for fold ``j`` removes the
  fold itself and every *same-unit* cell within ``q`` periods **after** a
  fold-``j`` cell:

      I^pur_{-j} = { (t,i) : sigma(t,i) != j  AND
                     sigma(s,i) != j for all max(P+1, t-q) <= s < t }.

  In words: drop a cell from training if it is held out, or if the same unit
  was held out at any of the preceding ``q`` dates.  This deletes exactly the
  future same-unit descendants through which a held-out ``u_{it}`` propagates.

Bookkeeping returned per fold:
  * ``p_j``      = |fold j| / (Tp*N)                  realized held-out share
  * ``n_pur_j``  = |I^pur_{-j}|                       training-cell count
  * ``alpha_j``  = Tp*N / n_pur_j                     rescales training sums to
                                                      full-sample (per-cell) scale
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Fold:
    val: np.ndarray      # (Tp, N) bool: held-out (fold-j) cells   = Pi_j
    train: np.ndarray    # (Tp, N) bool: purged training cells     = I^pur_{-j}
    p: float             # |val| / (Tp*N)
    n_pur: int           # |train|
    alpha: float         # Tp*N / n_pur


def assign_folds(Tp: int, N: int, J: int,
                 rng: Optional[np.random.Generator] = None,
                 scheme: str = "scatter") -> np.ndarray:
    """Return an integer ``(Tp, N)`` fold-id grid, scattered and balanced.

    ``scheme='checker'`` is a fully deterministic interleaving;
    ``scheme='scatter'`` is a fixed-seed balanced random permutation
    (default -- balanced to within one cell per fold).
    """
    if scheme == "checker":
        idx = (np.arange(Tp)[:, None] + np.arange(N)[None, :]) % J
        return idx.astype(int)
    if scheme == "contiguous":
        # contiguous time blocks (the negative control of
        # lem:local_collinearity_singularity): each fold is a block of
        # consecutive dates, so a held-out date has no own training support.
        block = np.minimum((np.arange(Tp) * J) // Tp, J - 1)
        return np.repeat(block[:, None], N, axis=1).astype(int)
    if rng is None:
        rng = np.random.default_rng(0)
    n = Tp * N
    base = np.repeat(np.arange(J), int(np.ceil(n / J)))[:n]
    rng.shuffle(base)
    return base.reshape(Tp, N)


def make_folds(Tp: int, N: int, J: int, q: int, P: int = 1,
               rng: Optional[np.random.Generator] = None,
               scheme: str = "scatter",
               foldid: Optional[np.ndarray] = None) -> List[Fold]:
    """Build the ``J`` purged folds with forward exclusion window ``q``.

    ``foldid`` may be supplied to reuse a fixed assignment across runs.
    """
    if foldid is None:
        foldid = assign_folds(Tp, N, J, rng=rng, scheme=scheme)
    folds: List[Fold] = []
    TpN = Tp * N
    for j in range(J):
        val = (foldid == j)
        purged = np.zeros((Tp, N), dtype=bool)
        # purge same-unit cells in the q dates *after* a held-out cell:
        # cell (t,i) is purged if (t-h, i) is held out for some 1 <= h <= q.
        for h in range(1, q + 1):
            purged[h:, :] |= val[:-h, :]
        train = (~val) & (~purged)
        n_pur = int(train.sum())
        p = float(val.sum()) / TpN
        alpha = TpN / max(n_pur, 1)
        folds.append(Fold(val=val, train=train, p=p, n_pur=n_pur, alpha=alpha))
    return folds


def retained_share(J: int, q: int) -> float:
    """Approximate training share (1 - 1/J)^{q+1}; keep above ~0.35."""
    return (1.0 - 1.0 / J) ** (q + 1)
