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


def make_folds(Tp: int, N: int, J: int, q: int, r: int = 0, P: int = 1,
               rng: Optional[np.random.Generator] = None,
               scheme: str = "scatter",
               foldid: Optional[np.ndarray] = None) -> List[Fold]:
    """Build the ``J`` purged folds with the space--time exclusion buffer.

    The buffered training set (eq:purged_training / ass:purged_folds) removes a
    cell ``(s, l)`` whenever there is a held-out cell ``(t, i)`` with

        0 <= s - t <= q_TN     (temporal: future dynamic descendants)
        d_N(l, i) <= r_TN      (spatial: contemporaneous neighbours)

    ``q`` is the forward temporal window, ``r`` the spatial radius.  The unit
    metric is the index distance ``|l - i|`` (the 1D ordering used by the
    spatially-mixing DGP); ``r = 0`` recovers the time-only, same-unit purge used
    for the i.i.d. baseline and the empirical panels (which carry no metric).
    ``foldid`` may be supplied to reuse a fixed assignment across runs.
    """
    if foldid is None:
        foldid = assign_folds(Tp, N, J, rng=rng, scheme=scheme)
    folds: List[Fold] = []
    TpN = Tp * N
    for j in range(J):
        val = (foldid == j)
        purged = np.zeros((Tp, N), dtype=bool)
        for h in range(q + 1):                 # temporal shift 0 <= s - t <= q
            for dr in range(-r, r + 1):        # spatial shift |l - i| <= r
                if h == 0 and dr == 0:
                    continue                   # the held-out cell itself
                ts, te = 0, Tp - h             # source / target row windows
                if dr >= 0:
                    us, ue, ut = 0, N - dr, dr
                else:
                    us, ue, ut = -dr, N, 0
                purged[h:Tp, ut:ut + (ue - us)] |= val[ts:te, us:ue]
        train = (~val) & (~purged)
        n_pur = int(train.sum())
        p = float(val.sum()) / TpN
        alpha = TpN / max(n_pur, 1)
        folds.append(Fold(val=val, train=train, p=p, n_pur=n_pur, alpha=alpha))
    return folds


def retained_share(J: int, q: int, r: int = 0) -> float:
    """Approx training share after the space-time buffer: (1 - 1/J)^{(q+1)(2r+1)}.

    Each held-out cell purges a (q+1) x (2r+1) space-time block, so the deleted
    volume per fold scales with (q+1)(2r+1); keep this above ~0.35 (ass:purged_folds
    retention floor)."""
    return (1.0 - 1.0 / J) ** ((q + 1) * (2 * r + 1))
