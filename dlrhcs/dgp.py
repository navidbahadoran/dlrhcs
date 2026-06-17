"""
Monte Carlo data-generating process (spec sec 3, eq:sim_dgp, tab:sim_design).

Baseline P = K = 1 model

    y_{it} = a_{0,ti} y_{i,t-1} + x_{it} beta_{0,ti} + h_{0,ti} + u_{it},
    u_{it} ~ iid N(0, sigma_u^2).

Conditional on the exogenous frame G_0 (the rank-one surfaces, the regressor
paths, the burn-in initial conditions, and the fold assignment) the innovations
are mutually independent, mean zero, with deterministic conditional variances --
exactly the predictable-weight structure the CLT needs.  Three innovation laws
are provided, all conditional-mean-zero:

  * ``'iid'``    : u = sigma_u * N(0,1)                    (the paper's baseline)
  * ``'hetero'`` : u = sigma(G_0)_{it} * N(0,1)            (deterministic sigma^2)
  * ``'xs'``     : within-period cross-sectionally correlated, independent across
                   time slices (for the cross-sectional variance study).

Construction matches tab:sim_design: 50 burn-in periods, exact rank-1 surfaces
with smooth time factors and incoherent unit loadings, singular values of order
sqrt(Tp*N), the lag loading capped at 0.92*rho_y for stability, V_B orthogonal to
V_H, and a common-factor-plus-residual regressor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  building blocks
# --------------------------------------------------------------------------- #
def _orthonormal(M):
    Q, _ = np.linalg.qr(M)
    return Q


def smooth_time_factor(Tp, r, rng, positive=False, phase=0.0):
    t = np.arange(1, Tp + 1) / Tp
    cols = []
    for p in range(r):
        base = np.sin((p + 1) * np.pi * t + phase + 0.3 * p)
        cols.append(0.6 + 0.4 * np.abs(base) if positive else 0.7 + 0.3 * base)
    return np.column_stack(cols)


def incoherent_loadings(N, r, rng, positive=False):
    """Incoherent unit loadings: orthonormal columns (bounded coherence w.h.p.)."""
    if positive:
        V = 0.4 + np.abs(rng.standard_normal((N, r)))
        return V / np.linalg.norm(V, axis=0, keepdims=True)
    return _orthonormal(rng.standard_normal((N, r)))


def coherence(V):
    """max_i ||e_i' V||^2 * N / r  -- O(1) when incoherent."""
    N, r = V.shape
    row_norms = np.sum(V ** 2, axis=1)
    return float(np.max(row_norms) * N / max(r, 1))


def make_surface(Tp, N, r, rng, rms, positive=False, V=None, phase=0.0):
    """Exact rank-r surface with entrywise rms ``rms`` (=> sigma_1 ~ rms*sqrt(TpN))."""
    F = smooth_time_factor(Tp, r, rng, positive=positive, phase=phase)
    if V is None:
        V = incoherent_loadings(N, r, rng, positive=positive)
    sv = 1.0 - 0.2 * np.arange(r) / max(r, 1)
    M = (F * sv) @ V.T
    M *= rms / np.sqrt(np.mean(M ** 2))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return M, U[:, :r], s[:r], Vt[:r].T


@dataclass
class Panel:
    Y: np.ndarray
    Z: List[np.ndarray]          # [Ylag, X]  (designs Z^(1), Z^(2))
    surfaces: List[np.ndarray]   # [A0, B0, H0]
    U: List[np.ndarray]          # true left singular spaces
    V: List[np.ndarray]          # true right singular spaces
    U_innov: np.ndarray          # realized innovations (Tp x N)
    Tp: int
    N: int
    P: int = 1
    meta: Dict = field(default_factory=dict)


def _innovations(Tp, N, sigma_u, rng, noise, G0):
    if noise == "iid":
        return sigma_u * rng.standard_normal((Tp, N))
    if noise == "hetero":
        # deterministic conditional sd from the (exogenous) nuisance surface
        Hn = np.abs(G0["H0"])
        s = 0.6 + 0.8 * Hn / (Hn.mean() + 1e-12)
        s *= sigma_u / s.mean()
        return s * rng.standard_normal((Tp, N))
    if noise == "xs":
        # Cross-sectionally DECAYING within-period dependence: a spatial AR(1)
        # along the unit index, corr(u_{it}, u_{jt}) = theta^{|i-j|}, independent
        # across time.  Covariance row-sums are O(1) (sum_k theta^{|k|} =
        # (1+theta)/(1-theta)), so the cumulant-summability condition ass:dependent
        # (b) holds -- this is the dependence structure thm:xs_dependence covers
        # (NOT a pervasive common factor, whose row-sums grow like N).
        theta = 0.6
        e = rng.standard_normal((Tp, N))
        out = np.empty((Tp, N))
        out[:, 0] = e[:, 0]
        s = np.sqrt(1.0 - theta ** 2)
        for i in range(1, N):
            out[:, i] = theta * out[:, i - 1] + s * e[:, i]
        return sigma_u * out
    raise ValueError(f"unknown noise model {noise}")


# --------------------------------------------------------------------------- #
#  baseline P = 1 simulator
# --------------------------------------------------------------------------- #
def simulate(Tp, N, rng, *, r=1, rho_y=0.85, sigma_u=0.30, c_x=0.30,
             sigma_x=1.0, burn=50, a_rms=0.55, bh_rms=0.50, noise="iid"):
    """Simulate one P=1 panel on the effective sample (rows t = P+1..T)."""
    P = 1
    # ---- mutually structured loading spaces: V_B orthogonal to V_H ---------
    G = _orthonormal(rng.standard_normal((N, 3 * r)))
    V_H, V_B = G[:, :r], G[:, r:2 * r]

    A0, UA, sA, VA = make_surface(Tp, N, r, rng, a_rms, positive=True, phase=0.0)
    cap = 0.92 * rho_y
    if np.max(np.abs(A0)) > cap:
        A0 *= cap / np.max(np.abs(A0))
        UA, sA, VAt = np.linalg.svd(A0, full_matrices=False)
        UA, VA = UA[:, :r], VAt[:r].T
    B0, UB, sB, VB = make_surface(Tp, N, r, rng, bh_rms, V=V_B, phase=0.7)
    H0, UH, sH, VH = make_surface(Tp, N, r, rng, bh_rms, V=V_H, phase=1.4)

    # ---- regressor: common factor + residual identifying variation --------
    fx = rng.standard_normal(Tp + burn + 1)
    lx = rng.standard_normal(N)
    Xfull = c_x * fx[:, None] * lx[None, :] + sigma_x * rng.standard_normal(
        (Tp + burn + 1, N))
    Xfull = (Xfull - Xfull.mean()) / Xfull.std()

    G0 = {"A0": A0, "B0": B0, "H0": H0}
    # ---- recursion with burn-in (coefficients reuse row 0 during burn-in) --
    y_prev = 0.1 * rng.standard_normal(N)
    for s in range(burn):
        u = sigma_u * rng.standard_normal(N)
        y_prev = A0[0] * y_prev + Xfull[s] * B0[0] + H0[0] + u
    u = sigma_u * rng.standard_normal(N)
    y_init = A0[0] * y_prev + Xfull[burn] * B0[0] + H0[0] + u   # y at t = P

    Xeff = Xfull[burn + 1: burn + 1 + Tp]
    Uinnov = _innovations(Tp, N, sigma_u, rng, noise, G0)
    Ylag = np.empty((Tp, N))
    Y = np.empty((Tp, N))
    y_lag = y_init
    for k in range(Tp):
        y_cur = A0[k] * y_lag + Xeff[k] * B0[k] + H0[k] + Uinnov[k]
        Ylag[k] = y_lag
        Y[k] = y_cur
        y_lag = y_cur

    return Panel(Y=Y, Z=[Ylag, Xeff], surfaces=[A0, B0, H0],
                 U=[UA, UB, UH], V=[VA, VB, VH], U_innov=Uinnov,
                 Tp=Tp, N=N, P=1,
                 meta=dict(rho_y=rho_y, sigma_u=sigma_u, noise=noise,
                           coh_A=coherence(VA), coh_B=coherence(VB),
                           coh_H=coherence(VH)))


# --------------------------------------------------------------------------- #
#  P = 2 (AR(2)) simulator -- validates the empirical estimator (spec sec 13)
# --------------------------------------------------------------------------- #
def simulate_ar2(Tp, N, rng, *, rA=1, rB=1, rH=2, a_rms=0.35, b_rms=0.20,
                 h_rms=0.50, sigma_u=0.30, burn=50, radius_cap=0.90, noise="iid"):
    """Heterogeneous low-rank AR(2): y_it = a_ti y_{i,t-1} + b_ti y_{i,t-2}
    + h_ti + u_it, with rank-1 lag surfaces and a rank-rH interactive block.
    Per-cell companion spectral radius is capped for stability."""
    A0, UA, sA, VA = make_surface(Tp, N, rA, rng, a_rms, positive=True, phase=0.0)
    B0, UB, sB, VB = make_surface(Tp, N, rB, rng, b_rms, phase=0.7)
    H0, UH, sH, VH = make_surface(Tp, N, rH, rng, h_rms, phase=1.1)
    # enforce per-cell stability: rescale (a,b) where companion radius too large
    for _ in range(50):
        disc = A0 ** 2 + 4 * B0
        rad = np.where(disc >= 0,
                       np.maximum(np.abs((A0 + np.sqrt(np.abs(disc))) / 2),
                                  np.abs((A0 - np.sqrt(np.abs(disc))) / 2)),
                       np.sqrt(np.abs(-B0)))
        m = rad.max()
        if m <= radius_cap:
            break
        A0 *= radius_cap / m
        B0 *= (radius_cap / m) ** 2
    UA, sA, VAt = np.linalg.svd(A0, full_matrices=False); UA, VA = UA[:, :rA], VAt[:rA].T
    UB, sB, VBt = np.linalg.svd(B0, full_matrices=False); UB, VB = UB[:, :rB], VBt[:rB].T
    G0 = {"H0": H0}
    y1 = 0.1 * rng.standard_normal(N)            # y_{t-1}
    y2 = 0.1 * rng.standard_normal(N)            # y_{t-2}
    for _ in range(burn):
        u = sigma_u * rng.standard_normal(N)
        y = A0[0] * y1 + B0[0] * y2 + H0[0] + u
        y2, y1 = y1, y
    Uinnov = _innovations(Tp, N, sigma_u, rng, noise, G0)
    Y = np.empty((Tp, N)); lag1 = np.empty((Tp, N)); lag2 = np.empty((Tp, N))
    for k in range(Tp):
        y = A0[k] * y1 + B0[k] * y2 + H0[k] + Uinnov[k]
        lag1[k] = y1; lag2[k] = y2; Y[k] = y
        y2, y1 = y1, y
    return Panel(Y=Y, Z=[lag1, lag2], surfaces=[A0, B0, H0],
                 U=[UA, UB, UH], V=[VA, VB, VH], U_innov=Uinnov,
                 Tp=Tp, N=N, P=2, meta=dict(sigma_u=sigma_u, noise=noise))
