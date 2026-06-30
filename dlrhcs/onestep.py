"""
One-step debiased estimator (spec sec 9) and studentizers (spec sec 10).

Given, for each fold ``j``, the out-of-fold fit ``Theta_hat^0_{-j}``, its residual
panel ``R_hat^0_{-j} = Y - A(Theta_hat^0_{-j})`` and the feasible weights
``Psi_hat_{nu,-j}``:

    phi_check_nu = sum_j [ p_j * <D_nu, Theta_hat_{-j}>
                           + <Pi_j Psi_hat_{nu,-j}, Pi_j R_hat_{-j}> ].

Cellwise cross-fitted objects ``Psi^cf`` and ``u^cf`` (each cell uses the fit of
the fold that does NOT contain it) feed the variance estimators:

    s2_nu      = sum_a (Psi^cf_{nu,a})^2 (u^cf_a)^2                White / sandwich
    s2_{nu,xs} = sum_t ( sum_i Psi^cf_{nu,ti} u^cf_{ti} )^2        within-period
                                                                  (cross-sectional)
For smooth functionals (IRF / long-run multiplier) we form the joint covariance
of the base lag-loading targets and apply the delta method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .design import A, theta_dot
from .targets import Target, riesz_weights


@dataclass
class FoldFit:
    surfaces: List[np.ndarray]   # Theta_hat^0_{-j}
    U: List[np.ndarray]          # tangent left spaces (estimated or true)
    V: List[np.ndarray]          # tangent right spaces
    residual: np.ndarray         # R_hat^0_{-j} = Y - A(surfaces)  (Tp x N)
    train: np.ndarray            # Pi^pur_{-j} bool
    val: np.ndarray              # Pi_j bool
    p: float
    alpha: float


@dataclass
class OneStepResult:
    estimates: Dict[str, float]
    Psi_cf: Dict[str, np.ndarray]    # cellwise cross-fitted weights per target
    u_cf: np.ndarray                 # cellwise cross-fitted residuals (Tp x N)
    plugins: Dict[str, float] = field(default_factory=dict)  # plug-in (no debias)
    riesz_diag: Dict[str, dict] = field(default_factory=dict)


def one_step(blocks, foldfits: Sequence[FoldFit], targets: Sequence[Target],
             riesz_kwargs: Optional[dict] = None) -> OneStepResult:
    riesz_kwargs = riesz_kwargs or {}
    Tp, N = blocks[0].shape
    estimates = {tg.name: 0.0 for tg in targets}
    plugins = {tg.name: 0.0 for tg in targets}
    Psi_cf = {tg.name: np.zeros((Tp, N)) for tg in targets}
    u_cf = np.zeros((Tp, N))
    diag = {tg.name: {"cg_iters": [], "converged": [], "min_eig": []}
            for tg in targets}

    for ff in foldfits:
        u_cf[ff.val] = ff.residual[ff.val]
        for tg in targets:
            plug = theta_dot(tg.direction, ff.surfaces)
            rr = riesz_weights(tg.direction, blocks, ff.U, ff.V,
                               ff.train, ff.alpha, **riesz_kwargs)
            corr = float(np.sum(ff.val * rr.Psi * ff.residual))
            estimates[tg.name] += ff.p * plug + corr
            plugins[tg.name] += ff.p * plug
            Psi_cf[tg.name][ff.val] = rr.Psi[ff.val]
            diag[tg.name]["cg_iters"].append(rr.cg_iters)
            diag[tg.name]["converged"].append(rr.converged)
            diag[tg.name]["min_eig"].append(rr.min_eig_proxy)

    return OneStepResult(estimates=estimates, Psi_cf=Psi_cf, u_cf=u_cf,
                         plugins=plugins, riesz_diag=diag)


# --------------------------------------------------------------------------- #
#  Studentizers
# --------------------------------------------------------------------------- #
def white_se(res: OneStepResult, name: str) -> float:
    Psi = res.Psi_cf[name]
    s2 = float(np.sum((Psi ** 2) * (res.u_cf ** 2)))
    TpN = res.u_cf.size
    s2 = max(s2, TpN ** -2.0)
    return np.sqrt(s2)


def xs_se(res: OneStepResult, name: str, bandwidth: Optional[int] = None,
          kernel: str = "bartlett") -> float:
    """Within-period cross-sectional dependence-robust s.e. (eq:xs_estimator_main).

    Both forms are applied within each period (innovation time-slices are
    independent across periods, ass:dependent (a)) and are PSD by construction.

    kernel="bartlett" -- the spatial-kernel HAC over the unit-index metric
        d = |i - j| with the Bartlett kernel K(x) = max(0, 1 - |x|):
            s^2_xs = sum_t sum_{i,j} K(|i-j|/b_TN) V_{ti} V_{tj},  V = Psi_cf * u_cf.
        This is eq:xs_estimator_main and requires a MEANINGFUL cross-sectional
        ordering (used in the simulation, where the dependence is spatial).
        Bandwidth satisfies b_TN -> inf, b_TN^2/(Tp+N) -> 0; default (Tp+N)^{1/3}.

    kernel="cluster" -- one-way clustering by period, s^2 = sum_t (sum_i V_{ti})^2.
        Robust to ARBITRARY within-period dependence and needs NO metric; used for
        the empirical panels, whose units have no natural cross-sectional order.
    """
    V = res.Psi_cf[name] * res.u_cf          # cross-fitted score field (Tp x N)
    Tp, N = V.shape
    if kernel == "cluster":
        s2 = float(np.sum(V.sum(axis=1) ** 2))           # one-way cluster by period
    else:                                                 # "bartlett": spatial HAC
        b = bandwidth if bandwidth is not None else max(1, int(round((Tp + N) ** (1.0 / 3.0))))
        s2 = float(np.sum(V * V))            # lag d = 0  (Bartlett weight 1)
        for d in range(1, min(b, N)):        # within-period Bartlett-weighted lags
            w = 1.0 - d / b
            if w <= 0.0:
                break
            s2 += 2.0 * w * float(np.sum(V[:, :N - d] * V[:, d:]))
    s2 = max(s2, (Tp * N) ** -2.0)           # numerical floor only
    return np.sqrt(s2)


def xs_se_geo(res: OneStepResult, name: str, D: np.ndarray, bandwidth: float) -> float:
    """Geographic spatial-kernel HAC standard error (Conley 1999): the theorem-backed
    object of thm:xs_dependence with an *explicit* metric.  ``D`` is the (N x N)
    great-circle distance matrix between the units' geographic centroids (km) and
    ``bandwidth`` is the Bartlett radius (km).  Within each period (independent across
    periods) it forms the quadratic form

        s^2 = sum_t V_t' W V_t,   W_{ij} = max(0, 1 - D_{ij}/bandwidth),
              V = Psi_cf * u_cf.

    This is the same estimand as :func:`xs_se` with ``kernel="bartlett"`` but with a
    credible geographic metric in place of the unit-index distance |i-j|, so it is the
    spatial-mixing-robust standard error the theory is stated for.  PSD is not guaranteed
    on a 2-D metric (as in Conley's HAC); a numerical floor keeps the variance positive."""
    V = res.Psi_cf[name] * res.u_cf          # cross-fitted score field (Tp x N)
    Tp, N = V.shape
    W = np.maximum(0.0, 1.0 - np.asarray(D, dtype=float) / float(bandwidth))
    s2 = float(np.sum(V * (V @ W)))          # sum_t V_t' W V_t
    s2 = max(s2, (Tp * N) ** -2.0)           # numerical floor only
    return np.sqrt(s2)


def joint_cov(res: OneStepResult, names: Sequence[str]) -> np.ndarray:
    """Joint covariance of several targets (White form) for the delta method."""
    k = len(names)
    Sig = np.zeros((k, k))
    u2 = res.u_cf ** 2
    for a in range(k):
        for b in range(a, k):
            val = float(np.sum(res.Psi_cf[names[a]] * res.Psi_cf[names[b]] * u2))
            Sig[a, b] = Sig[b, a] = val
    return Sig


def joint_cov_xs(res: OneStepResult, names: Sequence[str],
                 kernel: str = "cluster", bandwidth=None) -> np.ndarray:
    """Within-period cross-sectional-dependence-robust joint covariance of several
    targets (thm:xs_dependence), for the delta method on LINEAR dynamic summaries
    such as cumulative persistence a+b.  Mirrors :func:`xs_se`: ``"cluster"`` sums
    the within-date unit-aggregated scores (robust to arbitrary within-date
    dependence, no metric); ``"bartlett"`` is the spatial HAC over |i-j|.  PSD by
    construction (cluster: Gram of period vectors; bartlett: symmetrized lags)."""
    k = len(names)
    V = [res.Psi_cf[n] * res.u_cf for n in names]        # score fields (Tp x N)
    Tp, N = V[0].shape
    Sig = np.zeros((k, k))
    if kernel == "cluster":
        S = [v.sum(axis=1) for v in V]                   # per-date unit sums (Tp,)
        for a in range(k):
            for b in range(a, k):
                Sig[a, b] = Sig[b, a] = float(np.dot(S[a], S[b]))
    else:
        bw = bandwidth if bandwidth is not None else max(1, int(round((Tp + N) ** (1.0 / 3.0))))
        for a in range(k):
            for b in range(a, k):
                s2 = float(np.sum(V[a] * V[b]))
                for d in range(1, min(bw, N)):
                    w = 1.0 - d / bw
                    if w <= 0.0:
                        break
                    s2 += w * float(np.sum(V[a][:, :N - d] * V[b][:, d:]
                                           + V[a][:, d:] * V[b][:, :N - d]))
                Sig[a, b] = Sig[b, a] = s2
    return Sig


def delta_se(res: OneStepResult, names: Sequence[str], grad: np.ndarray) -> float:
    """Delta-method s.e. for g(phi_1..phi_k): sqrt(grad' Sigma grad)."""
    Sig = joint_cov(res, names)
    g = np.asarray(grad, dtype=float)
    return float(np.sqrt(max(g @ Sig @ g, 0.0)))


# --- IRF / LRM closed forms ------------------------------------------------ #
def irf_p1(a_hat: float, h: int):
    """Horizon-h impulse response for P=1: psi_h = a^h, grad = h a^{h-1}."""
    return a_hat ** h, h * a_hat ** (h - 1)


def lrm_p1(a_hat: float):
    """Long-run multiplier P=1: m = 1/(1-a), grad = 1/(1-a)^2."""
    return 1.0 / (1.0 - a_hat), 1.0 / (1.0 - a_hat) ** 2


def companion_p2(a1: float, a2: float) -> np.ndarray:
    return np.array([[a1, a2], [1.0, 0.0]])


def irf_p2(a1: float, a2: float, h: int):
    """Horizon-h IRF for P=2: psi_h = e1' C^h e1, with numerical gradient."""
    def f(v):
        C = companion_p2(v[0], v[1])
        M = np.linalg.matrix_power(C, h)
        return M[0, 0]
    v = np.array([a1, a2])
    val = f(v)
    eps = 1e-6
    g = np.array([(f(v + eps * e) - f(v - eps * e)) / (2 * eps)
                  for e in np.eye(2)])
    return val, g


def lrm_p2(a1: float, a2: float):
    """Long-run multiplier P=2: m = 1/(1-a1-a2), grad wrt (a1,a2)."""
    d = 1.0 - a1 - a2
    return 1.0 / d, np.array([1.0 / d ** 2, 1.0 / d ** 2])
