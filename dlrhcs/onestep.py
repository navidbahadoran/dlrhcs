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


def xs_se(res: OneStepResult, name: str) -> float:
    """Within-period (cross-sectional) dependence-robust s.e.

    Cluster-by-time form of eq:xs_estimator_main: robust to arbitrary
    cross-sectional correlation within a period, independent across periods.
    """
    Psi = res.Psi_cf[name]; u = res.u_cf
    cluster = float(np.sum((Psi * u).sum(axis=1) ** 2))   # one-way cluster by time
    white = float(np.sum((Psi ** 2) * (u ** 2)))          # heteroskedastic (no dependence)
    TpN = u.size
    # the within-period (xs) variance is White + cross terms; under positive
    # dependence the cross terms are >=0, but their finite-sample estimate is
    # noisy and can be negative, so we floor the cluster estimator at White --
    # the xs s.e. is then never anti-conservative relative to White.
    s2 = max(cluster, white, TpN ** -2.0)
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
