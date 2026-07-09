"""
Monte Carlo data-generating process (spec sec 3, eq:sim_dgp, tab:sim_design).

Baseline P = K = 1 model

    y_{it} = a_{0,ti} y_{i,t-1} + x_{it} beta_{0,ti} + h_{0,ti} + u_{it},
    u_{it} ~ iid N(0, sigma_u^2).

Conditional on the exogenous frame G_0 (the rank-one surfaces, the regressor
paths, the burn-in initial conditions, and the fold assignment) the innovations
are mutually independent, mean zero, with deterministic conditional variances --
exactly the predictable-weight structure the CLT needs.  The legacy simulator
keeps three innovation laws, all conditional-mean-zero:

  * ``'iid'``    : u = sigma_u * N(0,1)                    (the paper's baseline)
  * ``'hetero'`` : u = sigma(G_0)_{it} * N(0,1)            (deterministic sigma^2)
  * ``'xs'``     : within-period cross-sectionally correlated, independent across
                   time slices (for the cross-sectional variance study).

The revised Monte Carlo designs are selected by ``dgp_type`` / ``dgp_id``:

  * ``'dgp1'`` : independent heteroskedastic errors and predetermined-free x.
  * ``'dgp2'`` : spatially dependent heteroskedastic errors and predetermined-free x.
  * ``'dgp3'`` : same errors as DGP 2, with x depending on lagged shocks.

Construction matches tab:sim_design: 50 burn-in periods, exact rank-1 surfaces
with smooth time factors and incoherent unit loadings, singular values of order
sqrt(Tp*N), the lag loading capped at 0.92*rho_y for stability, and V_B
orthogonal to V_H.  The explicit DGP 1--3 x-process is not standardized after
generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_CXI_CACHE: Dict[Tuple, Dict[str, float]] = {}


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
        # (1+theta)/(1-theta)), so the strong-mixing-over-d_N condition a:crossdep
        # holds -- this is the dependence structure thm:xs_dependence covers.  A
        # PERVASIVE common factor (row-sums ~ N) is EXCLUDED by a:crossdep
        # (manuscript: "mixing allows unrestricted local contemporaneous
        # dependence but excludes [pervasive common factors]") and is therefore
        # NOT used here.  theta controls the dependence STRENGTH; any theta<1 is
        # still geometrically strong-mixing with O(1) row-sums (=(1+theta)/
        # (1-theta)), so a larger theta strengthens the (compliant) dependence
        # without violating a:crossdep -- used to make the White-vs-spatial-kernel
        # coverage gap visible (eq:xs_estimator_main).
        theta = 0.85
        e = rng.standard_normal((Tp, N))
        out = np.empty((Tp, N))
        out[:, 0] = e[:, 0]
        s = np.sqrt(1.0 - theta ** 2)
        for i in range(1, N):
            out[:, i] = theta * out[:, i - 1] + s * e[:, i]
        return sigma_u * out
    raise ValueError(f"unknown noise model {noise}")


# --------------------------------------------------------------------------- #
#  revised Monte Carlo DGPs
# --------------------------------------------------------------------------- #
def _normalize_dgp_type(dgp_type=None, dgp_id=None) -> Optional[str]:
    """Normalize user-facing DGP selectors to 'dgp1', 'dgp2', or 'dgp3'."""
    raw = dgp_type if dgp_type is not None else dgp_id
    if dgp_type is not None and dgp_id is not None:
        a = _normalize_dgp_type(dgp_type)
        b = _normalize_dgp_type(dgp_id)
        if a != b:
            raise ValueError(f"conflicting dgp_type={dgp_type!r} and dgp_id={dgp_id!r}")
        return a
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        raw = f"dgp{int(raw)}"
    key = str(raw).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "1": "dgp1",
        "dgp1": "dgp1",
        "hetero": "dgp1",
        "independenthetero": "dgp1",
        "independentheteroskedastic": "dgp1",
        "2": "dgp2",
        "dgp2": "dgp2",
        "spatial": "dgp2",
        "spatialhetero": "dgp2",
        "spatialheteroskedastic": "dgp2",
        "3": "dgp3",
        "dgp3": "dgp3",
        "predetermined": "dgp3",
        "predeterminedcovariates": "dgp3",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("dgp_type/dgp_id must select DGP 1, DGP 2, or DGP 3") from exc


def _spatial_ar1_normals(T, N, rho_s, rng):
    """N(0, R_N) draws with R_ij = rho_s**|i-j| on the unit lattice."""
    e = rng.standard_normal((T, N))
    z = np.empty_like(e)
    z[:, 0] = e[:, 0]
    scale = np.sqrt(max(1.0 - rho_s ** 2, 0.0))
    for j in range(1, N):
        z[:, j] = rho_s * z[:, j - 1] + scale * e[:, j]
    return z


def _revised_errors(total, N, rng, kind, rho_s):
    """Draw the burn-in plus effective-sample errors for DGP 1--3."""
    sigma2 = rng.uniform(0.5, 1.5, size=N)
    sigma = np.sqrt(sigma2)
    if kind == "dgp1":
        base = rng.standard_normal((total, N))
    elif kind in ("dgp2", "dgp3"):
        base = _spatial_ar1_normals(total, N, rho_s, rng)
    else:
        raise ValueError(f"unknown revised DGP {kind!r}")
    return sigma[None, :] * base, sigma, sigma2


def _revised_x(total, N, rng, Ufull, kind, rho_x, delta_x, rho_fx, eta_x):
    """Generate the unstandardized regressor process for DGP 1--3."""
    sigma_e2 = rng.uniform(0.5, 1.5, size=N)
    sigma_e = np.sqrt(sigma_e2)
    lambda_x = rng.standard_normal(N)
    X = np.empty((total, N))
    fx = np.empty(total)
    x_prev = np.zeros(N)        # x_i,-50 = 0
    f_prev = 0.0                # f_x,-50 = 0
    u_prev = np.zeros(N)
    x_innov_scale = np.sqrt(max(1.0 - rho_x ** 2, 0.0))
    f_innov_scale = np.sqrt(max(1.0 - rho_fx ** 2, 0.0))
    for t in range(total):
        f_cur = rho_fx * f_prev + f_innov_scale * rng.standard_normal()
        e_cur = sigma_e * rng.standard_normal(N)
        pred = eta_x * u_prev if kind == "dgp3" else 0.0
        x_cur = rho_x * x_prev + delta_x * lambda_x * f_cur + x_innov_scale * e_cur + pred
        X[t] = x_cur
        fx[t] = f_cur
        x_prev = x_cur
        f_prev = f_cur
        u_prev = Ufull[t]
    return X, fx, lambda_x, sigma_e, sigma_e2


def _ar1_factor(total, rho, rng):
    out = np.empty(total)
    prev = 0.0
    scale = np.sqrt(max(1.0 - rho ** 2, 0.0))
    for t in range(total):
        cur = rho * prev + scale * rng.standard_normal()
        out[t] = cur
        prev = cur
    return out


def _svd_space(M, r=1):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :r], s[:r], Vt[:r].T


def _canonical_revised_components(total, N, rng, kind, rho_x, delta_x,
                                  rho_fx, rho_s, eta_x):
    rho_g = 0.5
    c_h = float(np.sqrt(0.3 / 0.7))

    g_a = _ar1_factor(total, rho_g, rng)
    g_b = _ar1_factor(total, rho_g, rng)
    g_h = _ar1_factor(total, rho_g, rng)
    f_a = 0.5 + 0.1 * g_a
    f_b = 0.6 + 0.2 * g_b

    lambda_a = rng.normal(1.0, 0.1, size=N)
    lambda_b = rng.normal(1.0, 0.4, size=N)
    lambda_h = rng.standard_normal(N)

    Araw = f_a[:, None] * lambda_a[None, :]
    c_a = float(min(1.0, 0.85 / max(float(np.max(np.abs(Araw))), 1e-12)))
    A = c_a * Araw
    Beta = f_b[:, None] * lambda_b[None, :]
    H = c_h * g_h[:, None] * lambda_h[None, :]

    Ufull, sigma_i, sigma_i2 = _revised_errors(total, N, rng, kind, rho_s)
    Xfull, fx, lambda_x, sigma_e, sigma_e2 = _revised_x(
        total, N, rng, Ufull, kind, rho_x, delta_x, rho_fx, eta_x)
    return dict(A=A, Beta=Beta, H=H, U=Ufull, X=Xfull,
                c_a=c_a, c_h=c_h, rho_g=rho_g,
                g_a=g_a, g_b=g_b, g_h=g_h, f_a=f_a, f_b=f_b,
                lambda_a=lambda_a, lambda_b=lambda_b, lambda_h=lambda_h,
                sigma_i=sigma_i, sigma_i2=sigma_i2,
                sigma_e=sigma_e, sigma_e2=sigma_e2,
                lambda_x=lambda_x, f_x=fx)


def _centered_ss(M):
    Z = np.asarray(M, dtype=float)
    Z = Z - Z.mean()
    return float(np.sum(Z * Z))


def _outcome_parts(A, Beta, X, Xi, burn):
    total, N = A.shape
    y0 = np.empty((total, N))
    yxi = np.empty((total, N))
    y0_prev = np.zeros(N)
    yxi_prev = np.zeros(N)
    for t in range(total):
        y0_cur = A[t] * y0_prev + Beta[t] * X[t]
        yxi_cur = A[t] * yxi_prev + Xi[t]
        y0[t] = y0_cur
        yxi[t] = yxi_cur
        y0_prev = y0_cur
        yxi_prev = yxi_cur
    return y0[burn:], yxi[burn:]


def _solve_c_xi(y0_eff, yxi_eff, xi_eff, target=0.5):
    sxi = _centered_ss(xi_eff)
    y0c = y0_eff - y0_eff.mean()
    yxc = yxi_eff - yxi_eff.mean()
    Acoef = float(np.sum(y0c * y0c))
    Bcoef = float(np.sum(y0c * yxc))
    Ccoef = float(np.sum(yxc * yxc))
    # PR2 = target means c^2 Sxi / Var(y0 + c yxi) = 1 - target.
    q2 = (1.0 - target) * Ccoef - sxi
    q1 = 2.0 * (1.0 - target) * Bcoef
    q0 = (1.0 - target) * Acoef
    roots = np.roots([q2, q1, q0]) if abs(q2) > 1e-14 else np.roots([q1, q0])
    candidates = [float(np.real(z)) for z in roots if abs(np.imag(z)) < 1e-8 and np.real(z) > 0]
    if candidates:
        return min(candidates)

    grid = np.geomspace(1e-4, 100.0, 2000)
    vals = []
    for c in grid:
        denom = _centered_ss(y0_eff + c * yxi_eff)
        vals.append(abs((1.0 - c * c * sxi / max(denom, 1e-12)) - target))
    return float(grid[int(np.argmin(vals))])


def _pr2_coeffs(y0_eff, yxi_eff, xi_eff):
    y0c = y0_eff - y0_eff.mean()
    yxc = yxi_eff - yxi_eff.mean()
    return (float(np.sum(y0c * y0c)),
            float(np.sum(y0c * yxc)),
            float(np.sum(yxc * yxc)),
            _centered_ss(xi_eff))


def _pr2_from_coeffs(coeffs, c_xi):
    Acoef, Bcoef, Ccoef, sxi = coeffs
    denom = Acoef + 2.0 * c_xi * Bcoef + c_xi * c_xi * Ccoef
    return float(1.0 - c_xi * c_xi * sxi / max(denom, 1e-12))


def _pr2_for_c(y0_eff, yxi_eff, xi_eff, c_xi):
    return _pr2_from_coeffs(_pr2_coeffs(y0_eff, yxi_eff, xi_eff), c_xi)


def _calibration_seed(Tp, N, kind, draw):
    dgp_num = {"dgp1": 1, "dgp2": 2, "dgp3": 3}[kind]
    return (87321 + 1009 * int(Tp) + 917 * int(N) + 101 * dgp_num
            + 104729 * int(draw))


def _mean_pr2(coeffs, c_xi):
    vals = np.array([_pr2_from_coeffs(coef, c_xi) for coef in coeffs], dtype=float)
    return float(vals.mean())


def _solve_average_c_xi(coeffs, target=0.5):
    def gap(c):
        return _mean_pr2(coeffs, c) - target

    lo = 0.0
    hi = 1.0
    flo = gap(lo)
    fhi = gap(hi)
    while flo * fhi > 0.0 and hi < 1e4:
        hi *= 2.0
        fhi = gap(hi)

    if flo * fhi <= 0.0:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = gap(mid)
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    grid = np.geomspace(1e-4, 1e4, 2000)
    vals = [abs(gap(c)) for c in grid]
    return float(grid[int(np.argmin(vals))])


def _calibrated_c_xi_info(Tp, N, kind, burn, rho_x, delta_x, rho_fx, rho_s,
                          eta_x, c_xi_calibration_draws=100):
    K = int(c_xi_calibration_draws)
    if K < 1:
        raise ValueError("c_xi_calibration_draws must be at least 1")
    key = (int(Tp), int(N), kind, int(burn), float(rho_x), float(delta_x),
           float(rho_fx), float(rho_s), float(eta_x), K)
    if key not in _CXI_CACHE:
        total = int(Tp) + int(burn)
        coeffs = []
        for draw in range(K):
            rng = np.random.default_rng(_calibration_seed(Tp, N, kind, draw))
            comp = _canonical_revised_components(total, int(N), rng, kind, rho_x,
                                                 delta_x, rho_fx, rho_s, eta_x)
            xi = comp["H"] + comp["U"]
            y0, yxi = _outcome_parts(comp["A"], comp["Beta"], comp["X"], xi, int(burn))
            coeffs.append(_pr2_coeffs(y0, yxi, xi[int(burn):]))
        c_xi = _solve_average_c_xi(coeffs, target=0.5)
        pr2_vals = np.array([_pr2_from_coeffs(coef, c_xi) for coef in coeffs], dtype=float)
        _CXI_CACHE[key] = dict(c_xi=float(c_xi), PR2_target=0.5,
                               PR2_calibration_mean=float(pr2_vals.mean()),
                               PR2_calibration_std=float(pr2_vals.std(ddof=1) if K > 1 else 0.0),
                               c_xi_calibration_draws=K)
    return _CXI_CACHE[key]


def _calibrated_c_xi(Tp, N, kind, burn, rho_x, delta_x, rho_fx, rho_s, eta_x,
                     c_xi_calibration_draws=100):
    return _calibrated_c_xi_info(Tp, N, kind, burn, rho_x, delta_x, rho_fx,
                                 rho_s, eta_x, c_xi_calibration_draws)["c_xi"]


def _summary_stats(M):
    Z = np.asarray(M, dtype=float)
    return dict(mean=float(np.mean(Z)), std=float(np.std(Z)),
                min=float(np.min(Z)), max=float(np.max(Z)))


def _simulate_revised(Tp, N, rng, *, r, burn, dgp_kind, rho_y, sigma_u, noise,
                      rho_x, delta_x, rho_fx, rho_s, eta_x,
                      c_xi_calibration_draws):
    if r != 1:
        raise ValueError("canonical revised DGP currently has true rank r=1 for A, beta, and H")
    total = Tp + burn
    comp = _canonical_revised_components(total, N, rng, dgp_kind, rho_x,
                                         delta_x, rho_fx, rho_s, eta_x)
    c_xi_info = _calibrated_c_xi_info(Tp, N, dgp_kind, burn, rho_x, delta_x,
                                      rho_fx, rho_s, eta_x,
                                      c_xi_calibration_draws)
    c_xi = c_xi_info["c_xi"]
    xi = comp["H"] + comp["U"]
    y_prev = np.zeros(N)
    Yfull = np.empty((total, N))
    Ylag_full = np.empty((total, N))
    for t in range(total):
        y_cur = comp["A"][t] * y_prev + comp["Beta"][t] * comp["X"][t] + c_xi * xi[t]
        Ylag_full[t] = y_prev
        Yfull[t] = y_cur
        y_prev = y_cur

    sl = slice(burn, burn + Tp)
    A0 = comp["A"][sl]
    B0 = comp["Beta"][sl]
    Hraw = comp["H"][sl]
    H0 = c_xi * Hraw
    Xeff = comp["X"][sl]
    Uraw_eff = comp["U"][sl]
    Uinnov = c_xi * Uraw_eff
    Y = Yfull[sl]
    Ylag = Ylag_full[sl]
    xi_eff = xi[sl]
    pr2 = _pr2_for_c(*_outcome_parts(comp["A"], comp["Beta"], comp["X"], xi, burn),
                     xi_eff, c_xi)

    UA, sA, VA = _svd_space(A0, 1)
    UB, sB, VB = _svd_space(B0, 1)
    UH, sH, VH = _svd_space(H0, 1)

    meta = dict(rho_y=rho_y, sigma_u=sigma_u, noise=noise,
                dgp_type=dgp_kind, rho_x=rho_x, delta_x=delta_x, rho_fx=rho_fx,
                rho_s=rho_s if dgp_kind in ("dgp2", "dgp3") else 0.0,
                eta_x=eta_x if dgp_kind == "dgp3" else 0.0,
                rho_g=comp["rho_g"], c_a=comp["c_a"], c_h=comp["c_h"],
                c_xi=float(c_xi), PR2_target=c_xi_info["PR2_target"],
                PR2_realized=pr2,
                PR2_calibration_mean=c_xi_info["PR2_calibration_mean"],
                PR2_calibration_std=c_xi_info["PR2_calibration_std"],
                c_xi_calibration_draws=c_xi_info["c_xi_calibration_draws"],
                max_abs_a_it=float(np.max(np.abs(A0))),
                a_it_summary=_summary_stats(A0),
                beta_it_summary=_summary_stats(B0),
                sigma_i=comp["sigma_i"], sigma_i2=comp["sigma_i2"],
                sigma_e_i=comp["sigma_e"], sigma_e_i2=comp["sigma_e2"],
                lambda_a=comp["lambda_a"], lambda_b=comp["lambda_b"],
                lambda_h=comp["lambda_h"], lambda_x=comp["lambda_x"],
                f_a=comp["f_a"], f_b=comp["f_b"], f_x=comp["f_x"],
                g_a=comp["g_a"], g_b=comp["g_b"], g_h=comp["g_h"],
                Xfull=comp["X"], Ufull=comp["U"], u_it=Uraw_eff,
                h_it=Hraw, xi_it=xi_eff, U_lag_for_Xeff=comp["U"][burn - 1: burn - 1 + Tp],
                surface_names=("A0", "B0", "c_xi_H0"),
                surface_rms=tuple(float(np.sqrt(np.mean(S ** 2))) for S in (A0, B0, H0)),
                coh_A=coherence(VA), coh_B=coherence(VB), coh_H=coherence(VH))

    for name, arr in (("Y", Y), ("Ylag", Ylag), ("Xeff", Xeff), ("Uinnov", Uinnov),
                      ("A0", A0), ("B0", B0), ("H0", H0)):
        _assert_draw(name, arr, (Tp, N))

    return Panel(Y=Y, Z=[Ylag, Xeff], surfaces=[A0, B0, H0],
                 U=[UA, UB, UH], V=[VA, VB, VH], U_innov=Uinnov,
                 Tp=Tp, N=N, P=1, meta=meta)


def _assert_draw(name, arr, shape):
    if arr.shape != shape:
        raise AssertionError(f"{name} has shape {arr.shape}, expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{name} contains non-finite values")


# --------------------------------------------------------------------------- #
#  baseline P = 1 simulator
# --------------------------------------------------------------------------- #
def simulate(Tp, N, rng, *, r=1, rho_y=0.85, sigma_u=0.30, c_x=0.30,
             sigma_x=1.0, burn=50, a_rms=0.55, bh_rms=0.50, noise="iid",
             dgp_type=None, dgp_id=None, rho_x=0.5, delta_x=0.5,
             rho_fx=0.5, rho_s=0.5, eta_x=0.3,
             c_xi_calibration_draws=100):
    """Simulate one P=1 panel on the effective sample (rows t = P+1..T).

    Parameters ``dgp_type`` and ``dgp_id`` select the revised Monte Carlo designs
    (``"dgp1"``, ``"dgp2"``, ``"dgp3"`` or integer 1/2/3).  When neither is
    supplied, the legacy simulator path is used for backwards compatibility with
    the existing configs.  In the revised DGPs, ``sigma_u``, ``c_x`` and
    ``sigma_x`` are not used by the error/x generators.  The revised DGPs use
    one fixed ``c_xi`` per DGP/panel-size/tuning key, calibrated from
    ``c_xi_calibration_draws`` deterministic draws so their average PR2 equals
    the target.
    """
    P = 1
    dgp_kind = _normalize_dgp_type(dgp_type, dgp_id)
    if dgp_kind is not None:
        return _simulate_revised(Tp, N, rng, r=r, burn=burn, dgp_kind=dgp_kind,
                                 rho_y=rho_y, sigma_u=sigma_u, noise=noise,
                                 rho_x=rho_x, delta_x=delta_x, rho_fx=rho_fx,
                                 rho_s=rho_s, eta_x=eta_x,
                                 c_xi_calibration_draws=c_xi_calibration_draws)
    total = Tp + burn + 1
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

    G0 = {"A0": A0, "B0": B0, "H0": H0}
    meta = dict(rho_y=rho_y, sigma_u=sigma_u, noise=noise,
                coh_A=coherence(VA), coh_B=coherence(VB), coh_H=coherence(VH),
                dgp_type=dgp_kind or "legacy")

    if dgp_kind is None:
        # ---- legacy regressor: common factor + residual identifying variation
        fx = rng.standard_normal(total)
        lx = rng.standard_normal(N)
        Xfull = c_x * fx[:, None] * lx[None, :] + sigma_x * rng.standard_normal((total, N))
        eff = Xfull[burn + 1: burn + 1 + Tp]      # the effective-sample window
        Xfull = (Xfull - eff.mean()) / eff.std()  # legacy standardization
        Ufull = None
    else:
        Ufull, sigma_i, sigma_i2 = _revised_errors(total, N, rng, dgp_kind, rho_s)
        Xfull, fx, lx, sigma_e, sigma_e2 = _revised_x(
            total, N, rng, Ufull, dgp_kind, rho_x, delta_x, rho_fx, eta_x)
        _assert_draw("Xfull", Xfull, (total, N))
        _assert_draw("Ufull", Ufull, (total, N))
        meta.update(dict(rho_x=rho_x, delta_x=delta_x, rho_fx=rho_fx,
                         rho_s=rho_s if dgp_kind in ("dgp2", "dgp3") else 0.0,
                         eta_x=eta_x if dgp_kind == "dgp3" else 0.0,
                         sigma_i=sigma_i, sigma_i2=sigma_i2,
                         sigma_e_i=sigma_e, sigma_e_i2=sigma_e2,
                         lambda_x=lx, f_x=fx, Xfull=Xfull, Ufull=Ufull,
                         U_lag_for_Xeff=Ufull[burn: burn + Tp],
                         surface_names=("A0", "B0", "H0"),
                         surface_rms=tuple(float(np.sqrt(np.mean(S ** 2)))
                                           for S in (A0, B0, H0))))

    # ---- recursion with burn-in (coefficients reuse row 0 during burn-in) --
    y_prev = 0.1 * rng.standard_normal(N)
    for s in range(burn):
        u = Ufull[s] if Ufull is not None else sigma_u * rng.standard_normal(N)
        y_prev = A0[0] * y_prev + Xfull[s] * B0[0] + H0[0] + u
    u = Ufull[burn] if Ufull is not None else sigma_u * rng.standard_normal(N)
    y_init = A0[0] * y_prev + Xfull[burn] * B0[0] + H0[0] + u   # y at t = P

    Xeff = Xfull[burn + 1: burn + 1 + Tp]
    Uinnov = (Ufull[burn + 1: burn + 1 + Tp]
              if Ufull is not None else _innovations(Tp, N, sigma_u, rng, noise, G0))
    Ylag = np.empty((Tp, N))
    Y = np.empty((Tp, N))
    y_lag = y_init
    for k in range(Tp):
        y_cur = A0[k] * y_lag + Xeff[k] * B0[k] + H0[k] + Uinnov[k]
        Ylag[k] = y_lag
        Y[k] = y_cur
        y_lag = y_cur

    for name, arr in (("Y", Y), ("Ylag", Ylag), ("Xeff", Xeff), ("Uinnov", Uinnov),
                      ("A0", A0), ("B0", B0), ("H0", H0)):
        _assert_draw(name, arr, (Tp, N))

    return Panel(Y=Y, Z=[Ylag, Xeff], surfaces=[A0, B0, H0],
                 U=[UA, UB, UH], V=[VA, VB, VH], U_innov=Uinnov,
                 Tp=Tp, N=N, P=1,
                 meta=meta)
