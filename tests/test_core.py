"""Spec sec 15 test checklist -- the correctness gates before scaling up.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
"""
import os
import sys
import dataclasses

# make `import dlrhcs` work when run directly (python tests/test_core.py) from a
# clean checkout, without requiring PYTHONPATH or an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dlrhcs.design import (A, A_adjoint, build_blocks, theta_dot)
from dlrhcs.dgp import simulate
from dlrhcs.factorridge import fit_factor_ridge
from dlrhcs.folds import make_folds
from dlrhcs.mc import run_replication
from dlrhcs.pipeline import Tuning
from dlrhcs.targets import (entry_direction, project_block, project_tangent,
                            riesz_weights, Target)


def _panel(Tp=24, N=20, seed=0, sigma_u=0.30):
    rng = np.random.default_rng(seed)
    return simulate(Tp, N, rng, sigma_u=sigma_u)


# 1. A / A* adjoint identity ------------------------------------------------- #
def test_adjoint_identity():
    p = _panel()
    blocks = build_blocks(p.Z)
    rng = np.random.default_rng(1)
    theta = [rng.standard_normal((p.Tp, p.N)) for _ in blocks]
    R = rng.standard_normal((p.Tp, p.N))
    lhs = float(np.vdot(A(theta, blocks), R))
    rhs = theta_dot(theta, A_adjoint(R, blocks))
    assert abs(lhs - rhs) < 1e-10 * (1 + abs(lhs))


# 2. forward-exclusion indexing on a hand-checked grid ----------------------- #
def test_forward_exclusion():
    Tp, N, J, q = 6, 1, 2, 2
    foldid = np.array([[0], [1], [0], [1], [0], [1]])  # alternate by time
    folds = make_folds(Tp, N, J, q, foldid=foldid)
    # fold j=1 held out at t=1,3,5 (0-based). Train excludes those AND the q=2
    # rows after each held-out row in the same unit.
    train1 = folds[1].train[:, 0]
    val1 = folds[1].val[:, 0]
    assert list(val1) == [False, True, False, True, False, True]
    # t=0 (not val, no prior held-out): train. t=2 has held-out at t=1 -> purged.
    # t=4 has held-out at t=3 -> purged. So only t=0 trains for fold 1.
    assert list(train1) == [True, False, False, False, False, False]


def test_spatial_buffer_no_wrap_and_seed_invariant_dgp_truth():
    Tp, N, J, q = 3, 5, 2, 0
    foldid = np.zeros((Tp, N), dtype=int)
    foldid[0, 0] = 1
    foldid[1, N - 1] = 1
    foldid[2, 2] = 1
    folds0 = make_folds(Tp, N, J, q, r=0, foldid=foldid)
    folds1 = make_folds(Tp, N, J, q, r=1, foldid=foldid)
    tr0 = folds0[1].train
    tr1 = folds1[1].train
    assert not np.array_equal(tr0, tr1)
    # Unit 1 (0-based 0) with r=1 removes units 0 and 1, not unit N (0-based 4).
    assert list(tr1[0]) == [False, False, True, True, True]
    # Unit N (0-based 4) removes units N-1 and N, with no circular wrap to unit 1.
    assert list(tr1[1]) == [True, True, True, False, False]
    # An interior held-out unit removes its immediate neighbours.
    assert list(tr1[2]) == [True, False, False, False, True]

    base = Tuning(ranks=(1, 1, 1), q=1, J_min=2, n_sweeps=2, n_restarts=0,
                  riesz_maxiter=25, riesz_tol=1e-6, buffer_r=0)
    dgp = dict(dgp_type="dgp3", c_xi_calibration_draws=3)
    r0 = run_replication(8, 6, 0, base, dgp_kwargs=dgp, master=777,
                         target_names=["lag_fmean"])
    r1 = run_replication(8, 6, 0, dataclasses.replace(base, buffer_r=1),
                         dgp_kwargs=dgp, master=777, target_names=["lag_fmean"])
    assert r0["_sim_seed_sequence"] == r1["_sim_seed_sequence"]
    assert r0["_est_seed_sequence"] == r1["_est_seed_sequence"]
    assert r0["lag_fmean"]["true_value"] == r1["lag_fmean"]["true_value"]
    assert r0["_r"] == 0
    assert r1["_r"] == 1
    assert r0["_retained_nonvalidation"] != r1["_retained_nonvalidation"]


# 3. ALS objective monotone non-increasing ----------------------------------- #
def test_als_monotone():
    p = _panel()
    blocks = build_blocks(p.Z)
    fit = fit_factor_ridge(p.Y, blocks, (1, 1, 1), n_restarts=1, n_sweeps=40)
    d = np.diff(fit.obj_path)
    assert np.all(d <= 1e-8 * (1 + np.abs(fit.obj_path[:-1])))
    assert fit.monotone


# 4. tangent projector idempotent & self-adjoint ----------------------------- #
def test_tangent_projector():
    rng = np.random.default_rng(2)
    Tp, N, r = 15, 12, 2
    U, _ = np.linalg.qr(rng.standard_normal((Tp, r)))
    V, _ = np.linalg.qr(rng.standard_normal((N, r)))
    X = rng.standard_normal((Tp, N))
    PX = project_block(X, U, V)
    PPX = project_block(PX, U, V)
    assert np.allclose(PX, PPX, atol=1e-10)            # P^2 = P
    Y = rng.standard_normal((Tp, N))
    a = float(np.vdot(project_block(X, U, V), Y))
    b = float(np.vdot(X, project_block(Y, U, V)))
    assert abs(a - b) < 1e-10                          # self-adjoint


# 5. Riesz representer identity (infeasible, true tangent) ------------------- #
def test_riesz_identity():
    p = _panel(Tp=20, N=16)
    blocks = build_blocks(p.Z)
    Tp, N = p.Tp, p.N
    train = np.ones((Tp, N), dtype=bool)
    D = entry_direction(blocks, 0, 3, 4)
    rr = riesz_weights(D, blocks, p.U, p.V, train, alpha=1.0,
                       ridge=1e-12, tol=1e-12)
    # <Psi, A(Delta)> = <D, Delta> for any admissible tangent Delta
    rng = np.random.default_rng(7)
    raw = [rng.standard_normal((Tp, N)) for _ in blocks]
    Delta = project_tangent(raw, p.U, p.V)
    lhs = float(np.vdot(rr.Psi, A(Delta, blocks)))
    rhs = theta_dot(D, Delta)
    assert abs(lhs - rhs) < 1e-5 * (1 + abs(rhs))


# 6. noiseless recovery at the true rank ------------------------------------- #
def test_noiseless_recovery():
    p = _panel(sigma_u=0.0)
    blocks = build_blocks(p.Z)
    fit = fit_factor_ridge(p.Y, blocks, (1, 1, 1), ridge=1e-6,
                           n_sweeps=300, n_restarts=4, tol=1e-12)
    R = p.Y - A(fit.surfaces, blocks)
    assert np.sqrt(np.mean(R ** 2)) < 1e-2


# 7. revised Monte Carlo DGP smoke checks ------------------------------------ #
def _mean_lag_cov(U, sigma, lag):
    Z = U / sigma[None, :]
    Z = Z - Z.mean(axis=0, keepdims=True)
    C = (Z.T @ Z) / Z.shape[0]
    return float(np.mean(np.diag(C, k=lag)))


def test_revised_dgp_shapes_and_finite():
    for k, dgp_type in enumerate(("dgp1", "dgp2", "dgp3")):
        p = simulate(36, 18, np.random.default_rng(100 + k), dgp_type=dgp_type)
        assert p.Y.shape == (36, 18)
        assert p.Z[0].shape == (36, 18)
        assert p.Z[1].shape == (36, 18)
        assert p.U_innov.shape == (36, 18)
        assert all(S.shape == (36, 18) for S in p.surfaces)
        assert np.all(np.isfinite(p.Y))
        assert np.all(np.isfinite(p.Z[0]))
        assert np.all(np.isfinite(p.Z[1]))
        assert np.all(np.isfinite(p.U_innov))
        assert p.meta["dgp_type"] == dgp_type
        assert p.meta["sigma_i"].shape == (18,)
        assert p.meta["sigma_e_i"].shape == (18,)
        assert np.all((0.5 <= p.meta["sigma_i2"]) & (p.meta["sigma_i2"] <= 1.5))
        assert np.all((0.5 <= p.meta["sigma_e_i2"]) & (p.meta["sigma_e_i2"] <= 1.5))
        assert abs(p.meta["c_h"] - np.sqrt(0.3 / 0.7)) < 1e-12
        assert p.meta["max_abs_a_it"] <= 0.85 + 1e-12
        assert abs(p.meta["PR2_realized"] - p.meta["PR2_target"]) < 0.15
        assert "a_it_summary" in p.meta
        assert "beta_it_summary" in p.meta


def test_revised_dgp1_heteroskedastic_independent_errors():
    p = simulate(900, 32, np.random.default_rng(201), dgp_type="dgp1")
    Uraw = p.meta["u_it"]
    emp_var = Uraw.var(axis=0)
    target_var = p.meta["sigma_i2"]
    assert np.corrcoef(emp_var, target_var)[0, 1] > 0.75
    c1 = abs(_mean_lag_cov(Uraw, p.meta["sigma_i"], 1))
    c4 = abs(_mean_lag_cov(Uraw, p.meta["sigma_i"], 4))
    assert c1 < 0.08
    assert c4 < 0.08


def test_revised_dgp2_dgp3_spatial_covariance_decay():
    for k, dgp_type in enumerate(("dgp2", "dgp3")):
        p = simulate(1200, 36, np.random.default_rng(300 + k), dgp_type=dgp_type)
        Uraw = p.meta["u_it"]
        c0 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 0)
        c1 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 1)
        c2 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 2)
        c4 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 4)
        assert 0.85 < c0 < 1.15
        assert c1 > c2 > c4
        assert c1 > 0.35
        assert c2 > 0.12
        assert c4 < 0.15


def test_revised_dgp3_uses_lagged_shocks_in_x():
    p = simulate(500, 30, np.random.default_rng(401), dgp_type="dgp3")
    burn = 50
    X = p.meta["Xfull"]
    U = p.meta["Ufull"]
    fx = p.meta["f_x"]
    lx = p.meta["lambda_x"]
    resid = X[burn:] - 0.5 * X[burn - 1:-1] - 0.5 * fx[burn:, None] * lx[None, :]
    lag_u = U[burn - 1:-1]
    cur_u = U[burn:]
    corr_lag = np.corrcoef(resid.ravel(), lag_u.ravel())[0, 1]
    corr_cur = np.corrcoef(resid.ravel(), cur_u.ravel())[0, 1]
    assert corr_lag > 0.20
    assert corr_lag > corr_cur + 0.05


# 8. Gram per-cell-average scale convention ---------------------------------- #
def test_gram_scale():
    p = _panel(Tp=18, N=14)
    blocks = build_blocks(p.Z)
    rng = np.random.default_rng(3)
    Delta = project_tangent([rng.standard_normal((p.Tp, p.N)) for _ in blocks],
                            p.U, p.V)
    AD = A(Delta, blocks)
    full = float(np.vdot(AD, AD))
    folds = make_folds(p.Tp, p.N, 6, 2, rng=rng)
    fd = folds[0]
    train_scaled = fd.alpha * float(np.vdot(AD * fd.train, AD * fd.train))
    held_scaled = (1.0 / fd.p) * float(np.vdot(AD * fd.val, AD * fd.val))
    assert 0.3 < train_scaled / full < 3.0
    assert 0.3 < held_scaled / full < 3.0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
