"""Spec sec 15 test checklist -- the correctness gates before scaling up.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
"""
import numpy as np

from dlrhcs.design import (A, A_adjoint, build_blocks, theta_dot)
from dlrhcs.dgp import simulate
from dlrhcs.factorridge import fit_factor_ridge
from dlrhcs.folds import make_folds
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
