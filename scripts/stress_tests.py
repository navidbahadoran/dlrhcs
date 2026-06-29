#!/usr/bin/env python3
"""Appendix stress tests (referee request).  Three Monte Carlo sweeps at a fixed panel
size, reporting coverage/RMSE of the lag full mean (the hardest, dependence-dominated
target) plus the retained training share:

  (1) FIXED-J SENSITIVITY.  The theory needs $J_{TN}\\to\\infty$; the runs use $J=6$.
      Sweep $J\\in\\{6,8,10,12\\}$ to show the finite-$J$ approximation is stable.
  (2) RANK MISSPECIFICATION.  True lag rank is 2; estimate with lag rank 1 (under),
      2 (correct), 3 (mild over) to show robustness to rank choice.
  (3) PERSISTENCE NEAR THE STABILITY BOUNDARY.  Sweep the stability modulus
      $\\rho_y\\in\\{0.85,0.92,0.96,0.99\\}$, since the dynamic-leakage tail
      $\\rho_*^{q}\\sqrt{TN}$ degrades as persistence approaches one.

The two remaining referee stress tests are already provided by sibling scripts:
no-buffer / insufficient-buffer failure -> ``scripts/fold_comparison.py``;
cross-sectional dependence (diagonal vs spatial-kernel) -> ``scripts/xs_stress.py``.

Resume-safe (JSONL checkpoints).  Run from the repo root:
    python scripts/stress_tests.py            # all three
    python scripts/stress_tests.py jsweep     # one sweep by name
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.mc import aggregate, run_grid          # noqa: E402
from dlrhcs.pipeline import Tuning                 # noqa: E402

SIM = os.path.join(ROOT, "outputs", "sim")
SEED, TP, N, R = 2024, 100, 100, 400


def _base():
    t = json.load(open(os.path.join(ROOT, "configs", "full.json")))["tuning"]
    t["ranks"] = tuple(t["ranks"])
    return t


def _row(path):
    a = aggregate(path)
    lf = a["lag_fmean"]
    return dict(lag_cov=lf["cov"], lag_cov_xs=lf["cov_xs"], lag_rmse=lf["rmse"],
                retained=a["_meta"]["retained"])


def jsweep(nj):
    base = _base()
    out = {}
    for J in (6, 8, 10, 12):
        tun = Tuning(**{**base, "J": int(J)})
        p = os.path.join(SIM, f"stress_jsweep_J{J}.jsonl")
        print(f"[jsweep] J={J}", flush=True)
        run_grid(TP, N, R, tun, p, dgp_kwargs=dict(noise="xs"), master=SEED, n_jobs=nj)
        out[f"J={J}"] = _row(p)
    return out


def rankmis(nj):
    base = _base()
    out = {}
    dgp = dict(noise="xs", r=2)            # TRUE lag rank = 2
    for lag_rank, tag in ((1, "under (1)"), (2, "correct (2)"), (3, "over (3)")):
        ranks = (lag_rank,) + tuple(base["ranks"][1:])   # (lag, x, H)
        tun = Tuning(**{**base, "ranks": ranks})
        p = os.path.join(SIM, f"stress_rank_lag{lag_rank}.jsonl")
        print(f"[rankmis] estimate lag rank {lag_rank} ({tag}); true rank 2", flush=True)
        run_grid(TP, N, R, tun, p, dgp_kwargs=dgp, master=SEED, n_jobs=nj)
        out[tag] = _row(p)
    return out


def persist(nj):
    base = _base()
    out = {}
    for rho in (0.85, 0.92, 0.96, 0.99):
        tun = Tuning(**base)
        p = os.path.join(SIM, f"stress_persist_rho{int(rho*100)}.jsonl")
        print(f"[persist] rho_y={rho}", flush=True)
        run_grid(TP, N, R, tun, p, dgp_kwargs=dict(noise="xs", rho_y=rho),
                 master=SEED, n_jobs=nj)
        out[f"rho_y={rho}"] = _row(p)
    return out


SWEEPS = {"jsweep": jsweep, "rankmis": rankmis, "persist": persist}


def main():
    nj = int(os.environ.get("N_JOBS", "-1") or -1)
    os.makedirs(SIM, exist_ok=True)
    which = [a for a in sys.argv[1:] if a in SWEEPS] or list(SWEEPS)
    res = {}
    for name in which:
        res[name] = SWEEPS[name](nj)
    summ = os.path.join(SIM, "stress_tests.json")
    prev = json.load(open(summ)) if os.path.exists(summ) else {}
    prev.update(res)
    json.dump(prev, open(summ, "w"), indent=2)
    for name, rows in res.items():
        print(f"\n=== {name} ===")
        print(f"{'cell':16}{'lag cov':>9}{'lag xs cov':>11}{'lag RMSE':>10}{'retained':>10}")
        for k, v in rows.items():
            print(f"{k:16}{v['lag_cov']:>9.3f}{v['lag_cov_xs']:>11.3f}"
                  f"{v['lag_rmse']:>10.4f}{v['retained']:>10.3f}")
    print(f"\nwrote {os.path.relpath(summ, ROOT)}")


if __name__ == "__main__":
    main()
