"""
Monte Carlo harness (spec sec 12) with deterministic seeds, JSONL checkpointing
(resume-safe), and an optional process-parallel backend.

Three studies:
  * ``run_grid``        -- feasible convergence study over the (Tp,N) grid.
  * ``run_grid`` w/ oracle=True  -- the infeasible oracle benchmark (true tangent
                          spaces in the Riesz solve), the sec-12 checkpoint.
  * ``run_purge_sweep`` -- forward-exclusion-window sensitivity at fixed (Tp,N).

Per-replication seeds are derived as ``SeedSequence([master, rep])`` so any rep
can be (re)run independently and reproducibly, on any number of workers.

Parallelism note: NumPy/OpenBLAS deadlocks under ``fork``; we use joblib's
``loky`` backend (separate interpreters) and recommend ``OMP_NUM_THREADS=1`` so
each worker is single-threaded (set in ``run_all``).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .design import build_blocks
from .dgp import simulate
from .pipeline import Tuning, estimate
from .targets import make_target, group_weights


# --------------------------------------------------------------------------- #
#  the eight standard MC targets and their truths
# --------------------------------------------------------------------------- #
def standard_targets(blocks, Tp, N, t0=None, i0=None):
    """Entry, group mean, full mean and between-group contrast for lag & slope."""
    t0 = Tp // 2 if t0 is None else t0
    i0 = N // 2 if i0 is None else i0
    g1 = np.arange(0, N // 2)
    g2 = np.arange(N // 2, N)
    w1, w2 = group_weights(N, g1), group_weights(N, g2)
    wf = np.full(N, 1.0 / N)
    targets = []
    for blk, lab in [(0, "lag"), (1, "slope")]:
        targets += [
            make_target(blocks, f"{lab}_entry", blk, "entry", t=t0, i=i0),
            make_target(blocks, f"{lab}_gmean", blk, "mean", t=t0, weights=w1),
            make_target(blocks, f"{lab}_fmean", blk, "mean", t=t0, weights=wf),
            make_target(blocks, f"{lab}_contrast", blk, "contrast", t=t0,
                        weights=w1, weights2=w2),
        ]
    ctx = dict(t0=t0, i0=i0, w1=w1, w2=w2, wf=wf)
    return targets, ctx


def true_value(panel, tg, ctx):
    S = {0: panel.surfaces[0], 1: panel.surfaces[1]}[tg.block]
    name, t0 = tg.name, ctx["t0"]
    if "entry" in name:
        return float(S[t0, ctx["i0"]])
    if "gmean" in name:
        return float(S[t0] @ ctx["w1"])
    if "fmean" in name:
        return float(S[t0] @ ctx["wf"])
    return float(S[t0] @ (ctx["w1"] - ctx["w2"]))


# --------------------------------------------------------------------------- #
#  one replication
# --------------------------------------------------------------------------- #
def run_replication(Tp, N, rep, tuning: Tuning, *, oracle=False,
                    dgp_kwargs=None, master=2024) -> Dict:
    dgp_kwargs = dgp_kwargs or {}
    sim_rng = np.random.default_rng(np.random.SeedSequence([master, rep]))
    est_rng = np.random.default_rng(np.random.SeedSequence([master + 1, rep]))
    panel = simulate(Tp, N, sim_rng, **dgp_kwargs)
    blocks = build_blocks(panel.Z)
    targets, ctx = standard_targets(blocks, Tp, N)
    res = estimate(panel.Y, panel.Z, targets, tuning, rng=est_rng,
                   oracle=oracle, true_U=panel.U, true_V=panel.V)
    rec = {"rep": int(rep)}
    for tg in targets:
        v = true_value(panel, tg, ctx)
        lo, hi = res.ci[tg.name]
        lox, hix = res.ci_xs[tg.name]
        rec[tg.name] = dict(err=res.estimates[tg.name] - v,
                            se=res.se[tg.name], se_xs=res.se_xs[tg.name],
                            cov=int(lo <= v <= hi), cov_xs=int(lox <= v <= hix))
    rec["_q"], rec["_J"], rec["_ranks"] = res.q, res.J, list(res.ranks)
    rec["_monotone"] = bool(res.diagnostics["monotone"])
    return rec


# --------------------------------------------------------------------------- #
#  checkpointed grid runner
# --------------------------------------------------------------------------- #
def _done_reps(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["rep"])
            except Exception:
                pass
    return done


def run_grid(Tp, N, R, tuning: Tuning, out_path, *, oracle=False,
             dgp_kwargs=None, master=2024, n_jobs=1, resume=True):
    """Run R replications at (Tp,N), appending JSONL records to ``out_path``.

    Resume-safe: already-recorded reps are skipped.  ``n_jobs>1`` uses joblib's
    loky backend if available.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done_reps(out_path) if resume else set()
    todo = [r for r in range(R) if r not in done]
    if not todo:
        return out_path

    def work(rep):
        return run_replication(Tp, N, rep, tuning, oracle=oracle,
                               dgp_kwargs=dgp_kwargs, master=master)

    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(work)(r) for r in todo)
    else:
        recs = [work(r) for r in todo]

    with open(out_path, "a") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
#  forward-exclusion-window sweep
# --------------------------------------------------------------------------- #
def run_purge_sweep(Tp, N, R, q_grid, base_tuning: Tuning, out_dir, *,
                    master=2024, n_jobs=1, dgp_kwargs=None):
    paths = {}
    for q in q_grid:
        tun = Tuning(**{**asdict(base_tuning), "q": int(q)})
        p = os.path.join(out_dir, f"purge_q{q}_{Tp}.jsonl")
        run_grid(Tp, N, R, tun, p, dgp_kwargs=dgp_kwargs,
                 master=master + 100 * int(q), n_jobs=n_jobs)
        paths[int(q)] = p
    return paths


# --------------------------------------------------------------------------- #
#  aggregation
# --------------------------------------------------------------------------- #
def aggregate(path) -> Dict[str, dict]:
    recs = [json.loads(l) for l in open(path)]
    names = [k for k in recs[0] if not k.startswith("_") and k != "rep"]
    out = {}
    for nm in names:
        err = np.array([r[nm]["err"] for r in recs])
        se = np.array([r[nm]["se"] for r in recs])
        se_xs = np.array([r[nm]["se_xs"] for r in recs])
        cov = np.array([r[nm]["cov"] for r in recs])
        cov_xs = np.array([r[nm]["cov_xs"] for r in recs])
        out[nm] = dict(R=len(recs), bias=float(err.mean()),
                       rmse=float(np.sqrt((err ** 2).mean())),
                       mean_se=float(se.mean()), mean_se_xs=float(se_xs.mean()),
                       mc_sd=float(err.std()), cov=float(cov.mean()),
                       cov_xs=float(cov_xs.mean()))
    return out


def print_table(agg, title=""):
    if title:
        print(title)
    head = f"{'target':16s} {'bias':>8s} {'rmse':>8s} {'mean_se':>8s} {'mc_sd':>8s} {'cov95':>6s}"
    print(head)
    for nm, r in agg.items():
        print(f"{nm:16s} {r['bias']:8.4f} {r['rmse']:8.4f} {r['mean_se']:8.4f} "
              f"{r['mc_sd']:8.4f} {r['cov']:6.3f}")
