#!/usr/bin/env python3
"""One-command reproduction of every number in the paper's simulation and
empirical sections.

    python run_all.py --config configs/pilot.json --stage all
    python run_all.py --config configs/full.json  --stage oracle   # sec-12 gate
    python run_all.py --config configs/full.json  --stage grid
    python run_all.py --config configs/full.json  --stage purge
    python run_all.py --config configs/full.json  --stage theorems

Stages are resume-safe (per-replication JSONL checkpoints in outputs/sim/), so a
long grid can be stopped and restarted, or split across machines by replication.
The simulation tables and figure coordinates are then built by
``scripts/sim_report.py``; the two empirical applications run separately via
``scripts/zillow_abc.py`` and ``scripts/unemp_abc.py``.

Reproducibility: single-thread BLAS so every worker is deterministic; per-rep
seeds are SeedSequence([master, rep]).  Nothing is transcribed by hand.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time

import numpy as np

from dlrhcs.mc import aggregate, print_table, run_grid, run_purge_sweep
from dlrhcs.pipeline import Tuning
from dlrhcs import experiments as X

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
SIM = os.path.join(OUT, "sim")
TAB = os.path.join(OUT, "tables")
EMP = os.path.join(OUT, "empirical")


def _tuning(cfg, overrides=None):
    t = dict(cfg["tuning"])
    t["ranks"] = tuple(t["ranks"])
    if overrides:
        t.update(overrides)
    return Tuning(**t)


def stage_oracle(cfg):
    o = cfg["oracle"]
    tun = _tuning(cfg)
    path = os.path.join(SIM, f"oracle_{o['Tp']}.jsonl")
    print(f"[oracle] (Tp,N)=({o['Tp']},{o['N']}) R={o['R']}  -- sec-12 checkpoint")
    run_grid(o["Tp"], o["N"], o["R"], tun, path, oracle=True,
             dgp_kwargs=cfg["dgp"], master=cfg["master_seed"], n_jobs=cfg["n_jobs"])
    agg = aggregate(path)
    json.dump(agg, open(os.path.join(SIM, f"oracle_{o['Tp']}.json"), "w"), indent=2)
    print_table(agg, "[oracle] aggregated")
    mean_cov = np.mean([v["cov"] for k, v in agg.items() if not k.startswith("_")])
    print(f"[oracle] mean coverage = {mean_cov:.3f}  (target band [0.93, 0.96])")
    return agg


def stage_grid(cfg, only=None):
    tun = _tuning(cfg)
    aggs = {}
    grid = [row for row in cfg["grid"] if only is None or row[0] == int(only)]
    if only is not None and not grid:
        print(f"[grid] no grid row with Tp={only}; available: "
              f"{[r[0] for r in cfg['grid']]}")
    for Tp, N, R in grid:
        path = os.path.join(SIM, f"grid_{Tp}.jsonl")
        print(f"[grid] (Tp,N)=({Tp},{N}) R={R}")
        run_grid(Tp, N, R, tun, path, dgp_kwargs=cfg["dgp"],
                 master=cfg["master_seed"], n_jobs=cfg["n_jobs"])
        aggs[Tp] = aggregate(path)
        json.dump(aggs[Tp], open(os.path.join(SIM, f"grid_{Tp}.json"), "w"), indent=2)
        print_table(aggs[Tp], f"[grid] (Tp,N)=({Tp},{N})")
    return aggs


def stage_purge(cfg, only=None):
    p = cfg["purge"]
    # purge-specific overrides: a larger J keeps the buffered training share
    # workable up to large q (roadmap Step 2 retention rule), and a slightly
    # larger riesz_ridge caps any residual near-singular debiasing solve at the
    # most over-purged windows (numerical regularization) -- so q=6 degrades
    # gracefully instead of exploding in the occasional starved replication.
    ov = {}
    if "J" in p:
        ov["J"] = p["J"]
    if "riesz_ridge" in p:
        ov["riesz_ridge"] = p["riesz_ridge"]
    tun = _tuning(cfg, ov)
    q_grid = p["q_grid"] if only is None else [int(only)]
    pdgp = cfg.get("purge_dgp", cfg["dgp"])
    paths = run_purge_sweep(p["Tp"], p["N"], p["R"], q_grid, tun, SIM,
                            master=cfg["master_seed"], n_jobs=cfg["n_jobs"],
                            dgp_kwargs=pdgp)
    aggs = {q: aggregate(pp) for q, pp in paths.items()}
    json.dump({str(q): a for q, a in aggs.items()},
              open(os.path.join(SIM, f"purge_{p['Tp']}.json"), "w"), indent=2)
    return aggs


def stage_empirical(cfg):
    from dlrhcs.empirical import load_zillow, load_metro, metro_groups, run_ar2
    os.makedirs(EMP, exist_ok=True)
    e = cfg["empirical"]
    sel = bool(e.get("select", False))   # data-driven rank selection (roadmap box)
    tun = Tuning(ranks=None if sel else tuple(e["ranks"]),
                 select=sel, use_roadmap=sel,   # keep configured q,J; select ranks
                 q=e["q"], J=e["J"], ridge=e.get("ridge", 0.1),
                 n_restarts=e["n_restarts"], n_sweeps=e["n_sweeps"],
                 riesz_tol=e["riesz_tol"], riesz_ridge=e.get("riesz_ridge", 1e-6),
                 riesz_maxiter=e.get("riesz_maxiter", 600), kappa_c=e.get("kappa_c", 1.0),
                 n_jobs=cfg.get("n_jobs", 1) if sel else 1,   # parallel rank selection
                 r_bar=tuple(e["r_bar"]) if e.get("r_bar") else None,  # fixed box caps
                 xs_kernel="cluster")   # metros have no spatial metric -> cluster-by-period
    data = os.path.join(ROOT, "data")
    out = {}
    # ---- Application 1: Zillow metro-tier house values ----------------------
    zt = os.path.join(data, "zillow", "zillow_metro_top.csv")
    zb = os.path.join(data, "zillow", "zillow_metro_bottom.csv")
    if os.path.exists(zt) and os.path.exists(zb):
        z = load_zillow(zt, zb)
        r = run_ar2(z["Y"], tun, groups=z["tier"], group_labels=("top", "bottom"),
                    rng=np.random.default_rng(cfg["master_seed"] + 7))
        r["fingerprint"] = z["fingerprint"]; r["T"], r["Nunits"] = z["T"], z["N"]
        r["n_differenced"] = z["n_differenced"]
        json.dump(r, open(os.path.join(EMP, "zillow.json"), "w"), indent=2, default=str)
        out["zillow"] = r
        print(f"[empirical] Zillow T={z['T']} N={z['N']} differenced={z['n_differenced']} "
              f"ranks={r['ranks']} lag1={r['targets']['lag1_mean']['est']:+.3f}")
    else:
        print("[empirical] Zillow CSVs not found in data/ -- skipping.")
    # ---- Application 2: metro-area unemployment (levels) --------------------
    mu = os.path.join(data, "unemp", "metro_unemployment.csv")
    me = cfg.get("metro", {})
    if me.get("enabled", False) and os.path.exists(mu):
        msel = bool(me.get("select", True))
        # Unemployment RATES are economically stationary (bounded, mean-reverting);
        # ADF lacks power at T=36, so we keep LEVELS and let the rank-r_H interactive
        # block absorb the strong common business-cycle factor (~76% of variance).
        mtun = Tuning(ranks=None if msel else tuple(me.get("ranks", [1, 1, 4])),
                      select=msel, use_roadmap=msel,
                      q=me.get("q", 1), J=me.get("J", 6), ridge=me.get("ridge", 0.5),
                      n_restarts=me.get("n_restarts", 2), n_sweeps=me.get("n_sweeps", 60),
                      riesz_tol=me.get("riesz_tol", 1e-5),
                      riesz_ridge=me.get("riesz_ridge", 1e-6),
                      riesz_maxiter=me.get("riesz_maxiter", 600),
                      kappa_c=me.get("kappa_c", 0.03),
                      n_jobs=cfg.get("n_jobs", 1) if msel else 1,
                      r_bar=tuple(me.get("r_bar", [2, 2, 6])), xs_kernel="cluster")
        u = load_metro(mu, stationarize=False)            # LEVELS
        g = metro_groups(u)
        r = run_ar2(u["Y"], mtun, groups=g, group_labels=("hi_unemp", "lo_unemp"),
                    rng=np.random.default_rng(cfg["master_seed"] + 13))
        r["fingerprint"] = u["fingerprint"]; r["T"], r["Nunits"] = u["T"], u["N"]
        r["n_differenced"] = u["n_differenced"]; r["stationarized"] = "levels"
        json.dump(r, open(os.path.join(EMP, "metro_unemployment.json"), "w"),
                  indent=2, default=str)
        out["metro"] = r
        print(f"[empirical] Metro-unemp (levels) T={u['T']} N={u['N']} ranks={r['ranks']} "
              f"lag1={r['targets']['lag1_mean']['est']:+.3f} "
              f"radius={r['derived']['companion_radius']:.3f}")
    elif me.get("enabled", False):
        print("[empirical] metro_unemployment.csv not found -- run "
              "scripts/build_metro_panel.py first.")
    return out
def stage_theorems(cfg):
    """Run the theorem-justification suite (small, illustrative scales).

    The experiments are embarrassingly parallel over replications; they run on
    ``n_jobs`` cores (joblib loky), like the grid.  At the full config this turns
    a multi-hour serial run into roughly (serial time / n_cores).
    """
    import json as _json
    tun = _tuning(cfg)
    th = cfg.get("theorems", {})
    nj = cfg.get("n_jobs", 1)
    out = {}
    dbg=th.get("debias_TpN",[200,200])
    out["debiasing_thm_feasible"] = X.debiasing_demo(
        dbg[0], dbg[1], th.get("R", 200), tun, master=cfg["master_seed"], n_jobs=nj)
    out["rank_consistency"] = {f"{Tp}x{N}": v for (Tp, N), v in X.rank_consistency(
        th.get("rank_grid", [[50, 50], [100, 100], [200, 200]]),
        th.get("R_rank", 100), tun, kappa_c=cfg["tuning"].get("kappa_c", 0.5),
        n_jobs=nj).items()}
    ir=th.get("irf_TpN",[100,100])
    out["irf_lrm_coverage"] = X.irf_lrm_coverage(
        ir[0], ir[1], th.get("R", 200), tun, oracle=True, n_jobs=nj)
    xt=th.get("xs_TpN",[120,120])
    out["xs_dependence"] = X.xs_coverage(xt[0], xt[1], th.get("R", 200), tun, n_jobs=nj)
    out["contiguous_fold_singular"] = X.contiguous_fold_singular(40, 40, tun)
    _json.dump(out, open(os.path.join(OUT, "theorems.json"), "w"),
               indent=2, default=str)
    print("[theorems] rank_consistency:",
          {k: v["p_correct"] for k, v in out["rank_consistency"].items()})
    print("[theorems] contiguous min_eig scatter=%.2e contiguous=%.2e" % (
        out["contiguous_fold_singular"]["scatter"]["min_eig"],
        out["contiguous_fold_singular"]["contiguous"]["min_eig"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--stage", default="all",
                    choices=["all", "oracle", "grid", "purge", "empirical", "theorems"])
    ap.add_argument("--only", default=None,
                    help="restrict 'grid' to one panel size (e.g. --only 200) or "
                         "'purge' to one window q (e.g. --only 4)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    for d in (SIM, TAB, EMP):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()
    if args.stage in ("all", "oracle"):
        stage_oracle(cfg)
    if args.stage in ("all", "grid"):
        stage_grid(cfg, only=args.only)
    if args.stage in ("all", "purge"):
        stage_purge(cfg, only=args.only)
    if args.stage in ("all", "empirical"):
        stage_empirical(cfg)
    if args.stage in ("all", "theorems"):
        stage_theorems(cfg)
    print(f"[run_all] stage={args.stage} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
