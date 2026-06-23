#!/usr/bin/env python3
"""One-command reproduction of every number in the paper's simulation and
empirical sections.

    python run_all.py --config configs/pilot.json --stage all
    python run_all.py --config configs/full.json  --stage oracle   # sec-12 gate
    python run_all.py --config configs/full.json  --stage grid
    python run_all.py --config configs/full.json  --stage purge
    python run_all.py --config configs/full.json  --stage empirical
    python run_all.py --config configs/full.json  --stage tables

Stages are resume-safe (per-replication JSONL checkpoints in outputs/sim/), so a
long grid can be stopped and restarted, or split across machines by replication.

Reproducibility: single-thread BLAS so every worker is deterministic; per-rep
seeds are SeedSequence([master, rep]).  All tables are written into
outputs/tables/ by dlrhcs.report -- nothing is transcribed by hand.
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
from dlrhcs import report
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
    mean_cov = np.mean([v["cov"] for v in agg.values()])
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
    from dlrhcs.empirical import load_zillow, run_ar2
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
    zt = os.path.join(data, "zillow_metro_top.csv")
    zb = os.path.join(data, "zillow_metro_bottom.csv")
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
    return out


def stage_tables(cfg):
    os.makedirs(TAB, exist_ok=True)
    sizes = [g[0] for g in cfg["grid"]]
    o = cfg["oracle"]
    try:
        agg_by_size = {s: json.load(open(os.path.join(SIM, f"grid_{s}.json")))
                       for s in sizes}
        oracle_agg = json.load(open(os.path.join(SIM, f"oracle_{o['Tp']}.json")))
        report.write_tex(report.convergence_table(agg_by_size, oracle_agg, sizes, o["Tp"]),
                         os.path.join(TAB, "tab_sim_convergence.tex"))
        report.write_tex(report.precision_table(agg_by_size, sizes),
                         os.path.join(TAB, "tab_sim_precision.tex"))
        report.write_tex(report.precision_figure(agg_by_size, sizes),
                         os.path.join(TAB, "fig_sim_precision_rate.tex"))
        print("[tables] wrote tab_sim_convergence/precision.tex, fig_sim_precision_rate.tex")
    except FileNotFoundError as ex:
        print(f"[tables] skip convergence/precision ({ex})")
    p = cfg["purge"]
    try:
        agg_by_q = {int(q): a for q, a in
                    json.load(open(os.path.join(SIM, f"purge_{p['Tp']}.json"))).items()}
        q_grid = [q for q in p["q_grid"] if q in agg_by_q]
        report.write_tex(report.purge_table(agg_by_q, q_grid),
                         os.path.join(TAB, "tab_sim_purge.tex"))
        report.write_tex(report.purge_figure(agg_by_q, q_grid),
                         os.path.join(TAB, "fig_sim_purge.tex"))
        print("[tables] wrote tab_sim_purge.tex, fig_sim_purge.tex")
    except FileNotFoundError as ex:
        print(f"[tables] skip purge ({ex})")
    # ---- theorem-justification table ---------------------------------------
    try:
        th = json.load(open(os.path.join(OUT, "theorems.json")))
        report.write_tex(report.theorems_table(th),
                         os.path.join(TAB, "tab_theorems.tex"))
        print("[tables] wrote tab_theorems.tex")
    except FileNotFoundError as ex:
        print(f"[tables] skip theorems ({ex})")
    # ---- empirical table + IRF figure (Zillow) -----------------------------
    try:
        z = json.load(open(os.path.join(EMP, "zillow.json")))
        rows = [("Lag-1 mean", "lag1_mean"), ("Lag-2 mean", "lag2_mean"),
                ("Lag-1, top tier", "lag1_top"), ("Lag-1, bottom tier", "lag1_bottom"),
                ("Lag-1 top-vs-bottom contrast", "lag1_contrast")]
        rows = [r for r in rows if r[1] in z.get("targets", {})]
        report.write_tex(report.empirical_table(z, rows),
                         os.path.join(TAB, "tab_emp_zillow.tex"))
        report.write_tex(report.empirical_irf_figure(z),
                         os.path.join(TAB, "fig_emp_zillow_irf.tex"))
        print("[tables] wrote tab_emp_zillow.tex, fig_emp_zillow_irf.tex")
    except FileNotFoundError as ex:
        print(f"[tables] skip empirical ({ex})")


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
                    choices=["all", "oracle", "grid", "purge", "empirical", "theorems", "tables"])
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
    if args.stage in ("all", "tables"):
        stage_tables(cfg)
    print(f"[run_all] stage={args.stage} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
