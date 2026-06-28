#!/usr/bin/env python3
"""Build the simulation LaTeX tables and figure data from the Monte Carlo outputs in
``outputs/sim/``.  EVERY table and figure separates the two theory objects -- the lag
coefficient a_{ti} and the covariate coefficient b_{ti} -- and labels them as such,
never as a generic "target".  Reads the resume-safe JSONL checkpoints, re-aggregates
with the current battery, and writes ``.tex`` fragments + pgfplots coordinates to
``outputs/sim/tables/``.  Run after the grid/oracle/purge stages:
    python scripts/sim_report.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.mc import aggregate, studentized_sample          # noqa: E402

SIM = os.path.join(ROOT, "outputs", "sim")
OUT = os.path.join(SIM, "tables")
OBJ = {"lag": r"Lag coefficient $a_{ti}$", "slope": r"Covariate coefficient $b_{ti}$"}
TYPE = {"entry": "entry", "gmean": "group mean", "fmean": "full mean", "contrast": "contrast"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def _aggs(prefix, sizes):
    out = {}
    for T in sizes:
        p = os.path.join(SIM, f"{prefix}_{T}.jsonl")
        if os.path.exists(p):
            out[T] = aggregate(p)
    return out


def _grid_sizes():
    cfg = None
    for c in ("configs/full.json", "configs/fast.json"):
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            cfg = json.load(open(p)); break
    if cfg and "grid" in cfg:
        return sorted({row[0] for row in cfg["grid"]})
    return sorted(int(fn.split("_")[1].split(".")[0])
                  for fn in os.listdir(SIM) if fn.startswith("grid_") and fn.endswith(".jsonl"))


# --------------------------------------------------------------------------- #
#  Table 2: main finite-sample performance (lag & covariate, full mean + contrast)
# --------------------------------------------------------------------------- #
def main_performance(aggs):
    sizes = sorted(aggs)
    L = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"$T{=}N$ & bias & RMSE & mean s.e. & cov. & xs cov. & xs len. \\"]
    for obj in ("lag", "slope"):
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{l}{\emph{" + OBJ[obj] + r"}} \\")
        for tt in ("fmean", "contrast"):
            nm = f"{obj}_{tt}"
            L.append(r"\multicolumn{7}{l}{\quad " + TYPE[tt] + r"} \\")
            for T in sizes:
                a = aggs[T][nm]
                L.append(f"\\quad\\quad {T} & {f(a['bias'])} & {f(a['rmse'])} & {f(a['mean_se'])} "
                         f"& {f(a['cov'])} & {f(a['cov_xs'])} & {f(a['ci_len_xs'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Table 3: target-type comparison at the largest grid cell (panels per object)
# --------------------------------------------------------------------------- #
def target_type(aggs):
    T = max(aggs)
    a = aggs[T]
    L = [r"\begin{tabular}{l r r r r}", r"\toprule",
         r"Target type & bias & RMSE & cov. & xs cov. \\"]
    for obj in ("lag", "slope"):
        L.append(r"\midrule")
        L.append(r"\multicolumn{5}{l}{\emph{" + OBJ[obj] + f"}}}} (\\,$T{{=}}N{{=}}{T}$) \\\\")
        for tt in ("entry", "gmean", "fmean", "contrast"):
            d = a[f"{obj}_{tt}"]
            L.append(f"\\quad {TYPE[tt]} & {f(d['bias'])} & {f(d['rmse'])} "
                     f"& {f(d['cov'])} & {f(d['cov_xs'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Table 5: debiased vs plug-in (lag & covariate)
# --------------------------------------------------------------------------- #
def debiased_vs_plugin(aggs):
    T = max(aggs)
    a = aggs[T]
    L = [r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"Target & plug-in bias & plug-in RMSE & debiased bias & debiased RMSE & cov. \\"]
    for obj in ("lag", "slope"):
        L.append(r"\midrule")
        L.append(r"\multicolumn{6}{l}{\emph{" + OBJ[obj] + f"}}}} (\\,$T{{=}}N{{=}}{T}$) \\\\")
        for tt in ("fmean", "contrast"):
            d = a[f"{obj}_{tt}"]
            L.append(f"\\quad {TYPE[tt]} & {f(d['plugin_bias'])} & {f(d['plugin_rmse'])} "
                     f"& {f(d['bias'])} & {f(d['rmse'])} & {f(d['cov'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Table 4: oracle vs feasible (lag & covariate, at the oracle cell)
# --------------------------------------------------------------------------- #
def oracle_vs_feasible(grid_aggs):
    orc = _aggs("oracle", _grid_sizes())
    if not orc:
        return None
    T = max(orc)
    if T not in grid_aggs:
        return None
    L = [r"\begin{tabular}{l l r r r r}", r"\toprule",
         r"Object & estimator & bias & RMSE & cov. & mean s.e. \\"]
    for obj in ("lag", "slope"):
        for tt in ("fmean", "contrast"):
            nm = f"{obj}_{tt}"
            lab = OBJ[obj] + f" ({TYPE[tt]})"
            o, gg = orc[T][nm], grid_aggs[T][nm]
            L.append(r"\midrule")
            L.append(f"{lab} & oracle & {f(o['bias'])} & {f(o['rmse'])} & {f(o['cov'])} & {f(o['mean_se'])} \\\\")
            L.append(f" & feasible & {f(gg['bias'])} & {f(gg['rmse'])} & {f(gg['cov'])} & {f(gg['mean_se'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Table 6: purge / forward-exclusion sensitivity (headline lag & covariate)
# --------------------------------------------------------------------------- #
def purge_sensitivity():
    qs = sorted(int(fn.split("_")[1][1:]) for fn in os.listdir(SIM)
                if fn.startswith("purge_q") and fn.endswith(".jsonl"))
    if not qs:
        return None
    by_q = {}
    for q in qs:
        match = [fn for fn in os.listdir(SIM) if fn.startswith(f"purge_q{q}_")]
        if match:
            by_q[q] = aggregate(os.path.join(SIM, match[0]))
    L = [r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"$q$ & lag cov. & lag RMSE & cov.\ cov. & cov.\ RMSE & retained \\"]
    L.append(r"\midrule")
    for q in sorted(by_q):
        a = by_q[q]
        L.append(f"{q} & {f(a['lag_fmean']['cov'])} & {f(a['lag_fmean']['rmse'])} "
                 f"& {f(a['slope_fmean']['cov'])} & {f(a['slope_fmean']['rmse'])} "
                 f"& {f(a['_meta']['retained'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Figure data: RMSE convergence and coverage (separate lines per object)
# --------------------------------------------------------------------------- #
def figure_coords(aggs):
    sizes = sorted(aggs)
    series = [("lag_fmean", "lag full mean"), ("lag_contrast", "lag contrast"),
              ("slope_fmean", "covariate full mean"), ("slope_contrast", "covariate contrast")]
    out = ["% RMSE convergence (x = T=N)"]
    for nm, lab in series:
        coords = " ".join(f"({T},{aggs[T][nm]['rmse']:.5f})" for T in sizes)
        out.append(f"% {lab}\n\\addplot coordinates {{{coords}}};")
    out.append("\n% coverage (White) (x = T=N)")
    for nm, lab in series:
        coords = " ".join(f"({T},{aggs[T][nm]['cov']:.4f})" for T in sizes)
        out.append(f"% {lab} White\n\\addplot coordinates {{{coords}}};")
    out.append("\n% coverage (cross-sectional)")
    for nm, lab in series:
        coords = " ".join(f"({T},{aggs[T][nm]['cov_xs']:.4f})" for T in sizes)
        out.append(f"% {lab} xs\n\\addplot coordinates {{{coords}}};")
    return "\n".join(out)


def qq_coords(aggs):
    """Studentized-statistic sample quantiles vs normal quantiles, at the largest cell,
    for one lag and one covariate target."""
    from scipy.stats import norm
    T = max(aggs)
    out = ["% QQ: sample studentized quantile vs normal quantile (x=normal, y=sample)"]
    for nm, lab in (("lag_fmean", "lag full mean"), ("slope_contrast", "covariate contrast")):
        p = os.path.join(SIM, f"grid_{T}.jsonl")
        z = np.sort(studentized_sample(p, nm, "white"))
        if len(z) == 0:
            continue
        pp = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        nq = norm.ppf(pp)
        step = max(1, len(z) // 60)
        coords = " ".join(f"({nq[i]:.3f},{z[i]:.3f})" for i in range(0, len(z), step))
        out.append(f"% {lab} (T=N={T})\n\\addplot+[only marks] coordinates {{{coords}}};")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    sizes = _grid_sizes()
    aggs = _aggs("grid", sizes)
    if not aggs:
        print("no grid_*.jsonl found in outputs/sim/; run the grid stage first")
        return
    writers = {
        "tab_sim_main_performance.tex": main_performance(aggs),
        "tab_sim_target_type.tex": target_type(aggs),
        "tab_sim_debiased_vs_plugin.tex": debiased_vs_plugin(aggs),
        "fig_sim_convergence_coords.tex": figure_coords(aggs),
        "fig_sim_qq_coords.tex": qq_coords(aggs),
    }
    ovf = oracle_vs_feasible(aggs)
    if ovf:
        writers["tab_sim_oracle_vs_feasible.tex"] = ovf
    ps = purge_sensitivity()
    if ps:
        writers["tab_sim_purge.tex"] = ps
    for fn, txt in writers.items():
        open(os.path.join(OUT, fn), "w").write(txt + "\n")
        print(f"wrote {fn} ({len(txt)} chars)")
    print(f"\ngrid cells: {sizes}; objects reported separately: lag a_ti, covariate b_ti")


if __name__ == "__main__":
    main()
