"""Turn aggregated Monte Carlo / empirical JSON into the paper's LaTeX tables.

Tables are written by the code (spec sec 14): never transcribe numbers by hand.
The builders emit booktabs tabulars keyed to the manuscript labels.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

TARGET_LABELS = {
    "lag_entry": r"Lag-loading entry $a_{0,t_0 i_0}$",
    "slope_entry": r"Slope entry $\beta_{0,t_0 i_0}$",
    "lag_gmean": r"Lag-loading group mean",
    "slope_gmean": r"Slope group mean",
    "lag_fmean": r"Lag-loading full mean",
    "slope_fmean": r"Slope full mean",
    "lag_contrast": r"Lag-loading contrast",
    "slope_contrast": r"Slope contrast",
}
ORDER = ["lag_entry", "slope_entry", "lag_gmean", "slope_gmean",
         "lag_fmean", "slope_fmean", "lag_contrast", "slope_contrast"]


def _load(path):
    return json.load(open(path))


def convergence_table(agg_by_size: Dict[int, dict], oracle_agg: dict,
                      sizes: List[int], oracle_size: int) -> str:
    cols = "l" + "rr" * len(sizes) + "r"
    head = ["Target"] + [f"\\multicolumn{{2}}{{c}}{{$T_+{{=}}N{{=}}{s}$}}"
                         for s in sizes] + ["Oracle"]
    sub = [""] + ["RMSE", "Cov"] * len(sizes) + ["Cov"]
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(head) + r" \\",
             " & ".join(sub) + r" \\", r"\midrule"]
    for t in ORDER:
        row = [TARGET_LABELS[t]]
        for s in sizes:
            a = agg_by_size[s][t]
            row += [f"{a['rmse']:.3f}", f"{a['cov']:.3f}"]
        row += [f"{oracle_agg[t]['cov']:.3f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def precision_table(agg_by_size: Dict[int, dict], sizes: List[int]) -> str:
    cols = "l" + "r" * len(sizes) + "r"
    head = ["Target"] + [f"$\\hat s_\\nu$ ({s})" for s in sizes] + ["ratio"]
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]
    for t in ORDER:
        ses = [agg_by_size[s][t]["mean_se"] for s in sizes]
        ratio = ses[0] / ses[-1] if ses[-1] else float("nan")
        row = [TARGET_LABELS[t]] + [f"{v:.4f}" for v in ses] + [f"{ratio:.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def purge_table(agg_by_q: Dict[int, dict], q_grid: List[int]) -> str:
    cols = "l" + "rr" * len(q_grid)
    head = ["Target"] + [f"\\multicolumn{{2}}{{c}}{{$q={q}$}}" for q in q_grid]
    sub = [""] + ["Cov", "$\\hat s_\\nu$"] * len(q_grid)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(head) + r" \\", " & ".join(sub) + r" \\", r"\midrule"]
    for t in ORDER:
        row = [TARGET_LABELS[t]]
        for q in q_grid:
            a = agg_by_q[q][t]
            row += [f"{a['cov']:.3f}", f"{a['mean_se']:.3f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def empirical_table(out: dict, rows: List[tuple]) -> str:
    """rows = [(display_label, target_key), ...] from run_ar2 output['targets']."""
    lines = [r"\begin{tabular}{lrrrcc}", r"\toprule",
             r"Target & Estimate & s.e. & xs s.e. & 95\% CI & 95\% xs CI \\",
             r"\midrule"]
    for label, key in rows:
        d = out["targets"][key]
        ci = f"$[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]$"
        cix = f"$[{d['ci_xs'][0]:+.3f}, {d['ci_xs'][1]:+.3f}]$"
        lines.append(f"{label} & {d['est']:+.3f} & {d['se']:.3f} & "
                     f"{d['se_xs']:.3f} & {ci} & {cix} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  figures (LaTeX-native pgfplots -- needs \usepackage{pgfplots})
# --------------------------------------------------------------------------- #
def purge_figure(agg_by_q: Dict[int, dict], q_grid: List[int]) -> str:
    """Coverage-vs-q figure (fig:purge_sensitivity): demonstrates prop:dynamic_leakage
    -- the lag means are anti-conservative at q=0, reach nominal at moderate q, and
    lose coverage once an over-long purge starves the first stage."""
    series = [("lag_fmean", "lag full mean", "square*"),
              ("lag_gmean", "lag group mean", "triangle*"),
              ("slope_fmean", "slope full mean", "*")]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.8\textwidth,height=0.5\textwidth,",
         r"  xlabel={Forward exclusion window $q$}, ylabel={Coverage},",
         r"  ymin=0.5, ymax=1.0, grid=both, legend pos=south west]"]
    for key, _lab, mark in series:
        co = " ".join(f"({q},{agg_by_q[q][key]['cov']:.3f})" for q in q_grid)
        L.append(f"\\addplot+[mark={mark},thick] coordinates {{{co}}};")
    L.append(f"\\addplot[dashed,gray,thick] coordinates "
             f"{{({q_grid[0]},0.95)({q_grid[-1]},0.95)}};")
    L.append(r"\legend{lag full mean, lag group mean, slope full mean, nominal}")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def precision_figure(agg_by_size: Dict[int, dict], sizes: List[int]) -> str:
    """log-log mean-s.e. vs (T+N): the sqrt(T+N) contraction of thm:feasible
    (the dashed reference has slope -1/2)."""
    series = [("lag_fmean", "lag full mean", "square*"),
              ("slope_fmean", "slope full mean", "triangle*"),
              ("lag_entry", "lag entry", "*")]
    L = [r"\begin{tikzpicture}",
         r"\begin{loglogaxis}[width=0.8\textwidth,height=0.5\textwidth,",
         r"  xlabel={$T_+ + N$}, ylabel={mean s.e. $\hat s_\nu$},",
         r"  grid=both, legend pos=south west]"]
    for key, _lab, mark in series:
        co = " ".join(f"({2*s},{agg_by_size[s][key]['mean_se']:.5f})" for s in sizes)
        L.append(f"\\addplot+[mark={mark},thick] coordinates {{{co}}};")
    s0 = sizes[0]; base = agg_by_size[s0]["lag_fmean"]["mean_se"]
    ref = " ".join(f"({2*s},{base*(s0/s)**0.5:.5f})" for s in sizes)
    L.append(f"\\addplot[dashed,black,thick] coordinates {{{ref}}};")
    L.append(r"\legend{lag full mean, slope full mean, lag entry, slope $-1/2$}")
    L += [r"\end{loglogaxis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def empirical_irf_figure(z: dict) -> str:
    """Empirical impulse-response path psi_h vs horizon with 95% bars
    (cor:irf_body applied to the Zillow lag means)."""
    irf = z["derived"]["irf"]
    hs = sorted(int(h) for h in irf)
    pts = " ".join(f"({h},{irf[str(h)]['est']:.4f}) +- (0,{1.96*irf[str(h)]['se']:.4f})"
                   for h in hs)
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.8\textwidth,height=0.5\textwidth,",
         r"  xlabel={Horizon $h$}, ylabel={IRF $\psi_h$}, grid=both]",
         r"\addplot+[mark=*,thick,error bars/.cd,y dir=both,y explicit] coordinates {"
         + pts + "};",
         r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  theorem-justification table
# --------------------------------------------------------------------------- #
def theorems_table(th: dict) -> str:
    """One booktabs table demonstrating the theorem-justification experiments."""
    L = [r"\begin{tabular}{llr}", r"\toprule",
         r"Result & Experiment & Value \\", r"\midrule"]
    rc = th.get("rank_consistency", {})
    for k in sorted(rc, key=lambda x: int(x.split("x")[0])):
        L.append(f"thm:rank\\_consistency & $P(\\hat\\br=\\br_0)$ at ${k}$ & "
                 f"{rc[k]['p_correct']:.2f} \\\\")
    il = th.get("irf_lrm_coverage", {})
    for k in ("irf1", "irf2", "irf4", "lrm"):
        if k in il:
            L.append(f"thm:irf / cor:irf\\_body & {k.upper()} interval coverage & "
                     f"{il[k]['cov']:.3f} \\\\")
    for key, v in th.get("xs_dependence", {}).items():
        L.append(f"thm:xs\\_dependence & {key.replace('_', chr(92)+'_')} "
                 f"White / xs cov & {v['white_cov']:.3f} / {v['xs_cov']:.3f} \\\\")
    db = th.get("debiasing_thm_feasible", {})
    if db:
        L.append(f"thm:feasible (debias) & plug-in / debiased $|$bias$|$ & "
                 f"{db['plugin_absbias']:.3f} / {db['debiased_absbias']:.3f} \\\\")
    cf = th.get("contiguous_fold_singular", {})
    if cf:
        L.append(f"lem:local\\_collinearity & min-eig scatter / contiguous & "
                 f"{cf['scatter']['min_eig']:.2e} / {cf['contiguous']['min_eig']:.2e} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def write_tex(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path
