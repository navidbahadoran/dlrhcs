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


def write_tex(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path
