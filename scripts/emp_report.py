#!/usr/bin/env python3
"""Regenerate the empirical figure snippets (pgfplots/TikZ) from the saved
``outputs/empirical/*.json`` run records, so the paper's empirical figures are
reproducible rather than hand-spliced.  Writes one ``.tex`` snippet per figure into
``outputs/empirical/tex/``; paste (or \\input) each into the paper.

Covers the coefficient-path and impulse-response figures for both applications, plus
the companion-root and coefficient-histogram figures when the needed fields are present.
Run from the repo root:  python scripts/emp_report.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.report import (empirical_coefpath_figure,    # noqa: E402
                           empirical_irf_figure, companion_root_figure,
                           coef_hist_figure, write_tex)

EMP = os.path.join(ROOT, "outputs", "empirical")
OUT = os.path.join(EMP, "tex")

# (json file, kind) -> coef-path configuration.  ``ref`` is read from the record so the
# dashed headline line always matches the debiased estimate actually reported.
SPECS = {
    "housing":      dict(json="zillow_A.json", series="cum_t",
                         ylabel=r"$\bar a_t+\bar b_t$ (housing)",
                         ref_key=("derived", "cumulative_persistence", "est"),
                         ymin=0.5, ymax=1.3),
    "unemployment": dict(json="unemp_B.json", series="a_t",
                         ylabel=r"$\bar a_t$ (unemployment)",
                         ref_key=("targets", "lag1_mean", "est"),
                         ymin=0.75, ymax=1.25),
}


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    wrote = []
    for name, cfg in SPECS.items():
        path = os.path.join(EMP, cfg["json"])
        if not os.path.exists(path):
            print(f"[skip] {name}: {cfg['json']} not found", flush=True)
            continue
        z = json.load(open(path))
        ref = float(_dig(z, cfg["ref_key"]))
        # coefficient path (always available: coef_path is exported by run_ar2)
        tex = empirical_coefpath_figure(z, cfg["series"], cfg["ylabel"], ref,
                                        cfg["ymin"], cfg["ymax"])
        f = os.path.join(OUT, f"fig_emp_coefpath_{name}.tex")
        write_tex(tex, f); wrote.append(f)
        # impulse-response path (only if the run exported derived.irf)
        if z.get("derived", {}).get("irf"):
            f = os.path.join(OUT, f"fig_emp_irf_{name}.tex")
            write_tex(empirical_irf_figure(z), f); wrote.append(f)
        # companion-root scatter and coefficient histogram, when present
        try:
            f = os.path.join(OUT, f"fig_emp_companion_{name}.tex")
            write_tex(companion_root_figure(z), f); wrote.append(f)
        except Exception:
            pass
        try:
            f = os.path.join(OUT, f"fig_emp_hist_{name}.tex")
            write_tex(coef_hist_figure(z), f); wrote.append(f)
        except Exception:
            pass
        print(f"[done] {name}: ref={ref:+.3f} from {cfg['json']}", flush=True)
    print("\nwrote:")
    for f in wrote:
        print("  " + os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
