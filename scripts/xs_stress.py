#!/usr/bin/env python3
"""Cross-sectional-dependence stress (referee add-on): show the spatial-kernel
(dependence-robust) interval's coverage climbing toward nominal as N grows, under the
decaying spatial-AR(1) within-date error DGP, against the under-covering White interval.
Run from the repo root:  python scripts/xs_stress.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.experiments import xs_coverage                      # noqa: E402
from dlrhcs.pipeline import Tuning                              # noqa: E402

SIZES = [120, 200, 300]      # increasing T=N
R = 300
SEED = 2024


def _base_tuning():
    t = json.load(open(os.path.join(ROOT, "configs", "full.json")))["tuning"]
    t["ranks"] = tuple(t["ranks"])
    return t


def main():
    nj = int(os.environ.get("N_JOBS", "-1") or -1)
    base = _base_tuning()
    out = {}
    print(f"{'T=N':>6}{'target':>16}{'white_cov':>11}{'xs_cov':>9}")
    for s in SIZES:
        tun = Tuning(**base)
        res = xs_coverage(s, s, R, tun, master=SEED, n_jobs=nj)
        out[f"{s}x{s}"] = res
        for nm in ("lag_fmean", "slope_fmean"):
            r = res[nm]
            print(f"{s:>6}{nm:>16}{r['white_cov']:>11.3f}{r['xs_cov']:>9.3f}", flush=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "theorems_xs_stress.json"), "w"), indent=2)
    print("\nwrote outputs/theorems_xs_stress.json")


if __name__ == "__main__":
    main()
