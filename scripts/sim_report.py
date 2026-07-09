#!/usr/bin/env python3
"""Build journal-ready Monte Carlo tables from ``outputs/sim/*.jsonl``.

The revised MC schema stores truth, estimates, White/diagonal inference,
spatial-kernel inference, rank-selection metadata, and fold-retention diagnostics.
This script re-aggregates JSONL checkpoints with :func:`dlrhcs.mc.aggregate` and
writes CSV plus LaTeX fragments to ``outputs/sim/tables/``.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dlrhcs.mc import aggregate  # noqa: E402

SIM = os.path.join(ROOT, "outputs", "sim")
OUT = os.path.join(SIM, "tables")

TARGET_OBJECT = {"lag": "lag", "slope": "covariate"}
TARGET_TYPE = {
    "entry": "entry",
    "gmean": "group mean",
    "fmean": "full mean",
    "contrast": "contrast",
}


def fmt(x, d=3):
    """Format numbers without leading plus signs; return -- for missing values."""
    if x is None:
        return "--"
    try:
        xx = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xx):
        return "--"
    return f"{xx:.{d}f}"


def _latex_escape(x):
    return str(x).replace("_", r"\_")


def _rank_str(rank):
    if rank is None:
        return "--"
    return "(" + ",".join(str(int(x)) for x in rank) + ")"


def _dgp_label(dgp):
    key = str(dgp or "unknown").lower()
    labels = {"dgp1": "DGP 1", "dgp2": "DGP 2", "dgp3": "DGP 3", "legacy": "legacy"}
    return labels.get(key, str(dgp or "unknown"))


def _target_label(name):
    if "_" not in name:
        return name
    obj, kind = name.split("_", 1)
    return f"{TARGET_OBJECT.get(obj, obj)} {TARGET_TYPE.get(kind, kind)}"


def _grid_jsonl_paths():
    """Simulation grid files only; excludes purge/oracle/stress/fold-comparison files."""
    paths = []
    for path in glob(os.path.join(SIM, "grid*.jsonl")):
        base = os.path.basename(path)
        if base.startswith(("grid_", "grid-")):
            paths.append(path)
    return sorted(paths)


def _load_grid_aggs(paths=None):
    out = []
    missing = []
    for path in paths or _grid_jsonl_paths():
        try:
            agg = aggregate(path)
        except Exception as exc:  # keep one stale file from blocking all tables
            missing.append({"file": path, "reason": str(exc)})
            continue
        meta = dict(agg.get("_meta", {}))
        if not meta.get("Tp") or not meta.get("N"):
            m = re.search(r"grid[_-](\d+)", os.path.basename(path))
            if m:
                meta["Tp"] = int(m.group(1))
                meta["N"] = int(m.group(1))
        if meta.get("dgp_type") in (None, "", "unknown"):
            meta["dgp_type"] = "legacy"
        out.append({"path": path, "agg": agg, "meta": meta})
    out.sort(key=lambda x: (
        str(x["meta"].get("dgp_type", "")),
        int(x["meta"].get("Tp", 0)),
        int(x["meta"].get("N", 0)),
        os.path.basename(x["path"]),
    ))
    return out, missing


def _metric(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return float("nan")


def _main_inference(row, dgp):
    """Return table-ready inference metrics using theorem-aligned SEs."""
    if str(dgp).lower() in ("dgp2", "dgp3"):
        return {
            "se_type": "spatial-kernel",
            "mean_se": _metric(row, ["mean_se_spatial_kernel", "mean_se_spatial", "mean_se_xs"]),
            "size": _metric(row, ["size_5pct_spatial_kernel", "size_5pct_spatial", "size_5pct_xs"]),
            "coverage": _metric(row, ["coverage_95_spatial_kernel", "coverage_95_spatial", "coverage_95_xs", "cov_xs"]),
            "size_mcse": _metric(row, ["size_mcse_spatial_kernel", "size_mcse_spatial", "size_mcse_xs"]),
            "coverage_mcse": _metric(row, ["coverage_mcse_spatial_kernel", "coverage_mcse_spatial", "coverage_mcse_xs"]),
        }
    return {
        "se_type": "diagonal/white",
        "mean_se": _metric(row, ["mean_se_white", "mean_se"]),
        "size": _metric(row, ["size_5pct_white", "size_5pct"]),
        "coverage": _metric(row, ["coverage_95_white", "coverage_95", "cov"]),
        "size_mcse": _metric(row, ["size_mcse_white", "size_mcse"]),
        "coverage_mcse": _metric(row, ["coverage_mcse_white", "coverage_mcse"]),
    }


def main_performance_rows(items):
    rows = []
    missing = []
    for item in items:
        agg, meta = item["agg"], item["meta"]
        dgp = meta.get("dgp_type", "unknown")
        for target, vals in agg.items():
            if target.startswith("_"):
                continue
            inf = _main_inference(vals, dgp)
            row = {
                "DGP": _dgp_label(dgp),
                "target": _target_label(target),
                "T": meta.get("Tp", ""),
                "N": meta.get("N", ""),
                "true_value": _metric(vals, ["true_value", "mean_true_value"]),
                "mean_estimate": _metric(vals, ["mean_estimate"]),
                "bias": _metric(vals, ["bias"]),
                "rmse": _metric(vals, ["rmse"]),
                "se_type": inf["se_type"],
                "mean_se": inf["mean_se"],
                "empirical_size": inf["size"],
                "size_mcse": inf["size_mcse"],
                "coverage": inf["coverage"],
                "coverage_mcse": inf["coverage_mcse"],
                "replications": vals.get("R", ""),
            }
            for key in ("true_value", "mean_estimate", "bias", "rmse", "mean_se", "empirical_size", "coverage"):
                if not np.isfinite(float(row[key])) if row[key] != "" else True:
                    missing.append({"file": item["path"], "target": target, "field": key})
            rows.append(row)
    return rows, missing


def rank_frequency_rows(items):
    rows = []
    for item in items:
        meta = item["meta"]
        rf = item["agg"].get("_rank_frequency", {})
        if not rf.get("available", False):
            continue
        if not rf.get("rank_selection_enabled", False):
            continue
        rows.append({
            "DGP": _dgp_label(rf.get("dgp_type", meta.get("dgp_type", "unknown"))),
            "T": rf.get("Tp", meta.get("Tp", "")),
            "N": rf.get("N", meta.get("N", "")),
            "J_min": rf.get("J_min", meta.get("J_min", "")),
            "kappa_c": rf.get("kappa_c", meta.get("kappa_c", "")),
            "retained_nonvalidation": meta.get("retained_nonvalidation", ""),
            "p_correct_rank": rf.get("p_correct_rank", ""),
            "p_underfit": rf.get("p_underfit", ""),
            "p_overfit": rf.get("p_overfit", ""),
            "modal_selected_rank": _rank_str(rf.get("modal_selected_rank")),
            "replications": rf.get("R", ""),
        })
    return rows


def fold_retention_rows(items):
    rows = []
    for item in items:
        meta = item["meta"]
        rows.append({
            "DGP": _dgp_label(meta.get("dgp_type", "unknown")),
            "T": meta.get("Tp", ""),
            "N": meta.get("N", ""),
            "J_min": meta.get("J_min", ""),
            "realized_J": meta.get("J_realized", meta.get("J", "")),
            "retained_nonvalidation": meta.get("retained_nonvalidation", meta.get("retained", "")),
            "retained_total": meta.get("retained_total", ""),
        })
    return rows


def _write_csv(path, rows, fields):
    def clean(val):
        if isinstance(val, (float, np.floating)) and not np.isfinite(float(val)):
            return ""
        return val

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({key: clean(row.get(key, "")) for key in fields})


def _latex_table(rows, fields, align, caption_note=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(f) for f in fields) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        vals = []
        for fkey in fields:
            val = row.get(fkey, "")
            if isinstance(val, (float, np.floating)):
                vals.append(fmt(val))
            else:
                vals.append(_latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption_note:
        lines.append("% " + caption_note)
    return "\n".join(lines)


def _write_tex(path, rows, fields, align, note=None):
    with open(path, "w") as fh:
        fh.write(_latex_table(rows, fields, align, note) + "\n")


def write_journal_tables(items):
    os.makedirs(OUT, exist_ok=True)
    perf, missing = main_performance_rows(items)
    rank = rank_frequency_rows(items)
    folds = fold_retention_rows(items)

    perf_fields = [
        "DGP", "target", "T", "N", "true_value", "mean_estimate", "bias", "rmse",
        "se_type", "mean_se", "empirical_size", "size_mcse", "coverage",
        "coverage_mcse", "replications",
    ]
    rank_fields = [
        "DGP", "T", "N", "J_min", "kappa_c", "retained_nonvalidation",
        "p_correct_rank", "p_underfit", "p_overfit", "modal_selected_rank",
        "replications",
    ]
    fold_fields = [
        "DGP", "T", "N", "J_min", "realized_J", "retained_nonvalidation",
        "retained_total",
    ]

    _write_csv(os.path.join(OUT, "tab_mc_performance.csv"), perf, perf_fields)
    _write_csv(os.path.join(OUT, "tab_rank_frequency.csv"), rank, rank_fields)
    _write_csv(os.path.join(OUT, "tab_fold_retention.csv"), folds, fold_fields)

    _write_tex(
        os.path.join(OUT, "tab_mc_performance.tex"),
        perf,
        perf_fields,
        "llrrrrrlrrrrrrr",
        "Size and coverage Monte Carlo standard errors are reported in separate columns. "
        "DGP 1 uses diagonal/White inference; DGP 2--3 use spatial-kernel inference.",
    )
    _write_tex(os.path.join(OUT, "tab_rank_frequency.tex"), rank, rank_fields, "lrrrrrrrrlr")
    _write_tex(os.path.join(OUT, "tab_fold_retention.tex"), folds, fold_fields, "lrrrrrr")
    return {"performance": perf, "rank": rank, "folds": folds, "missing": missing}


def main():
    items, load_missing = _load_grid_aggs()
    if not items:
        print("no grid*.jsonl files found in outputs/sim/")
        return
    result = write_journal_tables(items)
    for name in (
        "tab_mc_performance.csv",
        "tab_mc_performance.tex",
        "tab_rank_frequency.csv",
        "tab_rank_frequency.tex",
        "tab_fold_retention.csv",
        "tab_fold_retention.tex",
    ):
        print(f"wrote {os.path.join(OUT, name)}")
    if load_missing:
        print(f"skipped {len(load_missing)} unreadable grid file(s)")
    if result["missing"]:
        print(f"missing numeric fields in {len(result['missing'])} table cell(s); see old-schema inputs")
    print(f"grid aggregates reported: {len(items)}")


if __name__ == "__main__":
    main()
