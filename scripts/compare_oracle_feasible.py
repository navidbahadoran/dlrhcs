#!/usr/bin/env python3
"""Read-only comparison of feasible and oracle Monte Carlo JSONL outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _load_jsonl(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rep = int(rec["rep"])
            if rep in out:
                raise SystemExit(f"duplicate rep_id {rep} in {path} at line {line_no}")
            out[rep] = rec
    return out


def _se(row: dict) -> float:
    for key in ("se_spatial", "se_xs", "se_white", "se"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _ci(row: dict, err: float, se: float):
    if "ci_spatial_low" in row and "ci_spatial_high" in row:
        return float(row["ci_spatial_low"]), float(row["ci_spatial_high"])
    if "ci_xs_low" in row and "ci_xs_high" in row:
        return float(row["ci_xs_low"]), float(row["ci_xs_high"])
    if "ci_white_low" in row and "ci_white_high" in row:
        return float(row["ci_white_low"]), float(row["ci_white_high"])
    true_value = float(row["true_value"])
    estimate = true_value + err
    return estimate - 1.96 * se, estimate + 1.96 * se


def _row(rec: dict, target: str) -> dict:
    if target not in rec:
        raise KeyError(target)
    row = rec[target]
    true_value = float(row["true_value"])
    estimate = float(row["estimate"])
    err = float(row.get("err", estimate - true_value))
    se = _se(row)
    z = float(err / se) if se > 0 and math.isfinite(se) else float("nan")
    lo, hi = _ci(row, err, se)
    return {
        "true_value": true_value,
        "estimate": estimate,
        "plugin": float(row.get("plugin", row.get("plugin_estimate", float("nan")))),
        "err": err,
        "se": se,
        "z": z,
        "reject": bool(abs(z) > 1.96) if math.isfinite(z) else False,
        "covered": bool(lo <= true_value <= hi),
    }


def _quantiles(values: Iterable[float], probs=(0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99)):
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {f"q{int(p * 1000):03d}": None for p in probs}
    return {f"q{int(p * 1000):03d}": float(np.quantile(arr, p)) for p in probs}


def _summary(rows: List[dict]) -> dict:
    err = np.asarray([r["err"] for r in rows], dtype=float)
    se = np.asarray([r["se"] for r in rows], dtype=float)
    z = np.asarray([r["z"] for r in rows], dtype=float)
    finite_z = z[np.isfinite(z)]
    sd = float(np.std(err, ddof=1)) if err.size > 1 else 0.0
    mean_se = float(np.mean(se))
    out = {
        "R": int(len(rows)),
        "mean_true_value": float(np.mean([r["true_value"] for r in rows])),
        "bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "empirical_sd_error": sd,
        "mean_se": mean_se,
        "median_se": float(np.median(se)),
        "mean_se_over_empirical_sd": float(mean_se / sd) if sd > 0 else None,
        "size": float(np.mean(np.abs(z) > 1.96)),
        "coverage": float(np.mean([r["covered"] for r in rows])),
        "z_mean": float(np.mean(finite_z)) if finite_z.size else None,
        "z_sd": float(np.std(finite_z, ddof=1)) if finite_z.size > 1 else None,
        "lower_tail_rejection": float(np.mean(z < -1.96)),
        "upper_tail_rejection": float(np.mean(z > 1.96)),
    }
    out["z_quantiles"] = _quantiles(z)
    return out


def _diff_summary(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "sd": None, "median": None, "q05": None, "q95": None, "max_abs": None}
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare oracle and feasible MC JSONL outputs")
    ap.add_argument("--feasible-path", required=True, type=Path)
    ap.add_argument("--oracle-path", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    feasible = _load_jsonl(args.feasible_path)
    oracle = _load_jsonl(args.oracle_path)
    feasible_ids = set(feasible)
    oracle_ids = set(oracle)
    common_ids = sorted(feasible_ids & oracle_ids)
    missing_in_oracle = sorted(feasible_ids - oracle_ids)
    missing_in_feasible = sorted(oracle_ids - feasible_ids)
    if not common_ids:
        raise SystemExit("no common replication IDs to compare")

    paired = []
    seed_mismatches = []
    truth_diffs = []
    missing_target_ids = []
    for rep in common_ids:
        fr = feasible[rep]
        orr = oracle[rep]
        if fr.get("_sim_seed_sequence") != orr.get("_sim_seed_sequence") or fr.get("_est_seed_sequence") != orr.get("_est_seed_sequence"):
            seed_mismatches.append(rep)
        try:
            f = _row(fr, args.target)
            o = _row(orr, args.target)
        except KeyError:
            missing_target_ids.append(rep)
            continue
        truth_diff = o["true_value"] - f["true_value"]
        truth_diffs.append(abs(truth_diff))
        paired.append({
            "rep": rep,
            "feasible_true": f["true_value"],
            "oracle_true": o["true_value"],
            "truth_diff": truth_diff,
            "feasible_estimate": f["estimate"],
            "oracle_estimate": o["estimate"],
            "diff_estimate": o["estimate"] - f["estimate"],
            "feasible_se": f["se"],
            "oracle_se": o["se"],
            "diff_se": o["se"] - f["se"],
            "feasible_z": f["z"],
            "oracle_z": o["z"],
            "diff_z": o["z"] - f["z"],
            "feasible_reject": int(f["reject"]),
            "oracle_reject": int(o["reject"]),
            "diff_reject": int(o["reject"]) - int(f["reject"]),
        })

    if seed_mismatches:
        raise SystemExit(f"seed sequence mismatch for common rep_ids: {seed_mismatches[:20]}")
    max_truth_diff = max(truth_diffs) if truth_diffs else float("inf")
    if max_truth_diff > 1e-10:
        raise SystemExit(f"target truth mismatch across common reps; max abs diff={max_truth_diff:.3g}")
    if not paired:
        raise SystemExit(f"target {args.target!r} is missing from all common records")

    feasible_rows = [_row(feasible[row["rep"]], args.target) for row in paired]
    oracle_rows = [_row(oracle[row["rep"]], args.target) for row in paired]
    summary = {
        "target": args.target,
        "feasible_path": str(args.feasible_path),
        "oracle_path": str(args.oracle_path),
        "common_R": int(len(paired)),
        "missing_in_oracle_count": int(len(missing_in_oracle)),
        "missing_in_feasible_count": int(len(missing_in_feasible)),
        "missing_in_oracle_first20": missing_in_oracle[:20],
        "missing_in_feasible_first20": missing_in_feasible[:20],
        "missing_target_ids": missing_target_ids,
        "seed_sequences_match": True,
        "max_abs_truth_diff": float(max_truth_diff),
        "feasible": _summary(feasible_rows),
        "oracle": _summary(oracle_rows),
        "paired_oracle_minus_feasible": {
            "estimate": _diff_summary([row["diff_estimate"] for row in paired]),
            "se": _diff_summary([row["diff_se"] for row in paired]),
            "z": _diff_summary([row["diff_z"] for row in paired]),
            "reject": _diff_summary([row["diff_reject"] for row in paired]),
            "rejection_changed_rate": float(np.mean([row["diff_reject"] != 0 for row in paired])),
            "oracle_only_reject_count": int(sum(row["diff_reject"] == 1 for row in paired)),
            "feasible_only_reject_count": int(sum(row["diff_reject"] == -1 for row in paired)),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"oracle_feasible_{args.target}"
    json_path = args.out_dir / f"{stem}_summary.json"
    csv_path = args.out_dir / f"{stem}_paired.csv"
    metrics_path = args.out_dir / f"{stem}_metrics.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "R", "bias", "rmse", "empirical_sd_error", "mean_se",
                         "median_se", "mean_se_over_empirical_sd", "size", "coverage",
                         "z_mean", "z_sd", "lower_tail_rejection", "upper_tail_rejection"])
        for label in ("feasible", "oracle"):
            s = summary[label]
            writer.writerow([label, s["R"], s["bias"], s["rmse"], s["empirical_sd_error"],
                             s["mean_se"], s["median_se"], s["mean_se_over_empirical_sd"],
                             s["size"], s["coverage"], s["z_mean"], s["z_sd"],
                             s["lower_tail_rejection"], s["upper_tail_rejection"]])

    print(json.dumps({
        "target": args.target,
        "common_R": len(paired),
        "feasible_size": summary["feasible"]["size"],
        "oracle_size": summary["oracle"]["size"],
        "feasible_se_over_sd": summary["feasible"]["mean_se_over_empirical_sd"],
        "oracle_se_over_sd": summary["oracle"]["mean_se_over_empirical_sd"],
        "json": str(json_path),
        "paired_csv": str(csv_path),
        "metrics_csv": str(metrics_path),
    }, indent=2))


if __name__ == "__main__":
    main()
