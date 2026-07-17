#!/usr/bin/env python3
"""Resume-safe batch runner for production Monte Carlo replications.

Examples
--------
Fixed-rank performance run:

    python scripts/run_mc_batches.py --dgp-type dgp1 --T 100 --N 100 \
      --R-total 1000 --batch-size 25 --out-path outputs/sim/grid_dgp1_100.jsonl \
      --select false --fixed-ranks 1,1,1 --n-jobs 4

Rank-selection run:

    python scripts/run_mc_batches.py --dgp-type dgp3 --T 100 --N 100 \
      --R-total 100 --batch-size 10 --out-path outputs/sim/grid_rank_dgp3_100.jsonl \
      --select true --true-ranks 1,1,1 --n-jobs 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlrhcs.mc import MC_SCHEMA_VERSION, precompute_dgp_calibration, run_replication  # noqa: E402
from dlrhcs.pipeline import Tuning  # noqa: E402


def _parse_bool(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"1", "true", "t", "yes", "y"}:
        return True
    if key in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_ranks(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None or str(value).strip() == "":
        return None
    try:
        ranks = tuple(int(x.strip()) for x in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks:
        raise argparse.ArgumentTypeError("rank vector cannot be empty")
    if any(r < 0 for r in ranks):
        raise argparse.ArgumentTypeError("ranks must be nonnegative")
    return ranks


def _parse_targets(value: Optional[str]) -> Optional[list[str]]:
    if value is None or str(value).strip() == "":
        return None
    names = [x.strip() for x in str(value).split(",") if x.strip()]
    if not names:
        raise argparse.ArgumentTypeError("targets must be a comma-separated list of target names")
    return names


def _parse_nonnegative_int(value: str) -> int:
    try:
        out = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if out < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return out


def _load_config(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _runtime_metadata() -> Dict:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def _git_metadata() -> Dict:
    git = ["git", "-c", "safe.directory=*"]
    try:
        commit = subprocess.run(
            git + ["rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode != 0:
            return {
                "git_commit": None,
                "git_dirty": None,
                "git_status_available": False,
                "git_error": (commit.stderr or commit.stdout).strip() or None,
            }
        status = subprocess.run(
            git + ["status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "git_status_available": status.returncode == 0,
            "git_error": None if status.returncode == 0 else (status.stderr or status.stdout).strip() or None,
        }
    except Exception as exc:
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_status_available": False,
            "git_error": str(exc),
        }


def _tuning_from_config(cfg: Dict, *, select: bool,
                        fixed_ranks: Optional[Tuple[int, ...]],
                        rank_caps: Optional[Tuple[int, ...]]) -> Tuning:
    raw = dict(cfg.get("tuning", {}))
    if raw.get("ranks") is not None:
        raw["ranks"] = tuple(raw["ranks"])
    if raw.get("r_bar") is not None:
        raw["r_bar"] = tuple(raw["r_bar"])

    raw.setdefault("J_min", 10)
    raw.setdefault("c_J", 1.0)
    raw.setdefault("kappa_c", 0.015)

    if select:
        if fixed_ranks is not None:
            raise ValueError("--fixed-ranks fixes estimator ranks and cannot be used with --select true; use --true-ranks for rank-frequency truth")
        caps = rank_caps or raw.get("r_bar") or (1, 1, 1)
        raw["ranks"] = None
        raw["select"] = True
        raw["r_bar"] = tuple(caps)
    else:
        raw["ranks"] = tuple(fixed_ranks or raw.get("ranks") or (1, 1, 1))
        raw["select"] = False

    return Tuning(**raw)


def _done_reps(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add(int(rec["rep"]))
            except Exception:
                print(f"[warn] ignoring unreadable JSONL line {line_no} in {path}")
    return done


def _count_duplicate_reps(path: Path) -> int:
    counts: Dict[int, int] = {}
    if not path.exists():
        return 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rep = int(json.loads(line)["rep"])
            except Exception:
                continue
            counts[rep] = counts.get(rep, 0) + 1
    return int(sum(v - 1 for v in counts.values() if v > 1))


def _first_record(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def _norm_for_signature(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_norm_for_signature(v) for v in value]
    if isinstance(value, list):
        return [_norm_for_signature(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _norm_for_signature(v)
            for k, v in sorted(value.items())
            if k not in {"c_xi_info"}
        }
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 12)
    return value


def _spatial_bandwidth_for_signature(dgp_type: str, N: int):
    if str(dgp_type).lower() == "dgp1":
        return None
    return int(math.floor(int(N) ** (1.0 / 3.0)))


def _requested_signature(args, tuning: Tuning, dgp_kwargs: Dict, master: int) -> Dict:
    dgp_params = {
        key: value for key, value in dgp_kwargs.items()
        if key not in {"c_xi_info"}
    }
    return _norm_for_signature({
        "mc_schema_version": MC_SCHEMA_VERSION,
        "dgp_type": args.dgp_type,
        "T": int(args.T),
        "N": int(args.N),
        "master_seed": int(master),
        "oracle": bool(args.oracle),
        "target_filter": list(args.targets) if args.targets is not None else None,
        "select": bool(args.select),
        "fixed_ranks": list(args.fixed_ranks) if args.fixed_ranks is not None else None,
        "rank_caps": list(args.rank_caps) if args.rank_caps is not None else None,
        "q_T": int(tuning.q) if tuning.q is not None else None,
        "r_N": int(tuning.buffer_r),
        "J_min": int(tuning.J_min),
        "c_J": float(tuning.c_J),
        "kappa_c": float(tuning.kappa_c),
        "c_xi_calibration_draws": int(args.c_xi_calibration_draws),
        "dgp_parameters": dgp_params,
        "se_type": "diagonal" if str(args.dgp_type).lower() == "dgp1" else "spatial-kernel",
        "kernel": None if str(args.dgp_type).lower() == "dgp1" else "Bartlett",
        "distance_metric": None if str(args.dgp_type).lower() == "dgp1" else "lattice |i-j|",
        "h_N_formula": None if str(args.dgp_type).lower() == "dgp1" else "floor(N^(1/3))",
        "h_N_realized": _spatial_bandwidth_for_signature(args.dgp_type, args.N),
    })


def _existing_signature(path: Path, sidecar: Path) -> Optional[Dict]:
    meta = {}
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            meta = {}
    rec = _first_record(path) or {}
    if not meta and not rec:
        return None

    def get_meta_record(meta_key, rec_key=None, default=None):
        if meta_key in meta:
            return meta.get(meta_key)
        if rec_key is not None and rec_key in rec:
            return rec.get(rec_key)
        return default

    dgp_params = get_meta_record("resolved_dgp_kwargs", default=None)
    if isinstance(dgp_params, dict):
        dgp_params = dict(dgp_params)
        dgp_params.pop("c_xi_info", None)
    if dgp_params is None:
        dgp_params = {
            "dgp_type": get_meta_record("dgp_type", "_dgp_type"),
            "c_xi_calibration_draws": get_meta_record("c_xi_calibration_draws", "_c_xi_calibration_draws"),
            "rho_g": get_meta_record("rho_g", "_rho_g"),
            "rho_x": get_meta_record("rho_x", "_rho_x"),
            "rho_fx": get_meta_record("rho_fx", "_rho_fx"),
            "rho_s": get_meta_record("rho_s", "_rho_s"),
            "delta_x": get_meta_record("delta_x", "_delta_x"),
            "eta_x": get_meta_record("eta_x", "_eta_x"),
            "pi_h": get_meta_record("pi_h", "_pi_h"),
        }

    master_seed = get_meta_record("master_seed", "_master_seed")
    if master_seed is None and isinstance(meta.get("config_snapshot"), dict):
        master_seed = meta["config_snapshot"].get("master_seed")

    return _norm_for_signature({
        "mc_schema_version": get_meta_record("mc_schema_version", "_mc_schema_version"),
        "dgp_type": get_meta_record("dgp_type", "_dgp_type"),
        "T": get_meta_record("T", "_Tp"),
        "N": get_meta_record("N", "_N"),
        "master_seed": master_seed,
        "oracle": get_meta_record("oracle", "_oracle", False),
        "target_filter": get_meta_record("target_filter", "_target_filter"),
        "select": get_meta_record("select", "_rank_selection_enabled"),
        "fixed_ranks": get_meta_record("fixed_ranks", "_tuning_fixed_ranks"),
        "rank_caps": get_meta_record("rank_caps", "_rank_candidate_caps"),
        "q_T": get_meta_record("q_T", "_q"),
        "r_N": get_meta_record("r_N", "_r"),
        "J_min": get_meta_record("J_min", "_J_min"),
        "c_J": get_meta_record("c_J", "_c_J"),
        "kappa_c": get_meta_record("kappa_c", "_kappa_c"),
        "c_xi_calibration_draws": get_meta_record("c_xi_calibration_draws", "_c_xi_calibration_draws"),
        "dgp_parameters": dgp_params,
        "se_type": get_meta_record("se_type"),
        "kernel": get_meta_record("kernel"),
        "distance_metric": get_meta_record("distance_metric"),
        "h_N_formula": get_meta_record("h_N_formula"),
        "h_N_realized": get_meta_record("h_N_realized", "_h_N"),
    })


def _assert_resume_compatible(path: Path, sidecar: Path, requested: Dict) -> None:
    existing = _existing_signature(path, sidecar)
    if existing is None:
        return
    mismatches = []
    for key, requested_value in requested.items():
        existing_value = existing.get(key)
        if existing_value != requested_value:
            mismatches.append((key, existing_value, requested_value))
    if mismatches:
        lines = [
            f"Refusing to append to {path}: existing output has a different substantive run signature.",
            "Allowed resume changes are operational only: R_total, batch_size, n_jobs, progress_every, and verbosity.",
        ]
        for key, old, new in mismatches:
            lines.append(f"  {key}: existing={old!r} requested={new!r}")
        raise SystemExit("\n".join(lines))


def _chunks(items: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _run_one_replication(T: int, N: int, rep: int, tuning: Tuning,
                         dgp_kwargs: Dict, master: int,
                         profile_timing: bool = False,
                         oracle: bool = False,
                         target_names: Optional[Sequence[str]] = None) -> Dict:
    return run_replication(T, N, rep, tuning, dgp_kwargs=dgp_kwargs,
                           master=master, profile_timing=profile_timing,
                           oracle=oracle, target_names=target_names)


def _format_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _emit(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _mode_label(args) -> str:
    base = "rank-selection" if args.select else "fixed-rank"
    return f"oracle {base}" if getattr(args, "oracle", False) else base


def _progress_message(args, *, completed_total: int, current_rep,
                      elapsed: float, avg_seconds: float, remaining: int,
                      rec=None, prefix: str = "progress") -> str:
    eta = _format_eta(avg_seconds * remaining) if completed_total else "unknown"
    jval = rec.get("_J_realized", rec.get("_J")) if rec is not None else None
    retained = rec.get("_retained_nonvalidation") if rec is not None else None
    jtxt = f" J={jval}" if jval is not None else ""
    rtxt = f" retained_nonvalidation={retained:.4f}" if retained is not None else ""
    return (
        f"[mc-batch] {prefix} dgp={args.dgp_type} T={args.T} N={args.N} "
        f"mode={_mode_label(args)} completed={completed_total}/{args.R_total} "
        f"current_rep={current_rep} elapsed={_format_eta(elapsed)} "
        f"avg_sec_per_rep={avg_seconds:.1f} eta={eta}{jtxt}{rtxt}"
    )


def _sidecar_path(out_path: Path) -> Path:
    suffix = "".join(out_path.suffixes)
    if suffix:
        stem = str(out_path)[:-len(suffix)]
        return Path(stem + ".meta.json")
    return Path(str(out_path) + ".meta.json")


def _timing_summary(path: Path, desired: Sequence[int]) -> Dict:
    desired_set = set(int(x) for x in desired)
    rows = []
    if not path.exists():
        return {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if int(rec.get("rep", -1)) in desired_set and "_time_total_sec" in rec:
                rows.append(rec)
    if not rows:
        return {}
    totals = sorted(float(r["_time_total_sec"]) for r in rows)
    n = len(totals)

    def percentile(sorted_vals, p):
        if not sorted_vals:
            return None
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        pos = (len(sorted_vals) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = pos - lo
        return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)

    stage_keys = sorted({
        key for row in rows for key in row
        if key.startswith("_time_") and key.endswith("_sec")
    })
    stage_means = {}
    for key in stage_keys:
        vals = [float(row[key]) for row in rows
                if key in row and isinstance(row[key], (int, float))]
        vals = [v for v in vals if v == v]
        if vals:
            stage_means[key] = float(sum(vals) / len(vals))
    slowest = max(rows, key=lambda row: float(row.get("_time_total_sec", 0.0)))
    return {
        "timing_completed_R": int(n),
        "timing_mean_sec_per_rep": float(sum(totals) / n),
        "timing_median_sec_per_rep": percentile(totals, 0.5),
        "timing_p90_sec_per_rep": percentile(totals, 0.9),
        "timing_max_sec_per_rep": float(max(totals)),
        "timing_mean_by_stage": stage_means,
        "timing_slowest_rep_id": int(slowest["rep"]),
    }


def _jsonl_metadata_summary(path: Path, desired: Sequence[int]) -> Dict:
    """Recover fold-rule and tuning metadata from completed replication records."""
    desired_set = set(int(x) for x in desired)
    rows = []
    if not path.exists():
        return {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if int(rec.get("rep", -1)) in desired_set:
                rows.append(rec)
    if not rows:
        return {}

    def first_present(*keys):
        for row in rows:
            for key in keys:
                if key in row and row[key] is not None:
                    return row[key]
        return None

    def mean_present(*keys):
        vals = []
        for row in rows:
            for key in keys:
                if key in row and row[key] is not None:
                    try:
                        vals.append(float(row[key]))
                    except (TypeError, ValueError):
                        pass
                    break
        return float(sum(vals) / len(vals)) if vals else None

    def values_present(*keys):
        vals = []
        for row in rows:
            for key in keys:
                if key in row and row[key] is not None:
                    try:
                        val = float(row[key])
                    except (TypeError, ValueError):
                        break
                    if math.isfinite(val):
                        vals.append(val)
                    break
        return vals

    def sd_present(*keys):
        vals = values_present(*keys)
        if len(vals) < 2:
            return None
        mu = sum(vals) / len(vals)
        return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

    def min_present(*keys):
        vals = values_present(*keys)
        return min(vals) if vals else None

    def max_present(*keys):
        vals = values_present(*keys)
        return max(vals) if vals else None

    q = first_present("_q")
    r = first_present("_r")
    B = first_present("_B_TN_fold_rule", "_B_NT_fold_rule")
    L = first_present("_L_TN_J", "_L_NT_J")
    J_rule = first_present("_J_rule_term")
    J_realized = first_present("_J_realized", "_J")
    out = {
        "mc_schema_version_from_records": first_present("_mc_schema_version"),
        "q_T": int(q) if q is not None else None,
        "q": int(q) if q is not None else None,
        "r_N": int(r) if r is not None else None,
        "buffer_r": int(r) if r is not None else None,
        "B_NT_fold_rule": float(B) if B is not None else None,
        "B_TN_fold_rule": float(B) if B is not None else None,
        "L_NT_J": float(L) if L is not None else None,
        "L_TN_J": float(L) if L is not None else None,
        "J_rule_term": int(J_rule) if J_rule is not None else None,
        "J_rule": int(J_rule) if J_rule is not None else None,
        "J_realized": int(J_realized) if J_realized is not None else None,
        "realized_J": int(J_realized) if J_realized is not None else None,
        "retained_nonvalidation": mean_present("_retained_nonvalidation"),
        "retained_total": mean_present("_retained_total"),
        "retained_nonvalidation_min": mean_present("_retained_nonvalidation_min"),
        "retained_nonvalidation_max": mean_present("_retained_nonvalidation_max"),
        "validation_fold_size_mean": mean_present("_validation_fold_size_mean"),
        "validation_fold_share_mean": mean_present("_validation_fold_share_mean"),
        "spatial_bandwidth": first_present("_spatial_bandwidth"),
        "se_xs_type": first_present("_se_xs_type"),
        "tuning_kappa_c_from_records": first_present("_kappa_c"),
        "true_ranks": first_present("_true_ranks"),
        "true_ranks_from_records": first_present("_true_ranks"),
        "selected_ranks_first_record": first_present("_selected_ranks", "_ranks"),
        "rho_g": mean_present("_rho_g"),
        "rho_x": mean_present("_rho_x"),
        "rho_fx": mean_present("_rho_fx"),
        "rho_s": mean_present("_rho_s"),
        "delta_x": mean_present("_delta_x"),
        "eta_x": mean_present("_eta_x"),
        "pi_h": mean_present("_pi_h"),
        "c_h": mean_present("_c_h"),
        "c_xi_from_records": mean_present("_c_xi"),
        "PR2_target": mean_present("_PR2_target"),
        "PR2_realized_mean": mean_present("_PR2_realized"),
        "PR2_realized_sd": sd_present("_PR2_realized"),
        "PR2_realized_available": bool(values_present("_PR2_realized")),
        "max_abs_a_it_mean": mean_present("_max_abs_a_it"),
        "max_abs_a_it_max": max_present("_max_abs_a_it"),
        "share_abs_a_ge_1_mean": mean_present("_share_abs_a_ge_1"),
        "companion_radius_mean_mean": mean_present("_companion_radius_mean"),
        "companion_radius_sd_mean": mean_present("_companion_radius_std"),
        "companion_radius_min_min": min_present("_companion_radius_min"),
        "companion_radius_max_max": max_present("_companion_radius_max"),
        "coefficient_summaries_available": bool(values_present("_a_it_mean", "_beta_it_mean")),
        "a_it_mean_mean": mean_present("_a_it_mean"),
        "a_it_sd_mean": mean_present("_a_it_std"),
        "a_it_min_min": min_present("_a_it_min"),
        "a_it_max_max": max_present("_a_it_max"),
        "beta_it_mean_mean": mean_present("_beta_it_mean"),
        "beta_it_sd_mean": mean_present("_beta_it_std"),
        "beta_it_min_min": min_present("_beta_it_min"),
        "beta_it_max_max": max_present("_beta_it_max"),
    }
    return {key: value for key, value in out.items() if value is not None}


def _standard_error_metadata(dgp_type: str, N: int) -> Dict:
    if str(dgp_type).lower() == "dgp1":
        return {
            "se_type": "diagonal",
            "kernel": None,
            "distance_metric": None,
            "h_N_formula": None,
            "h_N_realized": None,
        }
    return {
        "se_type": "spatial-kernel",
        "kernel": "Bartlett",
        "distance_metric": "lattice |i-j|",
        "h_N_formula": "floor(N^(1/3))",
        "h_N_realized": int(math.floor(int(N) ** (1.0 / 3.0))),
    }


def _target_metadata(T: int, N: int) -> Dict:
    return {
        "target_cell_i": int(N) // 2 + 1,
        "target_cell_t": int(T) // 2 + 1,
        "group_1_definition": "first half of units: {1,...,floor(N/2)}",
        "group_0_definition": "second half of units: {floor(N/2)+1,...,N}",
        "contrast_definition": "group_1 mean minus group_0 mean",
    }


def _canonical_dgp_metadata(dgp_type: str, c_xi_calibration_draws: int) -> Dict:
    is_dgp3 = str(dgp_type).lower() == "dgp3"
    is_spatial = str(dgp_type).lower() in {"dgp2", "dgp3"}
    return {
        "rho_g": 0.5,
        "rho_x": 0.5,
        "rho_fx": 0.5,
        "rho_s": 0.5 if is_spatial else None,
        "delta_x": 0.5,
        "eta_x": 0.3 if is_dgp3 else None,
        "pi_h": 0.3,
        "c_h": math.sqrt(0.3 / 0.7),
        "PR2_target": 0.5,
        "c_xi_calibration_draws": int(c_xi_calibration_draws),
    }


def _write_sidecar(path: Path, *, args, tuning: Tuning, completed: int,
                   duplicate_reps: int, started_at: str,
                   timing_summary: Optional[Dict] = None) -> None:
    desired = list(range(int(args.start_rep), int(args.R_total)))
    record_summary = _jsonl_metadata_summary(args.out_path, desired)
    meta = {
        "mc_schema_version": MC_SCHEMA_VERSION,
        "dgp_type": args.dgp_type,
        "T": int(args.T),
        "N": int(args.N),
        "master_seed": int(getattr(args, "master_seed", 2024)),
        "R_total": int(args.R_total),
        "completed_R": int(completed),
        "J_min": int(tuning.J_min),
        "c_J": float(tuning.c_J),
        "q_T": int(tuning.q) if tuning.q is not None else None,
        "q": int(tuning.q) if tuning.q is not None else None,
        "r_N": int(tuning.buffer_r),
        "buffer_r": int(tuning.buffer_r),
        "kappa_c": float(tuning.kappa_c),
        "c_xi_calibration_draws": int(args.c_xi_calibration_draws),
        "c_xi_parent_precompute_sec": float(getattr(args, "c_xi_parent_precompute_sec", 0.0)),
        "c_xi_precomputed_in_parent": bool(getattr(args, "c_xi_precomputed_in_parent", False)),
        "c_xi": getattr(args, "c_xi", None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "command_line_args": _jsonable(getattr(args, "command_line_args", None)),
        "command_line": getattr(args, "command_line", None),
        "parsed_args": _jsonable(getattr(args, "parsed_args_snapshot", None)),
        "config": args.config,
        "config_path": args.config,
        "config_snapshot": _jsonable(getattr(args, "config_snapshot", None)),
        "resolved_tuning": _jsonable(getattr(args, "resolved_tuning", None)),
        "resolved_dgp_kwargs": _jsonable(getattr(args, "resolved_dgp_kwargs", None)),
        "out_path": str(args.out_path),
        "batch_size": int(args.batch_size),
        "n_jobs": int(args.n_jobs),
        "progress_every": int(args.progress_every),
        "quiet": bool(args.quiet),
        "profile_timing": bool(args.profile_timing),
        "oracle": bool(args.oracle),
        "target_filter": list(args.targets) if args.targets is not None else None,
        "start_rep": int(args.start_rep),
        "select": bool(args.select),
        "fixed_ranks": list(args.fixed_ranks) if args.fixed_ranks is not None else None,
        "true_ranks": list(args.true_ranks) if args.true_ranks is not None else None,
        "rank_caps": list(args.rank_caps) if args.rank_caps is not None else None,
        "duplicate_reps": int(duplicate_reps),
    }
    meta.update(_jsonable(getattr(args, "runtime_metadata", None)) or _runtime_metadata())
    meta.update(_jsonable(getattr(args, "git_metadata", None)) or _git_metadata())
    meta.update(_standard_error_metadata(args.dgp_type, args.N))
    meta.update(_target_metadata(args.T, args.N))
    meta.update(_canonical_dgp_metadata(args.dgp_type, args.c_xi_calibration_draws))
    meta.update(record_summary)
    if meta.get("c_xi") is None and meta.get("c_xi_from_records") is not None:
        meta["c_xi"] = meta["c_xi_from_records"]
    if timing_summary:
        meta.update(timing_summary)
    with path.open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume-safe Monte Carlo batch runner")
    ap.add_argument("--dgp-type", required=True, choices=["dgp1", "dgp2", "dgp3"])
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--N", required=True, type=int)
    ap.add_argument("--R-total", required=True, type=int)
    ap.add_argument("--batch-size", required=True, type=int)
    ap.add_argument("--start-rep", type=int, default=0)
    ap.add_argument("--out-path", required=True, type=Path)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--select", required=True, type=_parse_bool)
    ap.add_argument("--fixed-ranks", type=_parse_ranks, default=None)
    ap.add_argument("--true-ranks", type=_parse_ranks, default=None,
                    help="true DGP ranks used for rank-frequency diagnostics; does not fix estimator ranks")
    ap.add_argument("--rank-caps", type=_parse_ranks, default=None,
                    help="candidate rank caps for --select true; defaults to config r_bar or 1,1,1")
    ap.add_argument("--c-xi-calibration-draws", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel worker processes for replications within each batch")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="print progress after this many completed replications")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress output; JSONL and metadata are still written")
    ap.add_argument("--profile-timing", action="store_true",
                    help="record lightweight per-replication timing diagnostics")
    ap.add_argument("--oracle", type=_parse_bool, default=False,
                    help="use oracle true tangent spaces in the Riesz solve")
    ap.add_argument("--targets", type=_parse_targets, default=None,
                    help="comma-separated subset of standard targets to compute, e.g. lag_fmean")
    ap.add_argument("--buffer-r", type=_parse_nonnegative_int, default=None,
                    help="override spatial cross-fitting buffer radius; default uses config")
    args = ap.parse_args()
    args.command_line_args = list(sys.argv)
    args.command_line = " ".join(sys.argv)
    args.runtime_metadata = _runtime_metadata()
    args.git_metadata = _git_metadata()

    if args.R_total < 1:
        raise SystemExit("--R-total must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.start_rep < 0 or args.start_rep >= args.R_total:
        raise SystemExit("--start-rep must satisfy 0 <= start_rep < R_total")
    if args.c_xi_calibration_draws < 1:
        raise SystemExit("--c-xi-calibration-draws must be positive")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must be nonzero; use 1 for serial or -1 for all available cores")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    cfg = _load_config(args.config)
    try:
        tuning = _tuning_from_config(cfg, select=args.select,
                                     fixed_ranks=args.fixed_ranks,
                                     rank_caps=args.rank_caps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.buffer_r is not None:
        tuning = dataclasses.replace(tuning, buffer_r=int(args.buffer_r))
    dgp_kwargs = dict(cfg.get("dgp", {}))
    dgp_kwargs.update({
        "dgp_type": args.dgp_type,
        "c_xi_calibration_draws": int(args.c_xi_calibration_draws),
    })
    if args.true_ranks is not None:
        dgp_kwargs["true_ranks"] = tuple(args.true_ranks)
    args.config_snapshot = _jsonable(cfg)
    args.resolved_tuning = _jsonable(tuning)
    args.resolved_dgp_kwargs = _jsonable(dgp_kwargs)
    args.parsed_args_snapshot = _jsonable({
        key: value for key, value in vars(args).items()
        if key not in {
            "config_snapshot",
            "resolved_tuning",
            "resolved_dgp_kwargs",
            "parsed_args_snapshot",
        }
    })

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(args.out_path)
    master = int(cfg.get("master_seed", 2024))
    args.master_seed = int(master)
    desired = list(range(int(args.start_rep), int(args.R_total)))
    done = _done_reps(args.out_path)
    todo = [rep for rep in desired if rep not in done]
    started_at = datetime.now(timezone.utc).isoformat()
    duplicate_reps = _count_duplicate_reps(args.out_path)
    already_completed = len(done.intersection(desired))
    run_complete = already_completed >= len(desired)

    _emit(f"[mc-batch] dgp={args.dgp_type} T={args.T} N={args.N} "
          f"R_total={args.R_total} mode={_mode_label(args)}", quiet=args.quiet)
    _emit(f"[mc-batch] out={args.out_path}", quiet=args.quiet)
    _emit(f"[mc-batch] already_completed={already_completed} remaining={len(todo)} "
          f"duplicates={duplicate_reps} complete={run_complete}", quiet=args.quiet)

    args.c_xi_parent_precompute_sec = 0.0
    args.c_xi_precomputed_in_parent = False
    args.c_xi = None
    if todo:
        cxi_t0 = time.perf_counter()
        dgp_kwargs = precompute_dgp_calibration(args.T, args.N, dgp_kwargs)
        args.resolved_dgp_kwargs = _jsonable(dgp_kwargs)
        args.c_xi_parent_precompute_sec = float(time.perf_counter() - cxi_t0)
        info = dgp_kwargs.get("c_xi_info")
        if info is not None:
            args.c_xi_precomputed_in_parent = True
            args.c_xi = float(info["c_xi"])
            _emit(f"[mc-batch] parent c_xi calibration dgp={args.dgp_type} "
                  f"T={args.T} N={args.N} c_xi={args.c_xi:.12g} "
                  f"elapsed={_format_eta(args.c_xi_parent_precompute_sec)}",
                  quiet=args.quiet)

    if todo:
        _assert_resume_compatible(
            args.out_path,
            sidecar,
            _requested_signature(args, tuning, dgp_kwargs, master),
        )

    completed_now = 0
    t0 = time.time()
    with args.out_path.open("a", buffering=1) as fh:
        for batch_no, batch in enumerate(_chunks(todo, int(args.batch_size)), start=1):
            completed_at_batch_start = len(done.intersection(desired))
            workers = int(args.n_jobs)
            _emit(f"[mc-batch] batch {batch_no}: reps {batch[0]}..{batch[-1]} "
                  f"workers={workers} completed={completed_at_batch_start}/{args.R_total}",
                  quiet=args.quiet)
            batch_start = time.time()
            last_rec = None
            if int(args.n_jobs) == 1:
                recs = []
                for rep in batch:
                    _emit(f"[mc-batch] starting dgp={args.dgp_type} T={args.T} N={args.N} "
                          f"mode={_mode_label(args)} rep={rep}",
                          quiet=args.quiet)
                    recs.append(_run_one_replication(args.T, args.N, rep, tuning,
                                                     dgp_kwargs, master,
                                                     args.profile_timing,
                                                     args.oracle,
                                                     args.targets))
            else:
                from joblib import Parallel, delayed
                recs = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
                    delayed(_run_one_replication)(args.T, args.N, rep, tuning,
                                                  dgp_kwargs, master,
                                                  args.profile_timing,
                                                  args.oracle,
                                                  args.targets)
                    for rep in batch
                )

            for rec in sorted(recs, key=lambda row: int(row["rep"])):
                rep = int(rec["rep"])
                if args.profile_timing:
                    tw = time.perf_counter()
                    rec["_time_json_write_sec"] = 0.0
                    json.dumps(rec)
                    rec["_time_json_write_sec"] = float(time.perf_counter() - tw)
                    payload = json.dumps(rec) + "\n"
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                else:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                done.add(rep)
                completed_now += 1
                last_rec = rec
                elapsed = time.time() - t0
                avg = elapsed / max(completed_now, 1)
                remaining_total = max(int(args.R_total) - len(done.intersection(desired)), 0)
                completed_total = len(done.intersection(desired))
                if completed_now % int(args.progress_every) == 0:
                    _emit(_progress_message(args, completed_total=completed_total,
                                            current_rep=rep, elapsed=elapsed,
                                            avg_seconds=avg,
                                            remaining=remaining_total, rec=rec),
                          quiet=args.quiet)
            duplicate_reps = _count_duplicate_reps(args.out_path)
            completed_total = len(_done_reps(args.out_path).intersection(desired))
            timing_summary = (_timing_summary(args.out_path, desired)
                              if args.profile_timing else None)
            _write_sidecar(sidecar, args=args, tuning=tuning,
                           completed=completed_total,
                           duplicate_reps=duplicate_reps,
                           started_at=started_at,
                           timing_summary=timing_summary)
            elapsed = time.time() - t0
            avg = elapsed / max(completed_now, 1)
            remaining_total = max(int(args.R_total) - completed_total, 0)
            batch_elapsed = time.time() - batch_start
            batch_avg = batch_elapsed / max(len(batch), 1)
            batch_eta = _format_eta(avg * remaining_total) if completed_now else "unknown"
            _emit(f"[mc-batch] batch {batch_no} summary elapsed={_format_eta(batch_elapsed)} "
                  f"avg_sec_per_rep={batch_avg:.1f} eta={batch_eta}",
                  quiet=args.quiet)
            _emit(_progress_message(args, completed_total=completed_total,
                                    current_rep=batch[-1], elapsed=elapsed,
                                    avg_seconds=avg,
                                    remaining=remaining_total, rec=last_rec,
                                    prefix="batch-complete"),
                  quiet=args.quiet)
            _emit(f"[mc-batch] sidecar updated: {sidecar}", quiet=args.quiet)

    duplicate_reps = _count_duplicate_reps(args.out_path)
    completed_total = len(_done_reps(args.out_path).intersection(desired))
    _write_sidecar(sidecar, args=args, tuning=tuning,
                   completed=completed_total,
                   duplicate_reps=duplicate_reps,
                   started_at=started_at,
                   timing_summary=(_timing_summary(args.out_path, desired)
                                   if args.profile_timing else None))
    elapsed = time.time() - t0
    avg = elapsed / max(completed_now, 1) if completed_now else 0.0
    _emit(f"[mc-batch] done completed_R={completed_total}/{args.R_total} "
          f"elapsed={_format_eta(elapsed)} avg_sec_per_rep={avg:.1f} "
          f"duplicates={duplicate_reps} jsonl={args.out_path} metadata={sidecar}",
          quiet=args.quiet)


if __name__ == "__main__":
    main()
