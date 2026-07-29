#!/usr/bin/env python3
"""Prepare and smoke-test the all-homes housing empirical estimator input.

This script consumes the completed housing data-audit candidate panels.  It
does not download data, run X-13, interpolate, impute, winsorize, standardize,
forecast, backcast, or modify the legacy Zillow top/bottom-tier runner.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from dlrhcs.design import build_blocks  # noqa: E402
from dlrhcs.paths import find_repo_root, repo_relative, resolve_repo_path  # noqa: E402
from dlrhcs.pipeline import Tuning, estimate  # noqa: E402
from dlrhcs.targets import Target  # noqa: E402

SCHEMA_VERSION = "housing_all_homes_v2"
BASELINE_REL = Path("data") / "zillow" / "processed" / "estimation_panels" / "housing_baseline_2010_final"
FINAL_CANDIDATE_REL = Path("data") / "zillow" / "processed" / "candidate_panels_final_only"
OUTPUT_REL = Path("outputs") / "empirical" / "housing_all_homes"
CONFIG_REL = Path("configs") / "full.json"
PRIMARY_COLUMNS = ["zhvi_all_homes_sa", "permits_units_sa", "employment_thousands_sa"]
PANEL_COLUMNS = [
    "cbsa_code", "msa_title", "date",
    "zhvi_all_homes_sa", "permits_units_sa", "employment_thousands_sa",
    "log_zhvi", "asinh_permits", "log_employment",
    "lag_log_zhvi", "lag_asinh_permits", "lag_log_employment",
    "bls_preliminary_flag",
    "zhvi_source_vintage", "permits_source_vintage", "employment_source_vintage",
]
REQUIRED_PANEL_FILES = [
    "housing_estimation_panel.csv",
    "lag_check.csv",
    "metadata.json",
    "monthly_dates.csv",
    "msa_list.csv",
    "transformation_check.csv",
]
RECORDED_CHECKSUM_FILES = [
    "housing_estimation_panel.csv",
    "lag_check.csv",
    "monthly_dates.csv",
    "msa_list.csv",
    "transformation_check.csv",
]
BASELINE_EXPECTED_N = 169
BASELINE_EXPECTED_T = 197
BASELINE_EXPECTED_USABLE = 33124
RIESZ_RESIDUAL_NUMERICAL_MARGIN = 1e-8
PROGRESS_STATUS_VALUES = {"initializing", "running", "validating", "completed", "failed", "interrupted"}


def month_index(ym: str) -> int:
    y, m = ym[:7].split("-")
    return int(y) * 12 + int(m) - 1


def month_from_index(idx: int) -> str:
    y, m0 = divmod(idx, 12)
    return f"{y:04d}-{m0 + 1:02d}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def write_json(path: Path, obj: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def signature_hash(signature: Mapping[str, object]) -> str:
    payload = json.dumps(jsonable(signature), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(x: object) -> float:
    v = float(str(x).replace(",", ""))
    if not math.isfinite(v):
        raise ValueError(f"nonfinite value {x!r}")
    return v


def parse_ranks(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text in (None, "", "none", "None"):
        return None
    vals = tuple(int(v.strip()) for v in str(text).split(",") if v.strip())
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("ranks must be nonnegative integers")
    return vals


def parse_positive_float(text: str) -> float:
    try:
        val = float(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"expected positive finite float, got {text!r}") from exc
    if not math.isfinite(val) or val <= 0:
        raise argparse.ArgumentTypeError(f"expected positive finite float, got {text!r}")
    return val


def parse_positive_int(text: str) -> int:
    try:
        val = int(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {text!r}") from exc
    if val <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {text!r}")
    return val


def parse_bool(text: str) -> bool:
    if str(text).lower() in ("1", "true", "yes", "y", "on"):
        return True
    if str(text).lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def load_candidate_metadata(root: Path) -> List[Dict[str, object]]:
    out = []
    for p in sorted(root.iterdir() if root.exists() else []):
        meta_path = p / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["path"] = str(p)
        out.append(meta)
    return out


def portable_candidate_summary(candidate: Mapping[str, object], repo_root: Path) -> Dict[str, object]:
    out = dict(candidate)
    path = Path(str(out.get("path", ""))) if out.get("path") else None
    if path is not None:
        out["repo_relative_path"] = repo_relative(path, repo_root)
        out["resolved_absolute_path_info"] = str(path.expanduser().resolve())
        out.pop("path", None)
    return out


def choose_baseline_candidate(candidates: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    eligible = [c for c in candidates if str(c.get("start_date", ""))[:7] == "2010-01"]
    if not eligible:
        raise FileNotFoundError("no final-only candidate beginning in 2010-01 was found")
    return max(eligible, key=lambda c: (
        int(c.get("N", 0)),
        int(c.get("T_months", 0)),
        str(c.get("candidate_id", "")) == "start_2010",
    ))


def robustness_candidates(candidates: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    starts_2004 = [c for c in candidates if str(c.get("start_date", ""))[:7] == "2004-01"]
    starts_2012 = [c for c in candidates if str(c.get("start_date", ""))[:7] == "2012-01"]
    at_least_180 = [c for c in candidates if int(c.get("T_months", 0)) >= 180]
    if starts_2004:
        out["longest_feasible_final_only_panel_beginning_2004"] = max(
            starts_2004, key=lambda c: (int(c.get("T_months", 0)), int(c.get("N", 0)))
        )
    if starts_2012:
        out["largest_final_only_panel_beginning_2012"] = max(
            starts_2012, key=lambda c: (int(c.get("N", 0)), int(c.get("T_months", 0)))
        )
    if at_least_180:
        out["largest_final_only_panel_with_at_least_180_months"] = max(
            at_least_180, key=lambda c: (int(c.get("N", 0)), int(c.get("T_months", 0)))
        )
    return out


def transform_rows(level_rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = []
    seen = set()
    negative_permits = 0
    for raw in sorted(level_rows, key=lambda r: (str(r["cbsa_code"]).zfill(5), str(r["date"]))):
        key = (str(raw["cbsa_code"]).zfill(5), str(raw["date"])[:7])
        if key in seen:
            raise ValueError(f"duplicate CBSA-month row: {key}")
        seen.add(key)
        z = finite_float(raw["zhvi_all_homes_sa"])
        p = finite_float(raw["permits_units_sa"])
        e = finite_float(raw["employment_thousands_sa"])
        if z <= 0:
            raise ValueError(f"ZHVI must be strictly positive at {key}")
        if e <= 0:
            raise ValueError(f"employment must be strictly positive at {key}")
        if p < 0:
            negative_permits += 1
        rows.append({
            "cbsa_code": key[0],
            "msa_title": raw.get("msa_title", ""),
            "date": str(raw["date"])[:10],
            "zhvi_all_homes_sa": z,
            "permits_units_sa": p,
            "employment_thousands_sa": e,
            "log_zhvi": math.log(z),
            "asinh_permits": math.asinh(p),
            "log_employment": math.log(e),
            "lag_log_zhvi": "",
            "lag_asinh_permits": "",
            "lag_log_employment": "",
            "bls_preliminary_flag": str(raw.get("bls_preliminary_flag", "0")),
            "zhvi_source_vintage": raw.get("zhvi_source_vintage", ""),
            "permits_source_vintage": raw.get("permits_source_vintage", ""),
            "employment_source_vintage": raw.get("employment_source_vintage", ""),
        })
    diag = {
        "level_observations": len(rows),
        "negative_permit_count_retained": negative_permits,
        "missing_primary_values": sum(raw.get(c) in ("", None) for raw in level_rows for c in PRIMARY_COLUMNS),
        "duplicate_cbsa_month_rows": len(level_rows) - len(seen),
        "nonpositive_zhvi_count": 0,
        "nonpositive_employment_count": 0,
        "finite_transformed_values": all(math.isfinite(float(r[c])) for r in rows for c in ["log_zhvi", "asinh_permits", "log_employment"]),
    }
    return rows, diag


def add_exact_lags(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_key = {(str(r["cbsa_code"]).zfill(5), str(r["date"])[:7]): r for r in rows}
    out = []
    bridged_missing = 0
    usable = 0
    for r in rows:
        ym = str(r["date"])[:7]
        prev_ym = month_from_index(month_index(ym) - 1)
        prev = by_key.get((str(r["cbsa_code"]).zfill(5), prev_ym))
        row = dict(r)
        if prev is not None:
            row["lag_log_zhvi"] = prev["log_zhvi"]
            row["lag_asinh_permits"] = prev["asinh_permits"]
            row["lag_log_employment"] = prev["log_employment"]
            usable += 1
        elif any((str(r["cbsa_code"]).zfill(5), month_from_index(month_index(ym) - k)) in by_key for k in range(2, 13)):
            bridged_missing += 1
        out.append(row)
    return out, {
        "usable_dynamic_observations": usable,
        "lag_bridging_missing_month_count": bridged_missing,
        "lagged_columns_finite": all(
            math.isfinite(float(r[c]))
            for r in out
            if r["lag_log_zhvi"] != ""
            for c in ["lag_log_zhvi", "lag_asinh_permits", "lag_log_employment"]
        ),
    }


def validate_panel(rows: Sequence[Mapping[str, object]], meta: Mapping[str, object]) -> Dict[str, object]:
    n = int(meta["N"])
    t = int(meta["T_months"])
    codes = sorted({str(r["cbsa_code"]).zfill(5) for r in rows})
    dates = sorted({str(r["date"])[:7] for r in rows})
    by_code: Dict[str, List[str]] = {c: [] for c in codes}
    for r in rows:
        by_code[str(r["cbsa_code"]).zfill(5)].append(str(r["date"])[:7])
    grids_identical = all(sorted(v) == dates for v in by_code.values())
    consecutive = all(
        month_index(dates[i]) == month_index(dates[0]) + i
        for i in range(len(dates))
    )
    usable = sum(r["lag_log_zhvi"] != "" for r in rows)
    prelim = sum(str(r.get("bls_preliminary_flag", "0")) == "1" for r in rows)
    required = {
        "realized_N": len(codes),
        "realized_T": len(dates),
        "expected_N": n,
        "expected_T": t,
        "level_observation_identity_ok": len(rows) == n * t,
        "usable_dynamic_identity_ok": usable == n * (t - 1),
        "usable_dynamic_observations": usable,
        "identical_date_grid_by_msa": grids_identical,
        "consecutive_monthly_dates": consecutive,
        "preliminary_bls_observations": prelim,
        "no_preliminary_bls_observations": prelim == 0,
        "strictly_positive_zhvi": all(float(r["zhvi_all_homes_sa"]) > 0 for r in rows),
        "strictly_positive_employment": all(float(r["employment_thousands_sa"]) > 0 for r in rows),
        "transformed_values_finite": all(
            math.isfinite(float(r[c])) for r in rows for c in ["log_zhvi", "asinh_permits", "log_employment"]
        ),
        "lag_values_equal_preceding_observed_calendar_month": True,
    }
    by_key = {(str(r["cbsa_code"]).zfill(5), str(r["date"])[:7]): r for r in rows}
    for r in rows:
        if r["lag_log_zhvi"] == "":
            continue
        prev = by_key[(str(r["cbsa_code"]).zfill(5), month_from_index(month_index(str(r["date"])[:7]) - 1))]
        if any(abs(float(r[lag]) - float(prev[cur])) > 1e-12 for lag, cur in [
            ("lag_log_zhvi", "log_zhvi"),
            ("lag_asinh_permits", "asinh_permits"),
            ("lag_log_employment", "log_employment"),
        ]):
            required["lag_values_equal_preceding_observed_calendar_month"] = False
            break
    required["validation_passed"] = all([
        required["level_observation_identity_ok"],
        required["usable_dynamic_identity_ok"],
        required["identical_date_grid_by_msa"],
        required["consecutive_monthly_dates"],
        required["no_preliminary_bls_observations"],
        required["strictly_positive_zhvi"],
        required["strictly_positive_employment"],
        required["transformed_values_finite"],
        required["lag_values_equal_preceding_observed_calendar_month"],
    ])
    return required


def prepare_estimation_panel(candidate_root: Path,
                             out_dir: Path,
                             repo_root: Optional[Path] = None) -> Dict[str, object]:
    repo_root = repo_root or find_repo_root(start=__file__)
    candidates = load_candidate_metadata(candidate_root)
    baseline = choose_baseline_candidate(candidates)
    source_dir = Path(str(baseline["path"]))
    level_rows = read_csv(source_dir / "housing_panel_levels.csv")
    transformed, tdiag = transform_rows(level_rows)
    panel_rows, ldiag = add_exact_lags(transformed)
    validation = validate_panel(panel_rows, baseline)
    if not validation["validation_passed"]:
        raise ValueError(f"baseline panel validation failed: {validation}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "housing_estimation_panel.csv", panel_rows, PANEL_COLUMNS)
    write_csv(out_dir / "msa_list.csv", read_csv(source_dir / "msa_list.csv"), ["cbsa_code", "msa_title"])
    write_csv(out_dir / "monthly_dates.csv", read_csv(source_dir / "monthly_dates.csv"), ["date"])
    write_csv(out_dir / "transformation_check.csv", [{
        **tdiag,
        "asinh_permits_defined_for_negative_zero_positive": True,
        "standardized": False,
        "winsorized": False,
        "negative_permit_values_truncated_or_replaced": False,
    }], [
        "level_observations", "negative_permit_count_retained", "missing_primary_values",
        "duplicate_cbsa_month_rows", "nonpositive_zhvi_count", "nonpositive_employment_count",
        "finite_transformed_values", "asinh_permits_defined_for_negative_zero_positive",
        "standardized", "winsorized", "negative_permit_values_truncated_or_replaced",
    ])
    write_csv(out_dir / "lag_check.csv", [{
        **ldiag,
        **validation,
    }], sorted(set(ldiag) | set(validation)))
    checksums = {
        name: sha256_file(out_dir / name)
        for name in ["housing_estimation_panel.csv", "msa_list.csv", "monthly_dates.csv", "transformation_check.csv", "lag_check.csv"]
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": baseline["candidate_id"],
        "candidate_source_repo_relative_path": repo_relative(source_dir, repo_root),
        "candidate_source_resolved_absolute_path_info": str(source_dir.resolve()),
        "repo_relative_path": repo_relative(out_dir, repo_root),
        "resolved_absolute_path_info": str(out_dir.resolve()),
        "start_date": baseline["start_date"],
        "end_date": baseline["end_date"],
        "N": int(baseline["N"]),
        "T": int(baseline["T_months"]),
        "NT": int(baseline["NT"]),
        "usable_dynamic_observations": validation["usable_dynamic_observations"],
        "outcome": "log(zhvi_all_homes_sa)",
        "controls": ["lag_asinh_permits", "lag_log_employment"],
        "permit_transformation": "asinh(permits_units_sa)",
        "employment_transformation": "log(employment_thousands_sa)",
        "negative_permit_count_retained": tdiag["negative_permit_count_retained"],
        "preliminary_bls_observations": validation["preliminary_bls_observations"],
        "validation": validation,
        "transformation_diagnostics": tdiag,
        "lag_diagnostics": ldiag,
        "robustness_candidates_preserved_not_estimated": {
            k: portable_candidate_summary(v, repo_root)
            for k, v in robustness_candidates(candidates).items()
        },
        "checksums": checksums,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_interpolation_or_imputation": True,
        "no_standardization_or_winsorization": True,
        "no_download_or_x13_rerun": True,
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def prepare_baseline_panel(candidate_root: Path,
                           out_dir: Path,
                           repo_root: Optional[Path] = None) -> Dict[str, object]:
    return prepare_estimation_panel(candidate_root, out_dir, repo_root=repo_root)


def load_existing_estimation_panel(panel_dir: Path,
                                   expected_n: Optional[int] = None,
                                   expected_t: Optional[int] = None,
                                   expected_usable: Optional[int] = None) -> Dict[str, object]:
    missing = [name for name in REQUIRED_PANEL_FILES if not (panel_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"prepared housing estimation panel is missing required file(s): {', '.join(missing)}"
        )
    meta = json.loads((panel_dir / "metadata.json").read_text(encoding="utf-8"))
    checksums = meta.get("checksums", {})
    checksum_actual = {name: sha256_file(panel_dir / name) for name in RECORDED_CHECKSUM_FILES}
    checksum_mismatches = {
        name: {"recorded": checksums.get(name), "actual": actual}
        for name, actual in checksum_actual.items()
        if checksums.get(name) != actual
    }
    if checksum_mismatches:
        raise ValueError(f"prepared housing estimation panel checksum mismatch: {checksum_mismatches}")

    rows = read_csv(panel_dir / "housing_estimation_panel.csv")
    validation = validate_panel(rows, {"N": meta["N"], "T_months": meta["T"]})
    errors = []
    if not validation.get("validation_passed"):
        errors.append(f"validation failed: {validation}")
    if expected_n is not None and int(meta.get("N", -1)) != int(expected_n):
        errors.append(f"expected N={expected_n}, found N={meta.get('N')}")
    if expected_t is not None and int(meta.get("T", -1)) != int(expected_t):
        errors.append(f"expected T={expected_t}, found T={meta.get('T')}")
    usable = int(meta.get("usable_dynamic_observations", -1))
    if expected_usable is not None and usable != int(expected_usable):
        errors.append(f"expected usable lagged observations={expected_usable}, found {usable}")
    if int(meta.get("preliminary_bls_observations", -1)) != 0:
        errors.append(f"expected no preliminary BLS observations, found {meta.get('preliminary_bls_observations')}")
    if int(validation.get("preliminary_bls_observations", -1)) != 0:
        errors.append(f"validation found preliminary BLS observations={validation.get('preliminary_bls_observations')}")
    if not validation.get("lag_values_equal_preceding_observed_calendar_month"):
        errors.append("exact monthly lag validation failed")
    if errors:
        raise ValueError("prepared housing estimation panel is invalid: " + "; ".join(errors))
    return {
        "metadata": meta,
        "rows": rows,
        "validation": validation,
        "checksums": checksum_actual,
    }


def load_estimation_panel(panel_dir: Path,
                          first_n_msas: Optional[int] = None,
                          first_t_usable: Optional[int] = None) -> Dict[str, object]:
    loaded = load_existing_estimation_panel(panel_dir)
    meta = loaded["metadata"]
    rows = loaded["rows"]
    codes = [r["cbsa_code"] for r in read_csv(panel_dir / "msa_list.csv")]
    months = [r["date"][:7] for r in read_csv(panel_dir / "monthly_dates.csv")]
    if first_n_msas is not None:
        codes = codes[:int(first_n_msas)]
    if first_t_usable is not None:
        keep_months = months[:int(first_t_usable) + 1]
    else:
        keep_months = months
    code_set, month_set = set(codes), set(keep_months)
    rows = [r for r in rows if r["cbsa_code"] in code_set and r["date"][:7] in month_set]
    by = {(r["date"][:7], r["cbsa_code"]): r for r in rows}
    usable_months = keep_months[1:]
    y = np.array([[finite_float(by[(m, c)]["log_zhvi"]) for c in codes] for m in usable_months])
    lag_y = np.array([[finite_float(by[(m, c)]["lag_log_zhvi"]) for c in codes] for m in usable_months])
    lag_permits = np.array([[finite_float(by[(m, c)]["lag_asinh_permits"]) for c in codes] for m in usable_months])
    lag_employment = np.array([[finite_float(by[(m, c)]["lag_log_employment"]) for c in codes] for m in usable_months])
    return {
        "Y": y,
        "Z": [lag_y, lag_permits, lag_employment],
        "months": usable_months,
        "codes": codes,
        "rows": rows,
        "metadata": meta,
        "panel_checksum": sha256_file(panel_dir / "housing_estimation_panel.csv"),
        "panel_dir": str(panel_dir),
    }


def _target_mean(blocks, block: int, name: str) -> Target:
    w = np.full_like(blocks[block], 1.0 / blocks[block].size, dtype=float)
    dirs = [np.zeros_like(zb) for zb in blocks]
    dirs[block] = w
    return Target(name, block, dirs)


def housing_targets(blocks) -> List[Target]:
    return [
        _target_mean(blocks, 0, "lag_log_zhvi_mean"),
        _target_mean(blocks, 1, "asinh_permits_mean"),
        _target_mean(blocks, 2, "log_employment_mean"),
    ]


def tuning_from_config(config_path: Path, *, fixed_ranks: Optional[Tuple[int, ...]],
                       select: bool, n_jobs: int,
                       riesz_maxiter: Optional[int] = None,
                       riesz_tol: Optional[float] = None,
                       riesz_ridge: Optional[float] = None,
                       riesz_use_cached_scale: Optional[bool] = None) -> Tuning:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ecfg = cfg.get("empirical", {})
    ranks = fixed_ranks
    if select:
        ranks = None
    elif ranks is None:
        ranks = (1, 1, 1, 2)
    r_bar = tuple(ecfg.get("r_bar", [2, 2, 2, 4]))
    if len(r_bar) != 4:
        r_bar = (2, 2, 2, 4)
    tuning = Tuning(
        ranks=ranks,
        q=int(ecfg.get("q", 1)),
        J_min=int(ecfg.get("J_min", 10)),
        c_J=float(ecfg.get("c_J", 1.0)),
        ridge=float(ecfg.get("ridge", 0.1)),
        n_restarts=int(ecfg.get("n_restarts", 3)),
        n_sweeps=int(ecfg.get("n_sweeps", 60)),
        riesz_tol=float(ecfg.get("riesz_tol", 1e-5)),
        riesz_ridge=float(ecfg.get("riesz_ridge", 1e-6)),
        riesz_maxiter=int(ecfg.get("riesz_maxiter", 600)),
        select=bool(select),
        r_bar=r_bar,
        kappa_c=float(ecfg.get("kappa_c", 0.03)),
        xs_kernel=str(ecfg.get("xs_kernel", "cluster")),
        xs_bandwidth=ecfg.get("xs_bandwidth"),
        n_jobs=int(n_jobs),
    )
    if riesz_maxiter is not None:
        tuning = dataclasses.replace(tuning, riesz_maxiter=int(riesz_maxiter))
    if riesz_tol is not None:
        tuning = dataclasses.replace(tuning, riesz_tol=float(riesz_tol))
    if riesz_ridge is not None:
        tuning = dataclasses.replace(tuning, riesz_ridge=float(riesz_ridge))
    if riesz_use_cached_scale is not None:
        tuning = dataclasses.replace(tuning, riesz_use_cached_scale=bool(riesz_use_cached_scale))
    return tuning


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def package_checks() -> Dict[str, object]:
    packages = ["numpy", "scipy"]
    return {name: importlib.util.find_spec(name) is not None for name in packages}


def output_dirs_writable(output_root: Path) -> Dict[str, bool]:
    checks = {}
    for sub in ["", "tables", "figures", "runtime", "smoke", "production"]:
        d = output_root / sub if sub else output_root
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            checks[repo_relative(d, output_root)] = True
        except Exception:
            checks[repo_relative(d, output_root)] = False
    return checks


def summarize_riesz_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    rel = [float(r["achieved_relative_residual"]) for r in rows if r.get("achieved_relative_residual") not in ("", None)]
    sol = [float(r["solution_norm"]) for r in rows if r.get("solution_norm") not in ("", None)]
    iters = [int(r["iterations"]) for r in rows if r.get("iterations") not in ("", None)]
    conv = [bool(r["converged"]) for r in rows]
    maxiter_hits = [int(r["iterations"]) >= int(r["maxiter"]) for r in rows]
    nonfinite = [bool(r.get("contains_nonfinite", False)) for r in rows]
    return {
        "number_of_solves": len(rows),
        "convergence_fraction": float(np.mean(conv)) if conv else "",
        "iterations_mean": float(np.mean(iters)) if iters else "",
        "iterations_median": float(np.median(iters)) if iters else "",
        "iterations_max": int(max(iters)) if iters else "",
        "relative_residual_median": float(np.median(rel)) if rel else "",
        "relative_residual_max": float(max(rel)) if rel else "",
        "solution_norm_median": float(np.median(sol)) if sol else "",
        "solution_norm_max": float(max(sol)) if sol else "",
        "number_reaching_maxiter": int(sum(maxiter_hits)),
        "number_containing_nonfinite_values": int(sum(nonfinite)),
    }


def flatten_riesz_diagnostics(riesz_diag: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for target_name, diag in sorted(riesz_diag.items()):
        for row in diag.get("folds", []):
            out = dict(row)
            out["target_name"] = target_name
            rows.append(out)
    return rows


class ProductionProgress:
    def __init__(self, out_dir: Path, run_signature_hash: str = "pending",
                 progress_every: float = 30.0, overwrite_log: bool = False):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "production_progress.json"
        self.log_path = out_dir / "production.log"
        self.run_signature_hash = str(run_signature_hash)
        self.progress_every = max(float(progress_every), 0.0)
        self.started_monotonic = time.perf_counter()
        self.started_utc = utc_now()
        self.last_emit = 0.0
        if overwrite_log and self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

    def set_signature_hash(self, run_signature_hash: str) -> None:
        self.run_signature_hash = str(run_signature_hash)

    def emit(self, phase: str, message: str, *, status: str = "running",
             completed_units: Optional[int] = None, total_units: Optional[int] = None,
             current_target: Optional[str] = None, current_fold: Optional[int] = None,
             last_completed_checkpoint: Optional[str] = None, force: bool = True) -> None:
        if status not in PROGRESS_STATUS_VALUES:
            raise ValueError(f"unknown production progress status: {status}")
        now = time.perf_counter()
        if not force and (now - self.last_emit) < self.progress_every:
            return
        self.last_emit = now
        elapsed = now - self.started_monotonic
        updated = utc_now()
        line = (
            f"[{updated}] [housing-all-homes] [{self.run_signature_hash[:12]}] "
            f"{status} phase={phase} elapsed={elapsed:.1f}s {message}"
        )
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(line + "\n")
        write_json(self.path, {
            "schema_version": SCHEMA_VERSION,
            "run_signature_hash": self.run_signature_hash,
            "status": status,
            "phase": phase,
            "completed_units": completed_units,
            "total_units": total_units,
            "current_target": current_target,
            "current_fold": current_fold,
            "started_utc": self.started_utc,
            "updated_utc": updated,
            "elapsed_seconds": float(elapsed),
            "last_completed_checkpoint": last_completed_checkpoint,
            "message": message,
        })

    def heartbeat(self, phase: str, message: str, **kwargs) -> None:
        self.emit(phase, message, force=False, **kwargs)


def checkpoint_dir_for(out_dir: Path) -> Path:
    return out_dir / "checkpoints"


def checkpoint_manifest_path(out_dir: Path) -> Path:
    return checkpoint_dir_for(out_dir) / "manifest.json"


def estimator_checkpoint_path(out_dir: Path) -> Path:
    return checkpoint_dir_for(out_dir) / "estimator_output.json"


def write_checkpoint_manifest(out_dir: Path, run_hash: str, signature: Mapping[str, object],
                              status: str, last_completed_checkpoint: str,
                              message: str = "") -> None:
    checkpoint_dir_for(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_manifest_path(out_dir), {
        "schema_version": SCHEMA_VERSION,
        "run_signature_hash": run_hash,
        "run_signature": jsonable(signature),
        "status": status,
        "last_completed_checkpoint": last_completed_checkpoint,
        "checkpoint_file": "estimator_output.json",
        "updated_utc": utc_now(),
        "message": message,
    })


def load_compatible_estimator_checkpoint(out_dir: Path, run_hash: str,
                                         signature: Mapping[str, object],
                                         *, overwrite: bool = False,
                                         progress: Optional[ProductionProgress] = None) -> Optional[Dict[str, object]]:
    manifest_path = checkpoint_manifest_path(out_dir)
    ckpt_path = estimator_checkpoint_path(out_dir)
    if not manifest_path.exists() and not ckpt_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if progress:
            progress.emit("checkpoint", f"ignoring corrupt checkpoint manifest: {exc}", status="running")
        return None
    existing_hash = manifest.get("run_signature_hash")
    if existing_hash != run_hash or manifest.get("run_signature") != jsonable(signature):
        if overwrite:
            if progress:
                progress.emit("checkpoint", "overwriting incompatible checkpoint because --overwrite was supplied", status="running")
            return None
        raise SystemExit("refusing to reuse incompatible housing production checkpoint")
    try:
        checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if progress:
            progress.emit("checkpoint", f"ignoring corrupt estimator checkpoint: {exc}", status="running")
        return None
    if checkpoint.get("run_signature_hash") != run_hash or checkpoint.get("run_signature") != jsonable(signature):
        if overwrite:
            return None
        raise SystemExit("refusing to reuse estimator checkpoint with incompatible signature")
    output = checkpoint.get("output")
    if not isinstance(output, dict):
        if progress:
            progress.emit("checkpoint", "ignoring incomplete estimator checkpoint without output", status="running")
        return None
    return output


def validate_production_output(output: Mapping[str, object],
                               targets: Sequence[Target],
                               tuning: Tuning) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    target_names = [t.name for t in targets]
    if len(set(target_names)) != len(target_names):
        failures.append("requested target names are duplicated")
    target_table = output.get("targets", {})
    missing_targets = [name for name in target_names if name not in target_table]
    if missing_targets:
        failures.append(f"requested target(s) absent from output: {missing_targets}")
    for name in target_names:
        vals = target_table.get(name, {}) if isinstance(target_table, Mapping) else {}
        for field in ["estimate", "se_white", "se_xs"]:
            try:
                value = float(vals[field])
            except Exception:
                failures.append(f"{name}.{field} is missing or nonnumeric")
                continue
            if not math.isfinite(value):
                failures.append(f"{name}.{field} is nonfinite")

    riesz_summary = output.get("riesz_diagnostics", {})
    try:
        conv_frac = float(riesz_summary.get("convergence_fraction", "nan"))
    except Exception:
        conv_frac = float("nan")
    if not math.isfinite(conv_frac) or conv_frac < 1.0:
        failures.append(f"Riesz convergence fraction is below 1.0: {conv_frac}")
    if int(riesz_summary.get("number_containing_nonfinite_values", 0) or 0) > 0:
        failures.append("at least one Riesz solution contains nonfinite values")
    if int(riesz_summary.get("number_reaching_maxiter", 0) or 0) > 0:
        failures.append("at least one Riesz solve reached maxiter")
    try:
        max_rel = float(riesz_summary.get("relative_residual_max", "nan"))
    except Exception:
        max_rel = float("nan")
    residual_limit = float(tuning.riesz_tol) * (1.0 + RIESZ_RESIDUAL_NUMERICAL_MARGIN)
    if not math.isfinite(max_rel) or max_rel > residual_limit:
        failures.append(
            f"maximum Riesz relative residual {max_rel} exceeds tolerance "
            f"{tuning.riesz_tol} with margin {RIESZ_RESIDUAL_NUMERICAL_MARGIN}"
        )
    for row in output.get("riesz_fold_diagnostics", []) or []:
        if bool(row.get("contains_nonfinite", False)):
            failures.append(f"Riesz fold contains nonfinite values: {row.get('target_name')} fold {row.get('fold_id')}")
        if int(row.get("iterations", 0) or 0) >= int(row.get("maxiter", 0) or 0) and not bool(row.get("converged", False)):
            failures.append(f"Riesz fold reached maxiter without convergence: {row.get('target_name')} fold {row.get('fold_id')}")
        try:
            rel = float(row.get("achieved_relative_residual", "nan"))
        except Exception:
            rel = float("nan")
        if not math.isfinite(rel) or rel > residual_limit:
            failures.append(f"Riesz fold residual exceeds tolerance: {row.get('target_name')} fold {row.get('fold_id')}")
    return len(failures) == 0, failures


def preflight(panel_dir: Path, output_root: Path, repo_root: Path,
              config_path: Optional[Path] = None,
              fixed_ranks: Optional[Tuple[int, ...]] = (1, 1, 1, 2),
              select: bool = False, n_jobs: int = 1,
              riesz_maxiter: Optional[int] = None,
              riesz_tol: Optional[float] = None,
              riesz_ridge: Optional[float] = None,
              riesz_use_cached_scale: Optional[bool] = None,
              expected_n: Optional[int] = None,
              expected_t: Optional[int] = None,
              expected_usable: Optional[int] = None) -> Dict[str, object]:
    if expected_n is None:
        expected_n = BASELINE_EXPECTED_N
    if expected_t is None:
        expected_t = BASELINE_EXPECTED_T
    if expected_usable is None:
        expected_usable = BASELINE_EXPECTED_USABLE
    loaded = load_existing_estimation_panel(
        panel_dir,
        expected_n=expected_n,
        expected_t=expected_t,
        expected_usable=expected_usable,
    )
    meta = loaded["metadata"]
    validation = loaded["validation"]
    checksum_actual = loaded["checksums"]
    checks = {
        "required_files_exist": all((panel_dir / name).exists() for name in REQUIRED_PANEL_FILES),
        "input_checksum_ok": all(
            checksum_actual.get(name) == meta.get("checksums", {}).get(name)
            for name in RECORDED_CHECKSUM_FILES
        ),
        "validation_passed": bool(validation.get("validation_passed")),
        "no_missing_primary_values": int(meta.get("transformation_diagnostics", {}).get("missing_primary_values", -1)) == 0,
        "no_preliminary_bls_values": int(meta.get("preliminary_bls_observations", -1)) == 0,
        "exact_monthly_lags": bool(validation.get("lag_values_equal_preceding_observed_calendar_month")),
        "no_absolute_path_required": True,
        "substantive_signature_uses_checksum_not_absolute_root": True,
        "no_existing_conflicting_production_output": not (output_root / "housing_all_homes_results.json").exists(),
    }
    pkg = package_checks()
    writable = output_dirs_writable(output_root)
    tuning = None
    if config_path is not None and config_path.exists():
        tuning = tuning_from_config(
            config_path, fixed_ranks=fixed_ranks, select=select, n_jobs=n_jobs,
            riesz_maxiter=riesz_maxiter, riesz_tol=riesz_tol,
            riesz_ridge=riesz_ridge,
            riesz_use_cached_scale=riesz_use_cached_scale,
        )
    n, t = int(meta["N"]), int(meta["T"])
    tp = max(t - 1, 0)
    estimated_memory_mb = float((n * tp * 8 * 12) / (1024 ** 2))
    estimated_runtime = (
        "Use the N=50 full-time pilot as the local benchmark; production scales "
        "roughly superlinearly in N through the first-stage and Riesz solves."
    )
    ready = all(checks.values()) and all(pkg.values()) and all(writable.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "resolved_repo_root": str(repo_root),
        "repo_relative_input_path": repo_relative(panel_dir, repo_root),
        "resolved_input_path_info": str(panel_dir.resolve()),
        "N": int(meta["N"]),
        "T": int(meta["T"]),
        "usable_dynamic_observations": int(meta["usable_dynamic_observations"]),
        "start_date": meta["start_date"],
        "end_date": meta["end_date"],
        "output_root": repo_relative(output_root, repo_root),
        "resolved_output_root_info": str(output_root.resolve()),
        "package_checks": pkg,
        "output_writable_checks": writable,
        "resolved_tuning": jsonable(dataclasses.asdict(tuning)) if tuning is not None else {},
        "estimated_memory_mb_lower_bound": estimated_memory_mb,
        "estimated_runtime_note": estimated_runtime,
        "existing_production_result_path": str((output_root / "housing_all_homes_results.json").resolve()),
        "validation": validation,
        "portability_checks": checks,
        "ready_for_production": ready,
        "estimator_called": False,
    }
    report["input_checksum"] = checksum_actual["housing_estimation_panel.csv"]
    report["input_checksum_recorded"] = meta.get("checksums", {}).get("housing_estimation_panel.csv")
    report["panel_checksums"] = checksum_actual
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_json(runtime_dir / "production_preflight.json", report)
    return report


def write_second_laptop_checklist(path: Path) -> None:
    lines = [
        "# Second-Laptop Housing Production Checklist",
        "",
        "Run these commands from Git Bash on the second laptop. They assume the repository is at `/d/Programming/dlrhcs`.",
        "",
        "```bash",
        "cd /d/Programming/dlrhcs",
        "git pull",
        "git status --short data/zillow",
        "test -f data/zillow/processed/candidate_panels_final_only/start_2010/housing_panel_levels.csv",
        "test -f data/zillow/processed/estimation_panels/housing_baseline_2010_final/housing_estimation_panel.csv",
        "python scripts/housing_all_homes.py --preflight --repo-root . --panel-id start_2010",
        "python scripts/housing_all_homes.py --smoke --repo-root . --config configs/full.json --seed 2024 --n-jobs 1",
        "# Eventual production, after preflight and smoke pass:",
        "python scripts/housing_all_homes.py --repo-root . --config configs/full.json --seed 2024 --n-jobs 1",
        "```",
        "",
        "Required data are the validated files under `data/zillow/processed`, including `candidate_panels_final_only/start_2010` and `estimation_panels/housing_baseline_2010_final`. If these files are absent from Git on the second laptop, copy them from this machine or regenerate them from already-downloaded local data only after confirming the audit inputs are present. Do not rerun production estimation until `production_preflight.json` reports `ready_for_production: true`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simple_latex(path: Path, rows: Sequence[Mapping[str, object]],
                       columns: Sequence[Tuple[str, str]], caption: str, label: str,
                       note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(c[1] for c in columns) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        vals = []
        for key, _ in columns:
            v = r.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.extend(["", "\\begin{minipage}{0.95\\linewidth}", "\\footnotesize", note, "\\end{minipage}"])
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_metrics_for_pilot(result: Mapping[str, object],
                             baseline: Optional[Mapping[str, object]] = None) -> List[Dict[str, object]]:
    rows = []
    for name, vals in sorted(result["targets"].items()):
        bvals = (baseline or {}).get("targets", {}).get(name, {}) if baseline else {}
        rows.append({
            "target_name": name,
            "estimate": vals["estimate"],
            "se_white": vals["se_white"],
            "se_xs": vals["se_xs"],
            "delta_estimate_vs_baseline": vals["estimate"] - bvals.get("estimate", vals["estimate"]),
            "delta_white_se_vs_baseline": vals["se_white"] - bvals.get("se_white", vals["se_white"]),
            "delta_xs_se_vs_baseline": vals["se_xs"] - bvals.get("se_xs", vals["se_xs"]),
        })
    return rows


def run_riesz_pilots(panel_dir: Path, pilot_root: Path, config_path: Path, repo_root: Path,
                     *, seed: int, fixed_ranks: Optional[Tuple[int, ...]],
                     select: bool, n_jobs: int) -> List[Dict[str, object]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    current = cfg.get("empirical", {})
    current_tol = float(current.get("riesz_tol", 1e-5))
    current_ridge = float(current.get("riesz_ridge", 1e-6))
    geometries = [
        ("A_tiny_N20_T60", 20, 60),
        ("B_representative_N50_T100", 50, 100),
        ("C_full_time_N50", 50, None),
    ]
    tunings = [
        ("baseline", 600, current_tol, current_ridge),
        ("maxiter_2000", 2000, current_tol, current_ridge),
        ("tol_1e-5_ridge_1e-6", 2000, 1e-5, 1e-6),
        ("tol_1e-5_ridge_1e-5", 2000, 1e-5, 1e-5),
    ]
    pilot_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    baselines: Dict[str, Mapping[str, object]] = {}
    for geom, n_msas, t_usable in geometries:
        for label, maxiter, tol, ridge in tunings:
            pilot_id = f"{geom}__{label}"
            out_dir = pilot_root / pilot_id
            result = run_estimation(
                panel_dir, out_dir, config_path, repo_root=repo_root,
                smoke=False, pilot_id=pilot_id, first_n_msas=n_msas,
                first_t_usable=t_usable, seed=seed, fixed_ranks=fixed_ranks,
                select=select, n_jobs=n_jobs, overwrite=True,
                riesz_maxiter=maxiter, riesz_tol=tol, riesz_ridge=ridge,
            )
            if label == "baseline":
                baselines[geom] = result
            baseline = baselines.get(geom)
            rsum = result["riesz_diagnostics"]
            opt = result["diagnostics"]
            for tm in target_metrics_for_pilot(result, baseline):
                rows.append({
                    "geometry": geom,
                    "tuning": label,
                    "N": result["N"],
                    "usable_T": result["Tp"],
                    "maxiter": maxiter,
                    "tol": tol,
                    "ridge": ridge,
                    "number_of_solves": rsum["number_of_solves"],
                    "convergence_fraction": rsum["convergence_fraction"],
                    "iterations_mean": rsum["iterations_mean"],
                    "iterations_median": rsum["iterations_median"],
                    "iterations_max": rsum["iterations_max"],
                    "relative_residual_median": rsum["relative_residual_median"],
                    "relative_residual_max": rsum["relative_residual_max"],
                    "solution_norm_median": rsum["solution_norm_median"],
                    "solution_norm_max": rsum["solution_norm_max"],
                    "number_reaching_maxiter": rsum["number_reaching_maxiter"],
                    "number_containing_nonfinite_values": rsum["number_containing_nonfinite_values"],
                    "target_name": tm["target_name"],
                    "estimate": tm["estimate"],
                    "se_white": tm["se_white"],
                    "se_xs": tm["se_xs"],
                    "delta_estimate_vs_baseline": tm["delta_estimate_vs_baseline"],
                    "delta_white_se_vs_baseline": tm["delta_white_se_vs_baseline"],
                    "delta_xs_se_vs_baseline": tm["delta_xs_se_vs_baseline"],
                    "total_runtime_sec": result["runtime_sec"],
                    "first_stage_monotone": opt.get("monotone", ""),
                    "first_stage_sweep_cap_hit_rate": opt.get("first_stage_sweep_cap_hit_rate", opt.get("first_stage_max_iteration_hit_rate", "")),
                    "first_stage_final_relative_objective_decrease_mean": opt.get("first_stage_final_relative_objective_decrease_mean", ""),
                    "retained_nonvalidation": opt.get("retained_nonvalidation", ""),
                    "retained_total": opt.get("retained_total", ""),
                    "ranks": ",".join(str(x) for x in result["ranks"]),
                })
    fields = [
        "geometry", "tuning", "N", "usable_T", "maxiter", "tol", "ridge",
        "number_of_solves", "convergence_fraction", "iterations_mean",
        "iterations_median", "iterations_max", "relative_residual_median",
        "relative_residual_max", "solution_norm_median", "solution_norm_max",
        "number_reaching_maxiter", "number_containing_nonfinite_values",
        "target_name", "estimate", "se_white", "se_xs",
        "delta_estimate_vs_baseline", "delta_white_se_vs_baseline",
        "delta_xs_se_vs_baseline", "total_runtime_sec", "first_stage_monotone",
        "first_stage_sweep_cap_hit_rate", "first_stage_final_relative_objective_decrease_mean",
        "retained_nonvalidation", "retained_total", "ranks",
    ]
    write_csv(pilot_root / "tab_housing_riesz_pilots.csv", rows, fields)
    write_simple_latex(
        pilot_root / "tab_housing_riesz_pilots.tex",
        rows,
        [("geometry", "Geometry"), ("tuning", "Tuning"), ("target_name", "Target"),
         ("convergence_fraction", "Conv."), ("iterations_median", "Med. iter."),
         ("relative_residual_median", "Med. rel. resid."), ("estimate", "Est."),
         ("se_white", "White s.e."), ("se_xs", "XS s.e.")],
        "Housing Riesz diagnostic pilots.",
        "tab:housing-riesz-pilots",
        "Matched diagnostic pilots vary only Riesz solver settings within geometry."
    )
    report = [
        "# Housing Riesz Pilot Report",
        "",
        "These pilots use only the three existing housing mean targets and deterministic first-N-MSA/date subsets. They do not run the full N=169 production panel.",
        "",
        f"Rows written: {len(rows)}",
        f"Pilot root: `{repo_relative(pilot_root, repo_root)}`",
        "",
        "SciPy CG status convention: `info == 0` indicates successful convergence to the requested tolerance; `info > 0` indicates the tolerance was not achieved within the requested iteration limit; `info < 0` indicates illegal input or breakdown.",
    ]
    write_json(pilot_root / "housing_riesz_pilot_summary.json", {"rows": rows})
    (pilot_root / "housing_riesz_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def run_estimation(panel_dir: Path, out_dir: Path, config_path: Path, *,
                   repo_root: Optional[Path] = None,
                   smoke: bool = False, pilot_id: Optional[str] = None,
                   first_n_msas: Optional[int] = None,
                   first_t_usable: Optional[int] = None,
                   seed: int = 2024,
                   fixed_ranks: Optional[Tuple[int, ...]] = (1, 1, 1, 2),
                   select: bool = False, n_jobs: int = 1,
                   resume: bool = True, overwrite: bool = False,
                   production: bool = False,
                   progress_every: float = 30.0,
                   progress: Optional[ProductionProgress] = None,
                   riesz_maxiter: Optional[int] = None,
                   riesz_tol: Optional[float] = None,
                   riesz_ridge: Optional[float] = None,
                   riesz_use_cached_scale: Optional[bool] = None,
                   _testing_raise_after_estimator_checkpoint: Optional[str] = None) -> Dict[str, object]:
    repo_root = repo_root or find_repo_root(start=__file__)
    out_dir.mkdir(parents=True, exist_ok=True)
    if production and progress is None:
        progress = ProductionProgress(out_dir, progress_every=progress_every, overwrite_log=overwrite)
    housing_root = out_dir.parent if out_dir.name == "smoke" else out_dir
    for child in ["tables", "figures", "audit"]:
        (housing_root / child).mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "housing_all_homes_results.json"
    meta_path = out_dir / "metadata.json"
    if smoke:
        first_n_msas = 20 if first_n_msas is None else first_n_msas
        first_t_usable = 60 if first_t_usable is None else first_t_usable
    if progress:
        progress.emit("input_loading", "input panel loading started", status="running")
    panel = load_estimation_panel(
        panel_dir,
        first_n_msas=first_n_msas,
        first_t_usable=first_t_usable,
    )
    if progress:
        progress.emit(
            "input_loading",
            f"input panel loading completed N={panel['Y'].shape[1]} T={panel['Y'].shape[0] + 1} usable={panel['Y'].size}",
            status="running",
            completed_units=1,
            total_units=1,
        )
    tuning = tuning_from_config(
        config_path, fixed_ranks=fixed_ranks, select=select, n_jobs=n_jobs,
        riesz_maxiter=riesz_maxiter, riesz_tol=riesz_tol,
        riesz_ridge=riesz_ridge, riesz_use_cached_scale=riesz_use_cached_scale,
    )
    blocks = build_blocks(panel["Z"])
    targets = housing_targets(blocks)
    input_identity = {
        "schema_version": panel["metadata"].get("schema_version"),
        "input_checksum": panel["panel_checksum"],
        "candidate_id": panel["metadata"].get("candidate_id"),
        "N_source": int(panel["metadata"].get("N", 0)),
        "T_source": int(panel["metadata"].get("T", 0)),
        "start_date": panel["metadata"].get("start_date"),
        "end_date": panel["metadata"].get("end_date"),
        "outcome": panel["metadata"].get("outcome"),
        "controls": panel["metadata"].get("controls"),
        "repo_relative_input_path": repo_relative(panel_dir, repo_root),
    }
    signature = jsonable({
        "schema_version": SCHEMA_VERSION,
        "input_identity": input_identity,
        "mode": "production" if production else ("pilot" if pilot_id else ("smoke" if smoke else "production_candidate")),
        "full_production_run": bool(production),
        "panel_id": panel["metadata"].get("candidate_id"),
        "smoke": bool(smoke),
        "pilot_id": pilot_id,
        "first_n_msas": first_n_msas,
        "first_t_usable": first_t_usable,
        "seed": int(seed),
        "fixed_ranks": list(fixed_ranks) if fixed_ranks is not None else None,
        "select": bool(select),
        "resolved_tuning": dataclasses.asdict(tuning),
        "N": int(panel["Y"].shape[1]),
        "Tp": int(panel["Y"].shape[0]),
        "level_T": int(panel["Y"].shape[0] + 1),
        "date_range": {
            "start_date": panel["metadata"].get("start_date"),
            "end_date": panel["metadata"].get("end_date"),
        },
        "targets": [t.name for t in targets],
    })
    run_hash = signature_hash(signature)
    if progress:
        progress.set_signature_hash(run_hash)
        progress.emit(
            "resolved_tuning",
            (
                f"N={panel['Y'].shape[1]} T={panel['Y'].shape[0] + 1} usable={panel['Y'].size} "
                f"targets={[t.name for t in targets]} ranks={tuning.ranks} tuning={jsonable(dataclasses.asdict(tuning))}"
            ),
            status="running",
        )
    if resume and not overwrite and result_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old_sig = old.get("run_signature", {})
        if old_sig == signature:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if progress:
                progress.emit("completed_output_reuse", "compatible completed production output reused", status="completed")
            return result
        raise SystemExit(f"refusing to overwrite/resume incompatible housing output at {out_dir}")

    mode = "production" if production else ("pilot" if pilot_id else ("smoke" if smoke else "production_candidate"))
    t0 = time.perf_counter()
    output: Optional[Dict[str, object]] = None
    if production:
        output = load_compatible_estimator_checkpoint(
            out_dir, run_hash, signature, overwrite=overwrite, progress=progress,
        )
        if output is not None and progress:
            progress.emit(
                "checkpoint_resume",
                "resumed from completed estimator checkpoint; estimator will not be recomputed",
                status="running",
                completed_units=1,
                total_units=1,
                last_completed_checkpoint="estimator_output",
            )
    if output is None:
        if progress:
            progress.emit("estimator", "estimator started; internal ALS/fold/Riesz progress is not exposed by the estimator API", status="running")
            progress.heartbeat("estimator", "estimator running; no completed internal unit yet", status="running")
        res = estimate(panel["Y"], panel["Z"], targets, tuning, P=1,
                       rng=np.random.default_rng(seed), profile_timing=True)
        elapsed = time.perf_counter() - t0
        if progress:
            progress.emit("estimator", "estimator completed", status="running", completed_units=1, total_units=1)
        target_table = {}
        for tg in targets:
            target_table[tg.name] = {
                "estimate": float(res.estimates[tg.name]),
                "se_white": float(res.se[tg.name]),
                "se_xs": float(res.se_xs[tg.name]),
                "ci_white": [float(v) for v in res.ci[tg.name]],
                "ci_xs": [float(v) for v in res.ci_xs[tg.name]],
                "plugin": float(res.onestep.plugins.get(tg.name, float("nan"))),
            }
        riesz_diag = getattr(res.onestep, "riesz_diag", {}) or {}
        riesz_rows = flatten_riesz_diagnostics(riesz_diag)
        cg_iters = [int(x) for d in riesz_diag.values() for x in d.get("cg_iters", [])]
        cg_conv = [bool(x) for d in riesz_diag.values() for x in d.get("converged", [])]
        riesz_summary = {
            "number_of_targets": len(targets),
            "number_of_folds": int(res.J),
            "total_riesz_solves": int(sum(len(d.get("cg_iters", [])) for d in riesz_diag.values())),
            "cg_converged_fraction": float(np.mean(cg_conv)) if cg_conv else "",
            "cg_iterations_mean": float(np.mean(cg_iters)) if cg_iters else "",
            "cg_iterations_max": int(max(cg_iters)) if cg_iters else "",
        }
        riesz_summary.update(summarize_riesz_rows(riesz_rows))
        output = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "pilot_id": pilot_id,
            "full_production_run": bool(production),
            "input_panel": {
                "repo_relative_path": repo_relative(panel_dir, repo_root),
                "resolved_absolute_path_info": str(panel_dir.resolve()),
                "checksum": panel["panel_checksum"],
                "source_candidate_id": panel["metadata"].get("candidate_id"),
                "source_start_date": panel["metadata"].get("start_date"),
                "source_end_date": panel["metadata"].get("end_date"),
            },
            "N": int(panel["Y"].shape[1]),
            "Tp": int(panel["Y"].shape[0]),
            "level_T": int(panel["Y"].shape[0] + 1),
            "usable_dynamic_observations": int(panel["Y"].size),
            "months": panel["months"],
            "cbsa_codes": panel["codes"],
            "ranks": list(res.ranks),
            "q": int(res.q),
            "J": int(res.J),
            "targets": target_table,
            "diagnostics": jsonable(res.diagnostics),
            "riesz_diagnostics": jsonable(riesz_summary),
            "riesz_fold_diagnostics": jsonable(riesz_rows),
            "target_names_unique": len({t.name for t in targets}) == len(targets),
            "resolved_tuning": jsonable(dataclasses.asdict(tuning)),
            "runtime_sec": float(elapsed),
            "run_signature": signature,
        }
        if production:
            if progress:
                progress.emit("checkpoint", "writing completed-estimator checkpoint", status="running")
            write_json(estimator_checkpoint_path(out_dir), {
                "schema_version": SCHEMA_VERSION,
                "run_signature_hash": run_hash,
                "run_signature": signature,
                "checkpoint": "estimator_output",
                "updated_utc": utc_now(),
                "output": output,
            })
            write_checkpoint_manifest(
                out_dir, run_hash, signature, "running", "estimator_output",
                "completed estimator output serialized",
            )
            if progress:
                progress.emit("checkpoint", "completed-estimator checkpoint written", status="running",
                              last_completed_checkpoint="estimator_output")
            if _testing_raise_after_estimator_checkpoint == "keyboard":
                if progress:
                    progress.emit("interrupted", "interrupted after estimator checkpoint", status="interrupted",
                                  last_completed_checkpoint="estimator_output")
                raise KeyboardInterrupt()
            if _testing_raise_after_estimator_checkpoint == "exception":
                if progress:
                    progress.emit("failed", "injected failure after estimator checkpoint", status="failed",
                                  last_completed_checkpoint="estimator_output")
                raise RuntimeError("injected failure after estimator checkpoint")
    else:
        riesz_rows = output.get("riesz_fold_diagnostics", []) or []
        riesz_summary = output.get("riesz_diagnostics", {}) or {}
    if production:
        if progress:
            progress.emit("validation", "production validation started", status="validating")
        production_valid, production_failures = validate_production_output(output, targets, tuning)
        output["production_valid"] = bool(production_valid)
        output["production_validation_failures"] = production_failures
        if progress:
            progress.emit(
                "validation",
                f"production validation completed production_valid={production_valid}",
                status="validating",
            )
    if progress:
        progress.emit("output_writing", "output writing started", status="running")
    write_json(result_path, output)
    if pilot_id or production:
        fields = [
            "target_name", "fold_id", "solver_name", "convergence_info_code",
            "converged", "iterations", "maxiter", "requested_tolerance",
            "achieved_absolute_residual", "achieved_relative_residual",
            "rhs_norm", "solution_norm", "maximum_absolute_solution_entry",
            "riesz_ridge", "scaling_value", "cached_scale", "elapsed_seconds",
            "contains_nonfinite",
        ]
        write_csv(out_dir / "riesz_diagnostics.csv", riesz_rows, fields)
        write_json(out_dir / "riesz_summary.json", riesz_summary)
    meta_output = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": signature,
        "result_path": str(result_path),
        "result_repo_relative_path": repo_relative(result_path, repo_root),
        "result_resolved_absolute_path_info": str(result_path.resolve()),
        "input_panel_checksum": panel["panel_checksum"],
        "repo_relative_input_path": repo_relative(panel_dir, repo_root),
        "resolved_input_path_info": str(panel_dir.resolve()),
        "resolved_tuning": jsonable(dataclasses.asdict(tuning)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "full_production_run": bool(production),
        "production_valid": output.get("production_valid"),
        "production_validation_failures": output.get("production_validation_failures", []),
    }
    write_json(meta_path, meta_output)
    if production:
        write_checkpoint_manifest(
            out_dir, run_hash, signature, "completed", "estimator_output",
            "production output completed",
        )
    if progress:
        progress.emit("output_writing", "output writing completed", status="running",
                      last_completed_checkpoint="estimator_output" if production else None)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare and smoke-test all-homes housing empirical inputs.")
    ap.add_argument("--repo-root", default=None,
                    help="explicit DLRHCS repository root; default uses DLRHCS_ROOT or script discovery")
    ap.add_argument("--prepare-panel", action="store_true",
                    help="explicitly regenerate the prepared baseline input panel from the final-only candidate")
    ap.add_argument("--prepare-only", action="store_true",
                    help="legacy alias for --prepare-panel; prepare and validate the baseline input panel only")
    ap.add_argument("--preflight", action="store_true", help="validate production readiness without estimation")
    ap.add_argument("--run-riesz-pilots", action="store_true", help="run matched no-production Riesz diagnostic pilots")
    ap.add_argument("--production", action="store_true", help="run the validated full housing production estimator")
    ap.add_argument("--panel-id", default="start_2010",
                    help="final-only candidate id for preflight/preparation; baseline is start_2010")
    ap.add_argument("--smoke", action="store_true", help="run deterministic first-20-MSA/first-60-month smoke test")
    ap.add_argument("--panel-dir", default=str(BASELINE_REL))
    ap.add_argument("--candidate-root", default=str(FINAL_CANDIDATE_REL))
    ap.add_argument("--output-root", default=str(OUTPUT_REL))
    ap.add_argument("--out-dir", default=str(OUTPUT_REL / "smoke"))
    ap.add_argument("--config", default=str(CONFIG_REL))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--fixed-ranks", type=parse_ranks, default=(1, 1, 1, 2))
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing compatible smoke/output directory intentionally")
    ap.add_argument("--progress-every", type=parse_positive_float, default=30.0,
                    help="seconds between production heartbeat messages during long phases")
    ap.add_argument("--riesz-maxiter", type=parse_positive_int, default=None,
                    help="override Riesz CG max iterations for this housing run")
    ap.add_argument("--riesz-tol", type=parse_positive_float, default=None,
                    help="override Riesz CG relative tolerance for this housing run")
    ap.add_argument("--riesz-ridge", type=parse_positive_float, default=None,
                    help="override Riesz ridge regularization for this housing run only")
    ap.add_argument("--riesz-use-cached-scale", type=parse_bool, default=None,
                    help="override Riesz cached-scale flag for this housing run")
    args = ap.parse_args(argv)

    try:
        repo_root = find_repo_root(start=__file__, explicit=args.repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.panel_id != "start_2010":
        raise SystemExit("only panel-id start_2010 is prepared for the baseline all-homes runner in this revision")
    candidate_root = resolve_repo_path(args.candidate_root, repo_root)
    panel_dir = resolve_repo_path(args.panel_dir, repo_root)
    output_root = resolve_repo_path(args.output_root, repo_root)
    out_dir = resolve_repo_path(args.out_dir, repo_root)
    config_path = resolve_repo_path(args.config, repo_root)

    prepare_mode = bool(args.prepare_panel or args.prepare_only)
    mode_count = sum(bool(x) for x in [
        prepare_mode,
        args.preflight,
        args.run_riesz_pilots,
        args.smoke,
        args.production,
    ])
    if mode_count > 1:
        raise SystemExit(
            "choose exactly one mode: --prepare-panel/--prepare-only, "
            "--preflight, --run-riesz-pilots, --smoke, or --production"
        )

    if prepare_mode:
        panel_meta = prepare_estimation_panel(candidate_root, panel_dir, repo_root=repo_root)
        write_second_laptop_checklist(output_root.parent / "housing_data_audit" / "second_laptop_production_checklist.md")
        print(
            f"[housing-all-homes] prepared baseline {panel_meta['start_date']}..{panel_meta['end_date']} "
            f"N={panel_meta['N']} T={panel_meta['T']} usable={panel_meta['usable_dynamic_observations']}",
            flush=True,
        )
        return 0
    if args.preflight:
        report = preflight(
            panel_dir, output_root, repo_root, config_path=config_path,
            fixed_ranks=args.fixed_ranks, select=args.select, n_jobs=args.n_jobs,
            riesz_maxiter=args.riesz_maxiter, riesz_tol=args.riesz_tol,
            riesz_ridge=args.riesz_ridge,
            riesz_use_cached_scale=args.riesz_use_cached_scale,
        )
        print(f"[housing-all-homes] repo root: {report['resolved_repo_root']}", flush=True)
        print(f"[housing-all-homes] input: {report['repo_relative_input_path']}", flush=True)
        print(f"[housing-all-homes] preflight ready_for_production={report['ready_for_production']}", flush=True)
        return 0 if report["ready_for_production"] else 1
    if args.run_riesz_pilots:
        rows = run_riesz_pilots(
            panel_dir, output_root / "pilots", config_path, repo_root,
            seed=args.seed, fixed_ranks=args.fixed_ranks,
            select=args.select, n_jobs=args.n_jobs,
        )
        print(f"[housing-all-homes] Riesz pilots complete rows={len(rows)}", flush=True)
        return 0
    if args.smoke:
        result = run_estimation(
            panel_dir, out_dir, config_path,
            repo_root=repo_root,
            smoke=True, seed=args.seed, fixed_ranks=args.fixed_ranks,
            select=args.select, n_jobs=args.n_jobs, overwrite=args.overwrite,
            riesz_maxiter=args.riesz_maxiter, riesz_tol=args.riesz_tol,
            riesz_ridge=args.riesz_ridge,
            riesz_use_cached_scale=args.riesz_use_cached_scale,
        )
        print(
            f"[housing-all-homes] smoke complete N={result['N']} Tp={result['Tp']} "
            f"J={result['J']} ranks={result['ranks']} runtime={result['runtime_sec']:.2f}s",
            flush=True,
        )
        for name, vals in result["targets"].items():
            print(f"  {name}: est={vals['estimate']:.6g} se_white={vals['se_white']:.6g} se_xs={vals['se_xs']:.6g}", flush=True)
        return 0
    if args.production:
        production_dir = output_root / "production"
        progress = ProductionProgress(production_dir, progress_every=args.progress_every, overwrite_log=args.overwrite)
        total_start = time.perf_counter()
        try:
            progress.emit("initializing", "production requested", status="initializing")
            progress.emit("preflight", "preflight started", status="running")
            report = preflight(
                panel_dir, output_root, repo_root, config_path=config_path,
                fixed_ranks=args.fixed_ranks, select=args.select, n_jobs=args.n_jobs,
                riesz_maxiter=args.riesz_maxiter, riesz_tol=args.riesz_tol,
                riesz_ridge=args.riesz_ridge,
                riesz_use_cached_scale=args.riesz_use_cached_scale,
            )
            progress.emit("preflight", f"preflight completed ready_for_production={report.get('ready_for_production')}", status="running")
            preflight_failures = []
            if not report.get("ready_for_production"):
                preflight_failures.append("ready_for_production is not True")
            if int(report.get("N", -1)) != BASELINE_EXPECTED_N:
                preflight_failures.append(f"N is {report.get('N')}, expected {BASELINE_EXPECTED_N}")
            if int(report.get("T", -1)) != BASELINE_EXPECTED_T:
                preflight_failures.append(f"T is {report.get('T')}, expected {BASELINE_EXPECTED_T}")
            if int(report.get("usable_dynamic_observations", -1)) != BASELINE_EXPECTED_USABLE:
                preflight_failures.append(
                    f"usable_dynamic_observations is {report.get('usable_dynamic_observations')}, "
                    f"expected {BASELINE_EXPECTED_USABLE}"
                )
            validation = report.get("validation", {})
            if int(validation.get("preliminary_bls_observations", -1)) != 0:
                preflight_failures.append("preliminary BLS observations are present")
            if not report.get("portability_checks", {}).get("input_checksum_ok"):
                preflight_failures.append("input checksums are invalid")
            if preflight_failures:
                progress.emit("preflight", "production preflight failed", status="failed")
                print("[housing-all-homes] production preflight failed:", flush=True)
                for failure in preflight_failures:
                    print(f"  - {failure}", flush=True)
                return 1
            result = run_estimation(
                panel_dir, production_dir, config_path,
                repo_root=repo_root,
                smoke=False, production=True, seed=args.seed,
                fixed_ranks=args.fixed_ranks, select=args.select,
                n_jobs=args.n_jobs, overwrite=args.overwrite,
                progress_every=args.progress_every, progress=progress,
                riesz_maxiter=args.riesz_maxiter, riesz_tol=args.riesz_tol,
                riesz_ridge=args.riesz_ridge,
                riesz_use_cached_scale=args.riesz_use_cached_scale,
            )
            if not result.get("production_valid", False):
                progress.emit("validation", "production completed with invalid diagnostics", status="failed")
                print("[housing-all-homes] production completed with invalid diagnostics", flush=True)
                for failure in result.get("production_validation_failures", []):
                    print(f"  - {failure}", flush=True)
                print(f"[housing-all-homes] output directory: {repo_relative(production_dir, repo_root)}", flush=True)
                return 1
            rdiag = result.get("riesz_diagnostics", {})
            total_elapsed = time.perf_counter() - total_start
            progress.emit("completed", f"production completed total_elapsed={total_elapsed:.2f}s", status="completed",
                          last_completed_checkpoint="estimator_output")
        except KeyboardInterrupt:
            progress.emit("interrupted", "production interrupted", status="interrupted")
            return 1
        except SystemExit:
            progress.emit("failed", "production aborted", status="failed")
            raise
        except Exception as exc:
            progress.emit("failed", f"production failed: {exc}", status="failed")
            return 1
        print("[housing-all-homes] production complete", flush=True)
        print(f"  N={result['N']}", flush=True)
        print(f"  T={result['level_T']}", flush=True)
        print(f"  J={result['J']}", flush=True)
        print(f"  ranks={result['ranks']}", flush=True)
        print(f"  runtime={result['runtime_sec']:.2f}s", flush=True)
        print(f"  Riesz convergence fraction={rdiag.get('convergence_fraction')}", flush=True)
        print(f"  maximum Riesz iterations={rdiag.get('iterations_max')}", flush=True)
        print(f"  maximum relative residual={rdiag.get('relative_residual_max')}", flush=True)
        print(f"  output directory={repo_relative(production_dir, repo_root)}", flush=True)
        for name, vals in result["targets"].items():
            print(f"  {name}: est={vals['estimate']:.6g} se_white={vals['se_white']:.6g} se_xs={vals['se_xs']:.6g}", flush=True)
        return 0
    else:
        print(
            "[housing-all-homes] no mode selected; no estimation or panel preparation was run. "
            "Use --preflight to validate, --smoke for the deterministic smoke test, or --production for the full run.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
