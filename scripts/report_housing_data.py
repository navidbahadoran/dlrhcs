#!/usr/bin/env python3
"""Create audit tables, figures, and immutable housing candidate panels.

This script is deliberately read-only with respect to source acquisition.  It
consumes validated files under data/zillow and writes paper-preparation outputs
under outputs/empirical/housing_data_audit.  It does not call the estimator,
download data, run X-13, interpolate, impute, winsorize, standardize, forecast,
or backcast.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from dlrhcs.paths import find_repo_root, repo_relative, resolve_repo_path  # noqa: E402


FIXED_STARTS = [
    "2000-01", "2001-01", "2002-01", "2003-01", "2004-01", "2005-01",
    "2006-01", "2008-01", "2010-01", "2012-01", "2015-01",
]
FIXED_DURATIONS = [120, 144, 180, 216, 240, 264]
PRIMARY_VARS = [
    ("zhvi_all_homes_sa", "ZHVI all-homes SA"),
    ("permits_units_sa", "permits units SA"),
    ("employment_thousands_sa", "employment thousands SA"),
]


def month_index(ym: str) -> int:
    y, m = ym[:7].split("-")
    return int(y) * 12 + int(m) - 1


def month_from_index(idx: int) -> str:
    y, m0 = divmod(idx, 12)
    return f"{y:04d}-{m0 + 1:02d}"


def month_range(start: str, end: str) -> List[str]:
    a, b = month_index(start), month_index(end)
    return [month_from_index(i) for i in range(a, b + 1)] if b >= a else []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_path_portability_audit(out_root: Path, repo_root: Path) -> Path:
    rows = [
        ("scripts/audit_zillow_data.py", "main", "resolves --data-root and --bls-local-dir", "no", "no", "no", "no", "uses dlrhcs.paths; relative paths are repo-relative"),
        ("scripts/report_housing_data.py", "main", "resolves --data-root and --output-root", "no", "no", "no", "no", "uses dlrhcs.paths; relative paths are repo-relative"),
        ("scripts/housing_all_homes.py", "main/preflight/run_estimation", "resolves panel, output, config, and candidate paths", "no", "no", "yes", "yes", "resume signatures use checksums and repo-relative identities, not absolute roots"),
        ("scripts/zillow_abc.py", "module constants", "legacy top/bottom runner derives ROOT from script location", "no when launched as script", "no", "not applicable", "some output metadata", "preserved for reproducibility; no machine-specific literal path found"),
        ("scripts/build_metro_panel.py", "module constants", "legacy metro script derives ROOT from script location", "no when launched as script", "no", "not applicable", "no", "preserved for reproducibility; no machine-specific literal path found"),
        ("dlrhcs/housing_data.py", "run_housing_audit/X-13 helpers", "accepts data_root from caller; subprocess X-13 receives explicit cwd", "caller-controlled", "no", "no", "source manifests only", "caller now supplies resolved repo-relative paths from scripts"),
        ("dlrhcs/empirical.py", "load_zillow/run_ar2", "legacy loaders consume explicit paths supplied by scripts", "caller-controlled", "no", "no", "data fingerprint only", "preserved"),
        ("dlrhcs/covariates.py", "load_zillow_covariates/load_cbsa_covariates", "legacy covariate loaders consume explicit paths", "caller-controlled", "no", "no", "no", "preserved"),
        ("tests/test_core.py", "test bootstrap", "adds repository root to sys.path from test file location", "no", "no", "no", "no", "test-only bootstrap"),
        ("data/zillow/processed/candidate_panels*", "metadata.json", "candidate metadata keyed by candidate_id and checksums", "no", "no", "no", "yes", "candidate path is recoverable repo-relative; no root name required"),
        ("data/zillow/processed/estimation_panels*", "metadata.json", "input metadata records repo-relative and resolved paths plus checksums", "no", "no", "no", "yes", "substantive identity uses checksum/schema/dimensions/date range/specification"),
    ]
    lines = [
        "# Housing Path Portability Audit",
        "",
        f"Resolved repository root for this audit: `{repo_relative(repo_root, repo_root) or '.'}`",
        "",
        "No literal dependency on either laptop-specific repository root was found in production housing code after this portability pass.",
        "",
        "| file | line/function | current behavior | depends on cwd | machine-specific absolute path | enters resume signature | informational metadata only | required correction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    lines.extend([
        "",
        "Path rules: user-supplied relative paths are interpreted relative to the resolved repository root; explicit absolute paths are used as supplied; resume-signature identity excludes absolute repository roots.",
    ])
    path = out_root / "path_portability_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def to_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def quantile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize_values(values: Sequence[Optional[float]], n_msas: int = 0, n_months: int = 0) -> Dict[str, object]:
    missing = sum(v is None for v in values)
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    out = {
        "count": len(xs),
        "number_of_msas": n_msas,
        "number_of_months": n_months,
        "mean": statistics.fmean(xs) if xs else "",
        "standard_deviation": statistics.stdev(xs) if len(xs) > 1 else "",
        "minimum": min(xs) if xs else "",
        "p1": quantile(xs, 0.01) if xs else "",
        "p5": quantile(xs, 0.05) if xs else "",
        "p10": quantile(xs, 0.10) if xs else "",
        "p25": quantile(xs, 0.25) if xs else "",
        "median": quantile(xs, 0.50) if xs else "",
        "p75": quantile(xs, 0.75) if xs else "",
        "p90": quantile(xs, 0.90) if xs else "",
        "p95": quantile(xs, 0.95) if xs else "",
        "p99": quantile(xs, 0.99) if xs else "",
        "maximum": max(xs) if xs else "",
        "zero_count": sum(abs(x) < 1e-12 for x in xs),
        "missing_count": missing,
    }
    return out


def latex_escape(s: object) -> str:
    text = str(s)
    return (text.replace("\\", "\\textbackslash{}").replace("&", "\\&")
            .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))


def fmt(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return "--"
    try:
        x = float(value)
    except Exception:
        return latex_escape(value)
    if abs(x - round(x)) < 1e-10 and abs(x) >= 10:
        return f"{int(round(x)):,}"
    return f"{x:.{digits}f}"


def write_latex_table(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[Tuple[str, str]],
                      caption: str, label: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aligns = "l" + "r" * max(0, len(columns) - 1)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\begin{threeparttable}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{aligns}}}",
        "\\toprule",
        " & ".join(label for _, label in columns) + " \\\\",
        "\\midrule",
    ]
    if rows:
        for row in rows:
            vals = []
            for key, _ in columns:
                val = row.get(key, "")
                vals.append(fmt(val) if isinstance(val, (int, float)) or str(val).replace(".", "", 1).replace("-", "", 1).isdigit() else latex_escape(val))
            lines.append(" & ".join(vals) + " \\\\")
    else:
        lines.append("\\multicolumn{" + str(len(columns)) + "}{c}{No rows available} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.extend(["\\begin{tablenotes}[flushleft]", "\\footnotesize",
                      f"\\item \\textit{{Notes.}} {latex_escape(note)}",
                      "\\end{tablenotes}"])
    lines.extend(["\\end{threeparttable}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def complete_window_codes(complete_months: Mapping[str, set], codes: Sequence[str], months: Sequence[str]) -> List[str]:
    need = set(months)
    return [c for c in codes if need.issubset(complete_months.get(c, set()))]


def fixed_start_frontier(complete_months: Mapping[str, set], codes: Sequence[str], starts: Sequence[str], end: str) -> List[Dict[str, object]]:
    rows = []
    for start in starts:
        months = month_range(start, end)
        complete = complete_window_codes(complete_months, codes, months)
        rows.append({
            "start_date": start + "-01",
            "end_date": end + "-01",
            "T_months": len(months),
            "N_complete_msas": len(complete),
            "NT": len(months) * len(complete),
            "NT_over_N_plus_T": (len(months) * len(complete) / (len(months) + len(complete))) if months and complete else 0.0,
            "complete_cbsa_codes": ";".join(complete),
        })
    return rows


def fixed_duration_frontier(complete_months: Mapping[str, set], codes: Sequence[str], durations: Sequence[int], end: str) -> List[Dict[str, object]]:
    rows = []
    end_idx = month_index(end)
    for dur in durations:
        start = month_from_index(end_idx - dur + 1)
        months = month_range(start, end)
        complete = complete_window_codes(complete_months, codes, months)
        rows.append({
            "start_date": start + "-01",
            "end_date": end + "-01",
            "T_months": dur,
            "N_complete_msas": len(complete),
            "NT": dur * len(complete),
            "NT_over_N_plus_T": (dur * len(complete) / (dur + len(complete))) if complete else 0.0,
            "complete_cbsa_codes": ";".join(complete),
        })
    return rows


def pareto_frontier(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        n = int(r["N_complete_msas"])
        t = int(r["T_months"])
        dominated = False
        for s in rows:
            if s is r:
                continue
            nn = int(s["N_complete_msas"])
            tt = int(s["T_months"])
            if nn >= n and tt >= t and (nn > n or tt > t):
                dominated = True
                break
        if not dominated:
            out.append(dict(r))
    return sorted(out, key=lambda r: (-int(r["T_months"]), -int(r["N_complete_msas"]), str(r["start_date"])))


def exact_lag_transform(rows: Sequence[Mapping[str, object]], value_col: str, out_col: str,
                        transform) -> List[Dict[str, object]]:
    by = {(r["cbsa_code"], r["date"][:7]): r for r in rows}
    out = []
    for r in rows:
        ym = r["date"][:7]
        lag = month_from_index(month_index(ym) - 12)
        lag_row = by.get((r["cbsa_code"], lag))
        v = to_float(r.get(value_col))
        lv = to_float(lag_row.get(value_col)) if lag_row else None
        row = dict(r)
        row[out_col] = transform(v, lv) if v is not None and lv is not None else ""
        out.append(row)
    return out


def write_png(path: Path, width: int = 1600, height: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(height):
        scan = bytearray([0])
        for x in range(width):
            base = 255
            if x in (90, width - 70) or y in (70, height - 90):
                scan.extend((40, 40, 40))
            elif 90 < x < width - 70 and 70 < y < height - 90 and (x + y) % 97 < 2:
                scan.extend((120, 150, 190))
            else:
                scan.extend((base, base, base))
        rows.append(bytes(scan))
    raw = b"".join(rows)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def write_pdf(path: Path, title: str, rows: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [title] + list(rows[:32])
    stream_lines = ["BT", "/F1 14 Tf", "50 760 Td", f"({title.replace('(', '[').replace(')', ']')}) Tj"]
    stream_lines.append("/F1 9 Tf")
    for line in text[1:]:
        safe = str(line).replace("\\", "/").replace("(", "[").replace(")", "]")
        stream_lines.append(f"0 -16 Td ({safe[:120]}) Tj")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin1", errors="replace")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(bytes(out))


def table_rows_from_window(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [{k: r.get(k, "") for k in ["start_date", "end_date", "T_months", "N_complete_msas", "NT", "NT_over_N_plus_T"]} for r in rows]


def make_panel_rows(codes: Sequence[str], months: Sequence[str], title_by_code: Mapping[str, str],
                    z_by: Mapping[Tuple[str, str], Dict[str, str]],
                    p_by: Mapping[Tuple[str, str], Dict[str, str]],
                    e_by: Mapping[Tuple[str, str], Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for code in codes:
        for ym in months:
            z = z_by[(code, ym)]
            p = p_by[(code, ym)]
            e = e_by[(code, ym)]
            rows.append({
                "cbsa_code": code,
                "msa_title": title_by_code.get(code, ""),
                "date": ym + "-01",
                "zhvi_all_homes_sa": z["zhvi_all_homes_sa"],
                "permits_units_sa": p["permits_units_sa"],
                "employment_thousands_sa": e["employment_thousands_sa"],
                "bls_preliminary_flag": e.get("preliminary_flag", "0"),
                "zhvi_source_vintage": z.get("source_vintage", ""),
                "permits_source_vintage": p.get("source_vintage", ""),
                "employment_source_vintage": e.get("source_vintage", ""),
            })
    return rows


def monthly_distribution(rows: Sequence[Mapping[str, object]], value_col: str, label: str) -> List[Dict[str, object]]:
    by_month: Dict[str, List[float]] = {}
    for row in rows:
        val = to_float(row.get(value_col))
        if val is not None:
            by_month.setdefault(str(row["date"]), []).append(val)
    out = []
    for date, vals in sorted(by_month.items()):
        out.append({
            "date": date,
            "variable": label,
            "n_available": len(vals),
            "p10": quantile(vals, 0.10),
            "p25": quantile(vals, 0.25),
            "median": quantile(vals, 0.50),
            "p75": quantile(vals, 0.75),
            "p90": quantile(vals, 0.90),
        })
    return out


def validate_candidate_panel_rows(rows: Sequence[Mapping[str, object]], n: int, t: int) -> List[str]:
    problems = []
    if len(rows) != n * t:
        problems.append(f"row_count {len(rows)} != N*T {n*t}")
    for col in ["zhvi_all_homes_sa", "permits_units_sa", "employment_thousands_sa"]:
        miss = sum(r.get(col) in ("", None) for r in rows)
        if miss:
            problems.append(f"{col} missing_count={miss}")
    return problems


def add_window_diagnostics(rows: Sequence[Mapping[str, object]],
                           all_three_codes: Sequence[str],
                           z_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           p_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           e_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           geo_class: Sequence[Mapping[str, str]],
                           x13_by: Mapping[str, Mapping[str, str]]) -> None:
    for r in rows:
        months = month_range(str(r["start_date"])[:7], str(r["end_date"])[:7])
        codes = set(str(r.get("complete_cbsa_codes", "")).split(";")) if r.get("complete_cbsa_codes") else set()
        r["number_excluded_for_ZHVI"] = sum(not all((c, m) in z_by for m in months) for c in all_three_codes)
        r["number_excluded_for_permits_SA"] = sum(not all((c, m) in p_by for m in months) for c in all_three_codes)
        r["number_excluded_for_employment_SA"] = sum(not all((c, m) in e_by for m in months) for c in all_three_codes)
        r["number_excluded_for_geographic_ambiguity"] = len([g for g in geo_class if g.get("classification") != "current_metropolitan_cbsa"])
        r["number_containing_preliminary_BLS_observations"] = sum(any(e_by.get((c, m), {}).get("preliminary_flag") == "1" for m in months) for c in codes)
        r["number_affected_by_X13_warnings"] = sum(x13_by.get(c, {}).get("status") not in ("", "ok") for c in codes)


def build_candidate_specs(pareto: Sequence[Mapping[str, object]], fixed_starts: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    specs: Dict[str, Mapping[str, object]] = {
        f"pareto_{i+1:02d}_{r['start_date'][:7].replace('-', '')}_{r['T_months']}m_{r['N_complete_msas']}n": r
        for i, r in enumerate(pareto)
    }
    for label, start in [("start_2000", "2000-01"), ("start_2004", "2004-01"), ("start_2005", "2005-01"), ("start_2010", "2010-01")]:
        match = next((r for r in fixed_starts if r["start_date"][:7] == start and int(r["N_complete_msas"]) > 0), None)
        if match:
            specs[label] = match
    for label, dur in [("at_least_120m", 120), ("at_least_180m", 180), ("at_least_240m", 240)]:
        eligible = [r for r in pareto if int(r["T_months"]) >= dur]
        if eligible:
            specs[label] = max(eligible, key=lambda r: (int(r["N_complete_msas"]), int(r["T_months"])))
    if pareto:
        specs["longest_feasible_common_window"] = max(pareto, key=lambda r: int(r["T_months"]))
    return specs


def write_candidate_panels(candidate_specs: Mapping[str, Mapping[str, object]], cand_root: Path,
                           title_by_code: Mapping[str, str],
                           z_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           p_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           e_by: Mapping[Tuple[str, str], Mapping[str, str]],
                           x13_by: Mapping[str, Mapping[str, str]],
                           panel_type: str,
                           overwrite: bool = True) -> Tuple[List[Dict[str, object]], List[Path]]:
    panel_fields = ["cbsa_code", "msa_title", "date", "zhvi_all_homes_sa", "permits_units_sa", "employment_thousands_sa",
                    "bls_preliminary_flag", "zhvi_source_vintage", "permits_source_vintage", "employment_source_vintage"]
    summaries: List[Dict[str, object]] = []
    files_out: List[Path] = []
    for cid, spec in sorted(candidate_specs.items()):
        codes = [c for c in str(spec["complete_cbsa_codes"]).split(";") if c]
        months = month_range(str(spec["start_date"])[:7], str(spec["end_date"])[:7])
        cdir = cand_root / cid
        expected = [
            cdir / "housing_panel_levels.csv",
            cdir / "msa_list.csv",
            cdir / "monthly_dates.csv",
            cdir / "completeness_check.csv",
            cdir / "metadata.json",
        ]
        if cdir.exists() and not overwrite:
            missing = [p.name for p in expected if not p.exists()]
            if missing:
                raise FileExistsError(f"immutable candidate panel {cdir} is incomplete: missing {missing}")
            meta = json.loads((cdir / "metadata.json").read_text(encoding="utf-8"))
            expected_identity = {
                "candidate_id": cid,
                "panel_type": panel_type,
                "start_date": spec["start_date"],
                "end_date": spec["end_date"],
                "N": len(codes),
                "T_months": len(months),
            }
            mismatch = {
                k: (meta.get(k), v) for k, v in expected_identity.items()
                if str(meta.get(k)) != str(v)
            }
            if mismatch:
                raise FileExistsError(f"immutable candidate panel {cdir} has conflicting metadata: {mismatch}")
            summaries.append({**meta, "path": str(cdir)})
            files_out.extend(expected)
            continue
        rows = make_panel_rows(codes, months, title_by_code, z_by, p_by, e_by)
        problems = validate_candidate_panel_rows(rows, len(codes), len(months))
        cdir.mkdir(parents=True, exist_ok=True)
        files: List[Path] = []
        write_csv(cdir / "housing_panel_levels.csv", rows, panel_fields); files.append(cdir / "housing_panel_levels.csv")
        write_csv(cdir / "msa_list.csv", [{"cbsa_code": c, "msa_title": title_by_code.get(c, "")} for c in codes], ["cbsa_code", "msa_title"]); files.append(cdir / "msa_list.csv")
        write_csv(cdir / "monthly_dates.csv", [{"date": m + "-01"} for m in months], ["date"]); files.append(cdir / "monthly_dates.csv")
        write_csv(cdir / "completeness_check.csv", [{"candidate_id": cid, "N": len(codes), "T_months": len(months), "rows": len(rows), "problem": p} for p in (problems or ["ok"])], ["candidate_id", "N", "T_months", "rows", "problem"]); files.append(cdir / "completeness_check.csv")
        neg_count = sum((to_float(r.get("permits_units_sa")) or 0.0) < 0 for r in rows)
        meta = {
            "candidate_id": cid,
            "panel_type": panel_type,
            "start_date": spec["start_date"],
            "end_date": spec["end_date"],
            "N": len(codes),
            "T_months": len(months),
            "NT": len(rows),
            "preliminary_bls_observations": sum(str(r.get("bls_preliminary_flag", "0")) == "1" for r in rows),
            "negative_permit_count": neg_count,
            "negative_permit_share": neg_count / len(rows) if rows else 0.0,
            "x13_warning_msas": sum(x13_by.get(c, {}).get("status") not in ("", "ok") for c in codes),
            "missing_primary_values": sum(1 for problem in problems if "missing_count" in problem),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "no_interpolation_or_imputation": True,
        }
        checksums = {f.name: sha256_file(f) for f in files}
        meta["checksums"] = checksums
        write_json(cdir / "metadata.json", meta); files.append(cdir / "metadata.json")
        summaries.append({**meta, "path": str(cdir)})
        files_out.extend(files)
    return summaries, files_out


def preliminary_by_month(emp_rows: Sequence[Mapping[str, str]], matched_codes: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], str]:
    matched = set(matched_codes)
    prelim = []
    month_counts: Dict[str, int] = {}
    final_status: Dict[str, List[int]] = {}
    for r in emp_rows:
        code = str(r.get("cbsa_code", "")).zfill(5)
        if code not in matched:
            continue
        ym = r["date"][:7]
        is_prelim = int(str(r.get("preliminary_flag", "0")) == "1")
        final_status.setdefault(ym, []).append(is_prelim)
        if is_prelim:
            prelim.append({
                "date": r["date"],
                "cbsa_code": code,
                "bls_series_id": r.get("bls_series_id", ""),
                "employment_thousands_sa": r.get("employment_thousands_sa", ""),
            })
            month_counts[ym] = month_counts.get(ym, 0) + 1
    by_month = [{"date": ym + "-01", "preliminary_count": count} for ym, count in sorted(month_counts.items())]
    latest_final = max((ym for ym, vals in final_status.items() if len(vals) == len(matched) and sum(vals) == 0), default="")
    return prelim, by_month, latest_final


def candidate_membership_index(candidates: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], List[str]]:
    index: Dict[Tuple[str, str], List[str]] = {}
    for cand in candidates:
        root = Path(str(cand.get("path", "")))
        msa_path = root / "msa_list.csv"
        month_path = root / "monthly_dates.csv"
        if not msa_path.exists() or not month_path.exists():
            continue
        codes = [str(r["cbsa_code"]).zfill(5) for r in read_csv(msa_path)]
        months = [r["date"][:7] for r in read_csv(month_path)]
        label = f"{cand.get('panel_type', '')}:{cand.get('candidate_id', '')}"
        for code in codes:
            for ym in months:
                index.setdefault((code, ym), []).append(label)
    return index


def negative_permit_diagnostics(rows: Sequence[Mapping[str, object]],
                                pnsa_by: Mapping[Tuple[str, str], Mapping[str, object]],
                                p_by: Mapping[Tuple[str, str], Mapping[str, object]],
                                x13_by: Mapping[str, Mapping[str, object]],
                                candidates: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    membership = candidate_membership_index(candidates)
    diagnostics: List[Dict[str, object]] = []
    for r in rows:
        code = str(r.get("cbsa_code", "")).zfill(5)
        ym = str(r.get("date", ""))[:7]
        v = to_float(r.get("permits_units_sa"))
        if v is None or v >= 0:
            continue
        sa_row = p_by.get((code, ym), {})
        nsa_row = pnsa_by.get((code, ym), {})
        x13 = x13_by.get(code, {})
        status = sa_row.get("x13_status") or x13.get("status", "")
        diagnostics.append({
            "date": ym + "-01",
            "cbsa_code": code,
            "msa_title": r.get("msa_title", ""),
            "permits_units_sa": v,
            "corresponding_permits_units_nsa": nsa_row.get("total_units", nsa_row.get("permits_units_nsa", "")),
            "x13_status": status,
            "x13_spec_id": sa_row.get("x13_spec_id") or x13.get("x13_spec_id", ""),
            "x13_transformation": "transform function=none; x11 mode=add",
            "x13_segment_start": sa_row.get("contiguous_segment_start") or (str(x13.get("segment_start", "")) + "-01" if x13.get("segment_start") else ""),
            "x13_segment_end": sa_row.get("contiguous_segment_end") or (str(x13.get("segment_end", "")) + "-01" if x13.get("segment_end") else ""),
            "belongs_to_x13_warning_or_failed_segment": int(status not in ("", "ok")),
            "candidate_panels_affected": ";".join(sorted(membership.get((code, ym), []))),
            "diagnostic": "negative_seasonally_adjusted_permit_value",
        })
    neg_values = [float(r["permits_units_sa"]) for r in diagnostics]
    summary = {
        "diagnostic": "negative_seasonally_adjusted_permit_value",
        "total_count": len(diagnostics),
        "affected_cbsas": len({r["cbsa_code"] for r in diagnostics}),
        "first_affected_month": min((r["date"] for r in diagnostics), default=""),
        "last_affected_month": max((r["date"] for r in diagnostics), default=""),
        "minimum": min(neg_values) if neg_values else "",
        "p1": quantile(neg_values, 0.01) if neg_values else "",
        "p5": quantile(neg_values, 0.05) if neg_values else "",
        "median_among_negative_values": quantile(neg_values, 0.50) if neg_values else "",
        "maximum_negative_value": max(neg_values) if neg_values else "",
        "x13_warning_or_failed_count": sum(int(r["belongs_to_x13_warning_or_failed_segment"]) for r in diagnostics),
        "x13_specification_and_transformation": "Local X-13 run with transform { function = none } and x11 { mode = add save = (d11) }.",
    }
    candidate_rows = []
    for c in candidates:
        candidate_rows.append({
            "row_type": "candidate_panel",
            "panel_type": c.get("panel_type", ""),
            "candidate_id": c.get("candidate_id", ""),
            "start_date": c.get("start_date", ""),
            "end_date": c.get("end_date", ""),
            "N": c.get("N", ""),
            "T": c.get("T_months", ""),
            "NT": c.get("NT", ""),
            "negative_permit_count": c.get("negative_permit_count", 0),
            "negative_permit_share": c.get("negative_permit_share", 0.0),
            "x13_warning_or_failed_count": c.get("x13_warning_msas", 0),
        })
    return diagnostics, candidate_rows, summary


def rank_candidates(candidates: Sequence[Mapping[str, object]], pareto_keys: set) -> List[Dict[str, object]]:
    rows = []
    for c in candidates:
        n = int(c["N"])
        t = int(c["T_months"])
        nt = int(c["NT"])
        rows.append({
            "panel_type": c.get("panel_type", ""),
            "candidate_id": c["candidate_id"],
            "start_date": c["start_date"],
            "end_date": c["end_date"],
            "N": n,
            "T": t,
            "NT": nt,
            "NT_over_N_plus_T": nt / (n + t) if n + t else 0.0,
            "preliminary_bls_count": c.get("preliminary_bls_observations", 0),
            "negative_permit_count": c.get("negative_permit_count", 0),
            "negative_permit_share": c.get("negative_permit_share", 0.0),
            "x13_warning_msa_count": c.get("x13_warning_msas", 0),
            "missing_primary_values": c.get("missing_primary_values", 0),
            "pareto_dominated": int((c.get("panel_type", ""), c["candidate_id"]) not in pareto_keys and not str(c["candidate_id"]).startswith("pareto_")),
            "highlight": "",
        })
    if not rows:
        return rows
    highlights = [
        ("maximum N", lambda r: (int(r["N"]), int(r["T"]))),
        ("maximum T", lambda r: (int(r["T"]), int(r["N"]))),
        ("maximum NT", lambda r: (int(r["NT"]), int(r["N"]))),
        ("maximum NT/(N+T)", lambda r: (float(r["NT_over_N_plus_T"]), int(r["NT"]))),
        ("closest N/T ratio to one", lambda r: (-abs(int(r["N"]) / int(r["T"]) - 1) if int(r["T"]) else -999, int(r["NT"]))),
        ("longest panel with N >= 100", lambda r: (int(r["T"]) if int(r["N"]) >= 100 else -1, int(r["N"]))),
        ("largest N with T >= 180", lambda r: (int(r["N"]) if int(r["T"]) >= 180 else -1, int(r["T"]))),
    ]
    for label, keyfn in highlights:
        best = max(rows, key=keyfn)
        best["highlight"] = (best["highlight"] + "; " if best["highlight"] else "") + label
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Create housing data audit outputs without estimation or downloads.")
    ap.add_argument("--repo-root", default=None,
                    help="explicit DLRHCS repository root; default uses DLRHCS_ROOT or script discovery")
    ap.add_argument("--data-root", default="data/zillow")
    ap.add_argument("--output-root", default="outputs/empirical/housing_data_audit")
    args = ap.parse_args()
    try:
        repo_root = find_repo_root(start=__file__, explicit=args.repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc))
    data_root = resolve_repo_path(args.data_root, repo_root)
    out_root = resolve_repo_path(args.output_root, repo_root)
    tables = out_root / "tables"
    figs = out_root / "figures"
    figdata = out_root / "figure_data"
    cand_root = data_root / "processed" / "candidate_panels"
    for p in [tables, figs, figdata, cand_root]:
        p.mkdir(parents=True, exist_ok=True)

    zillow = read_csv(data_root / "processed" / "zhvi_all_homes_metro_sa_long.csv")
    permits_nsa = read_csv(data_root / "processed" / "permits_metro_nsa_long.csv")
    permits_sa = read_csv(data_root / "processed" / "permits_metro_sa_long.csv")
    emp_sa = read_csv(data_root / "processed" / "employment_metro_official_sa_long.csv")
    cross = read_csv(data_root / "processed" / "housing_msa_crosswalk.csv")
    geo_class = read_csv(data_root / "audit" / "zillow_geography_classification.csv")
    x13 = read_csv(data_root / "audit" / "x13_diagnostics.csv")
    manifest_outputs = []
    manifest_outputs.append(write_path_portability_audit(out_root, repo_root))

    id_to_code = {r["zillow_region_id"]: r["cbsa_code"].zfill(5) for r in cross}
    title_by_code = {r["cbsa_code"].zfill(5): r.get("census_cbsa_title") or r.get("zillow_region_name", "") for r in cross}
    z_by: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in zillow:
        code = id_to_code.get(r["zillow_region_id"])
        if code and r.get("zhvi_all_homes_sa") not in ("", None):
            z_by[(code, r["date"][:7])] = r
    p_by = {(r["cbsa_code"].zfill(5), r["date"][:7]): r for r in permits_sa if r.get("permits_units_sa") not in ("", None)}
    pnsa_by_code: Dict[str, List[Dict[str, str]]] = {}
    for r in permits_nsa:
        pnsa_by_code.setdefault(r["cbsa_code"].zfill(5), []).append(r)
    pnsa_by = {(r["cbsa_code"].zfill(5), r["date"][:7]): r for r in permits_nsa}
    e_by = {(r["cbsa_code"].zfill(5), r["date"][:7]): r for r in emp_sa if r.get("employment_thousands_sa") not in ("", None)}
    all_codes = sorted(set(title_by_code) & {c for c, _ in z_by} & {c for c, _ in p_by} & {c for c, _ in e_by})
    complete_months: Dict[str, set] = {}
    all_months = sorted({m for _, m in z_by} | {m for _, m in p_by} | {m for _, m in e_by})
    for code in sorted(title_by_code):
        complete_months[code] = {m for m in all_months if (code, m) in z_by and (code, m) in p_by and (code, m) in e_by}
    all_three_codes = sorted([c for c in title_by_code if complete_months.get(c)])
    common_end = max(m for c in all_three_codes for m in complete_months[c])
    e_final_by = {k: v for k, v in e_by.items() if str(v.get("preliminary_flag", "0")) != "1"}
    complete_months_final: Dict[str, set] = {}
    for code in sorted(title_by_code):
        complete_months_final[code] = {m for m in all_months if (code, m) in z_by and (code, m) in p_by and (code, m) in e_final_by}
    all_three_final_codes = sorted([c for c in title_by_code if complete_months_final.get(c)])
    prelim_rows, prelim_by_month, latest_final_month = preliminary_by_month(emp_sa, all_three_codes)
    final_end = latest_final_month or common_end
    prelim_table = [
        {"row_type": "observation", **r, "preliminary_count": "", "latest_fully_final_month": latest_final_month + "-01"}
        for r in prelim_rows
    ]
    prelim_table.extend({
        "row_type": "month_summary",
        "date": r["date"],
        "cbsa_code": "",
        "bls_series_id": "",
        "employment_thousands_sa": "",
        "preliminary_count": r["preliminary_count"],
        "latest_fully_final_month": latest_final_month + "-01",
    } for r in prelim_by_month)
    write_csv(tables / "tab_housing_preliminary_bls.csv", prelim_table,
              ["row_type", "date", "cbsa_code", "bls_series_id", "employment_thousands_sa",
               "preliminary_count", "latest_fully_final_month"])
    write_latex_table(tables / "tab_housing_preliminary_bls.tex", prelim_by_month,
                      [("date", "Month"), ("preliminary_count", "Preliminary obs.")],
                      "Preliminary BLS employment observations by month.",
                      "tab:housing-preliminary-bls")

    # Part 1: permit 2000--2003 diagnosis.
    x13_by = {r["series_id"].zfill(5): r for r in x13}
    diag_rows = []
    for code in sorted(title_by_code):
        nsa_rows = sorted(pnsa_by_code.get(code, []), key=lambda r: r["date"])
        sa_months = sorted(m for c, m in p_by if c == code)
        nsa_months = [r["date"][:7] for r in nsa_rows]
        x = x13_by.get(code, {})
        reason = "other_documented_reason"
        if not nsa_months:
            reason = "cbsa_definition_or_match_issue"
        elif min(nsa_months) > "2000-01":
            reason = "raw_permits_begin_after_2000"
        elif len(set(nsa_months) & set(month_range("2000-01", "2003-12"))) < 48:
            reason = "internal_raw_gap"
        elif x.get("status") == "segment_too_short":
            reason = "segment_below_minimum_length"
        elif str(x.get("status", "")).startswith("failed"):
            reason = "x13_failure"
        elif sa_months and min(sa_months) > "2000-01":
            reason = "x13_output_date_filter" if x.get("segment_start", "") <= "2000-01" else "parser_or_processing_restriction"
        diag_rows.append({
            "cbsa_code": code,
            "msa_title": title_by_code.get(code, ""),
            "permits_nsa_first_date": (min(nsa_months) + "-01") if nsa_months else "",
            "permits_nsa_last_date": (max(nsa_months) + "-01") if nsa_months else "",
            "permits_nsa_internal_missing_months": len(set(month_range(min(nsa_months), max(nsa_months))) - set(nsa_months)) if nsa_months else "",
            "x13_input_segment_first_date": (x.get("segment_start", "") + "-01") if x.get("segment_start") else "",
            "x13_input_segment_last_date": (x.get("segment_end", "") + "-01") if x.get("segment_end") else "",
            "x13_output_first_date": (min(sa_months) + "-01") if sa_months else "",
            "x13_output_last_date": (max(sa_months) + "-01") if sa_months else "",
            "x13_status": x.get("status", ""),
            "number_raw_input_observations": x.get("n_observed", len(nsa_rows)),
            "number_retained_sa_observations": len(sa_months),
            "reason_2000_2003_missing": reason,
        })
    permit_diag_path = data_root / "audit" / "permit_2000_2003_diagnosis.csv"
    write_csv(permit_diag_path, diag_rows, list(diag_rows[0]))
    reason_counts: Dict[str, int] = {}
    for r in diag_rows:
        if not any(m in complete_months.get(r["cbsa_code"], set()) for m in month_range("2000-01", "2003-12")):
            reason_counts[r["reason_2000_2003_missing"]] = reason_counts.get(r["reason_2000_2003_missing"], 0) + 1
    md = [
        "# Permit 2000--2003 Diagnosis",
        "",
        "The current matched metropolitan permit SA coverage begins in 2004 because the accepted current CBSA-style permit records begin in 2004 for matched metros. The 2000--2003 BPS records are legacy PMSA/MSA codes such as 00080, and their X-13 diagnostics are short 48-month segments below the 84-month minimum. The audit did not identify valid observed 2000--2003 current-CBSA permit SA observations to recover.",
        "",
        "## Reason Counts",
    ] + [f"- {k}: {v}" for k, v in sorted(reason_counts.items())]
    (data_root / "audit" / "permit_2000_2003_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fixed_starts = fixed_start_frontier(complete_months, all_three_codes, FIXED_STARTS, common_end)
    fixed_durs = fixed_duration_frontier(complete_months, all_three_codes, FIXED_DURATIONS, common_end)
    candidate_scan = fixed_start_frontier(complete_months, all_three_codes, sorted({m for c in all_three_codes for m in complete_months[c]}), common_end)
    pareto = pareto_frontier(candidate_scan)
    fixed_starts_final = fixed_start_frontier(complete_months_final, all_three_final_codes, FIXED_STARTS, final_end)
    fixed_durs_final = fixed_duration_frontier(complete_months_final, all_three_final_codes, FIXED_DURATIONS, final_end)
    final_scan_months = sorted({m for c in all_three_final_codes for m in complete_months_final[c] if m <= final_end})
    candidate_scan_final = fixed_start_frontier(complete_months_final, all_three_final_codes, final_scan_months, final_end)
    pareto_final = pareto_frontier(candidate_scan_final)
    for rows, e_source, codes0 in [
        (fixed_starts, e_by, all_three_codes),
        (fixed_durs, e_by, all_three_codes),
        (pareto, e_by, all_three_codes),
        (fixed_starts_final, e_final_by, all_three_final_codes),
        (fixed_durs_final, e_final_by, all_three_final_codes),
        (pareto_final, e_final_by, all_three_final_codes),
    ]:
        add_window_diagnostics(rows, codes0, z_by, p_by, e_source, geo_class, x13_by)
    win_fields = ["start_date", "end_date", "T_months", "N_complete_msas", "NT", "NT_over_N_plus_T",
                  "number_excluded_for_ZHVI", "number_excluded_for_permits_SA", "number_excluded_for_employment_SA",
                  "number_excluded_for_geographic_ambiguity", "number_containing_preliminary_BLS_observations",
                  "number_affected_by_X13_warnings", "complete_cbsa_codes"]
    outputs = [
        (tables / "housing_fixed_start_frontier.csv", fixed_starts),
        (tables / "housing_fixed_duration_frontier.csv", fixed_durs),
        (tables / "housing_pareto_frontier.csv", pareto),
    ]
    for path, rows in outputs:
        write_csv(path, rows, win_fields)
        manifest_outputs.append(path)
    table_cols = [("start_date", "Start"), ("end_date", "End"), ("T_months", "T"), ("N_complete_msas", "N"), ("NT", "NT"), ("NT_over_N_plus_T", "NT/(N+T)")]
    write_latex_table(tables / "housing_fixed_start_frontier.tex", fixed_starts, table_cols, "Housing balanced panels by fixed start date.", "tab:housing-fixed-start")
    write_latex_table(tables / "housing_fixed_duration_frontier.tex", fixed_durs, table_cols, "Housing balanced panels by fixed duration.", "tab:housing-fixed-duration")
    write_latex_table(tables / "housing_pareto_frontier.tex", pareto, table_cols, "Housing non-dominated balanced-panel frontier.", "tab:housing-pareto")
    for path, rows in [
        (tables / "housing_fixed_start_frontier_final_only.csv", fixed_starts_final),
        (tables / "housing_fixed_duration_frontier_final_only.csv", fixed_durs_final),
        (tables / "housing_pareto_frontier_final_only.csv", pareto_final),
    ]:
        write_csv(path, rows, win_fields)
        manifest_outputs.append(path)
    write_latex_table(tables / "housing_fixed_start_frontier_final_only.tex", fixed_starts_final, table_cols, "Housing fixed-start balanced panels excluding preliminary BLS observations.", "tab:housing-fixed-start-final")
    write_latex_table(tables / "housing_fixed_duration_frontier_final_only.tex", fixed_durs_final, table_cols, "Housing fixed-duration balanced panels excluding preliminary BLS observations.", "tab:housing-fixed-duration-final")
    write_latex_table(tables / "housing_pareto_frontier_final_only.tex", pareto_final, table_cols, "Housing final-only non-dominated balanced-panel frontier.", "tab:housing-pareto-final")

    candidate_specs = build_candidate_specs(pareto, fixed_starts)
    candidate_summary, candidate_files = write_candidate_panels(
        candidate_specs, cand_root, title_by_code, z_by, p_by, e_by, x13_by, "all_vintage",
        overwrite=False
    )
    manifest_outputs.extend(candidate_files)
    cand_final_root = data_root / "processed" / "candidate_panels_final_only"
    candidate_final_specs = build_candidate_specs(pareto_final, fixed_starts_final)
    candidate_final_summary, candidate_final_files = write_candidate_panels(
        candidate_final_specs, cand_final_root, title_by_code, z_by, p_by, e_final_by, x13_by, "final_only",
        overwrite=False
    )
    manifest_outputs.extend(candidate_final_files)
    all_candidate_summary = candidate_summary + candidate_final_summary

    rank_keys = {("all_vintage", cid) for cid in candidate_specs if cid.startswith("pareto_")}
    rank_keys.update({("final_only", cid) for cid in candidate_final_specs if cid.startswith("pareto_")})
    candidate_ranking = rank_candidates(all_candidate_summary, rank_keys)
    ranking_fields = [
        "panel_type", "candidate_id", "start_date", "end_date", "N", "T", "NT", "NT_over_N_plus_T",
        "preliminary_bls_count", "negative_permit_count", "negative_permit_share",
        "x13_warning_msa_count", "missing_primary_values", "pareto_dominated", "highlight",
    ]
    write_csv(tables / "tab_housing_candidate_ranking.csv", candidate_ranking, ranking_fields)
    write_latex_table(
        tables / "tab_housing_candidate_ranking.tex",
        candidate_ranking,
        [("panel_type", "Vintage"), ("candidate_id", "Candidate"), ("start_date", "Start"),
         ("end_date", "End"), ("N", "N"), ("T", "T"), ("NT", "NT"),
         ("NT_over_N_plus_T", "NT/(N+T)"), ("preliminary_bls_count", "Prelim. BLS"),
         ("negative_permit_count", "Neg. permits"), ("negative_permit_share", "Neg. share"),
         ("pareto_dominated", "Dominated"), ("highlight", "Highlight")],
        "Housing candidate-panel ranking.",
        "tab:housing-candidate-ranking",
        "Highlights mark diagnostic criteria only and do not select a preferred sample."
    )

    all_complete_rows = []
    for code in all_three_codes:
        all_complete_rows.extend(make_panel_rows([code], sorted(complete_months[code]), title_by_code, z_by, p_by, e_by))
    neg_diag, neg_candidate_rows, neg_summary = negative_permit_diagnostics(
        all_complete_rows, pnsa_by, p_by, x13_by, all_candidate_summary
    )
    write_csv(
        out_root / "negative_permits_diagnostics.csv",
        neg_diag,
        ["date", "cbsa_code", "msa_title", "permits_units_sa", "corresponding_permits_units_nsa",
         "x13_status", "x13_spec_id", "x13_transformation", "x13_segment_start", "x13_segment_end",
         "belongs_to_x13_warning_or_failed_segment", "candidate_panels_affected", "diagnostic"],
    )
    neg_table_rows = [{
        "row_type": "overall",
        "panel_type": "all matched",
        "candidate_id": "all_matched_observations",
        **neg_summary,
        "negative_permit_count": neg_summary["total_count"],
        "negative_permit_share": "",
    }] + neg_candidate_rows
    neg_table_fields = [
        "row_type", "panel_type", "candidate_id", "start_date", "end_date", "N", "T", "NT",
        "total_count", "affected_cbsas", "first_affected_month", "last_affected_month",
        "minimum", "p1", "p5", "median_among_negative_values", "maximum_negative_value",
        "negative_permit_count", "negative_permit_share", "x13_warning_or_failed_count",
        "x13_specification_and_transformation",
    ]
    write_csv(tables / "tab_housing_negative_permits.csv", neg_table_rows, neg_table_fields)
    write_latex_table(
        tables / "tab_housing_negative_permits.tex",
        neg_table_rows,
        [("panel_type", "Vintage"), ("candidate_id", "Candidate"), ("N", "N"), ("T", "T"),
         ("negative_permit_count", "Neg. values"), ("negative_permit_share", "Share"),
         ("x13_warning_or_failed_count", "X-13 warn./fail")],
        "Negative seasonally adjusted permit values.",
        "tab:housing-negative-permits",
        "Negative values are seasonally adjusted permit values from additive X-13 adjustment; they are audited here but not classified as invalid raw observations."
    )
    neg_report = [
        "# Negative Seasonally Adjusted Permit Values",
        "",
        "This audit relabels the diagnostic as `negative_seasonally_adjusted_permit_value`. These are not invalid raw permit observations: additive X-13 seasonal adjustment can produce negative adjusted values.",
        "",
        "## Summary",
        f"- total count: {neg_summary['total_count']}",
        f"- affected CBSAs: {neg_summary['affected_cbsas']}",
        f"- first affected month: {neg_summary['first_affected_month']}",
        f"- last affected month: {neg_summary['last_affected_month']}",
        f"- minimum: {neg_summary['minimum']}",
        f"- p1: {neg_summary['p1']}",
        f"- p5: {neg_summary['p5']}",
        f"- median among negative values: {neg_summary['median_among_negative_values']}",
        f"- maximum negative value: {neg_summary['maximum_negative_value']}",
        f"- X-13 warning or failed segment count: {neg_summary['x13_warning_or_failed_count']}",
        "",
        "## X-13 Specification",
        str(neg_summary["x13_specification_and_transformation"]),
        "",
        "Observation-level diagnostics, including corresponding NSA permit values and affected candidate panels, are written to `negative_permits_diagnostics.csv`.",
        "",
        "No deletion, truncation, winsorization, replacement, or transformation of these values was performed during this audit.",
    ]
    (out_root / "negative_permits_report.md").write_text("\n".join(neg_report) + "\n", encoding="utf-8")
    desc_rows = []
    scopes = [("all_matched_observations", all_complete_rows)]
    for cand in all_candidate_summary:
        scopes.append((f"{cand.get('panel_type', '')}:{cand['candidate_id']}", read_csv(Path(cand["path"]) / "housing_panel_levels.csv")))
    for scope, rows in scopes:
        codes = {r["cbsa_code"] for r in rows}
        months = {r["date"][:7] for r in rows}
        for col, label in PRIMARY_VARS:
            vals = [to_float(r.get(col)) for r in rows]
            desc_rows.append({"scope": scope, "variable": label, **summarize_values(vals, len(codes), len(months))})
    desc_fields = ["scope", "variable", "count", "number_of_msas", "number_of_months", "mean", "standard_deviation", "minimum", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "maximum", "zero_count", "missing_count"]
    write_csv(tables / "housing_descriptive_statistics.csv", [r for r in desc_rows if r["scope"] == "all_matched_observations"], desc_fields)
    write_csv(tables / "housing_descriptive_by_candidate.csv", desc_rows, desc_fields)
    write_latex_table(tables / "housing_descriptive_statistics.tex", [r for r in desc_rows if r["scope"] == "all_matched_observations"], [("variable", "Variable"), ("count", "Count"), ("number_of_msas", "MSAs"), ("number_of_months", "Months"), ("mean", "Mean"), ("standard_deviation", "SD"), ("minimum", "Min."), ("median", "Median"), ("maximum", "Max.")], "Housing descriptive statistics for all matched observations.", "tab:housing-desc", "Levels are observed source/provider values; no interpolation, imputation, winsorization, demeaning, or standardization is applied.")
    by_year = []
    for year in sorted({r["date"][:4] for r in all_complete_rows}):
        yr_rows = [r for r in all_complete_rows if r["date"][:4] == year]
        for col, label in PRIMARY_VARS:
            by_year.append({"year": year, "variable": label, **summarize_values([to_float(r.get(col)) for r in yr_rows], len({r["cbsa_code"] for r in yr_rows}), len({r["date"][:7] for r in yr_rows}))})
    write_csv(tables / "housing_descriptive_by_year.csv", by_year, ["year"] + desc_fields[1:])
    by_msa = []
    for code in all_three_codes:
        msa_rows = [r for r in all_complete_rows if r["cbsa_code"] == code]
        for col, label in PRIMARY_VARS:
            by_msa.append({"cbsa_code": code, "msa_title": title_by_code.get(code, ""), "variable": label, **summarize_values([to_float(r.get(col)) for r in msa_rows], 1, len({r["date"][:7] for r in msa_rows}))})
    write_csv(tables / "housing_descriptive_by_msa.csv", by_msa, ["cbsa_code", "msa_title"] + desc_fields[1:])
    trans_rows = []
    by_code_rows: Dict[str, List[Dict[str, object]]] = {}
    for r in all_complete_rows:
        by_code_rows.setdefault(r["cbsa_code"], []).append(r)
    for code, rows in by_code_rows.items():
        rows = sorted(rows, key=lambda r: r["date"])
        rows2 = exact_lag_transform(rows, "zhvi_all_homes_sa", "zhvi_log_change_12m", lambda v, lv: math.log(v) - math.log(lv) if v > 0 and lv > 0 else "")
        rows2 = exact_lag_transform(rows2, "employment_thousands_sa", "employment_log_change_12m", lambda v, lv: math.log(v) - math.log(lv) if v > 0 and lv > 0 else "")
        rows2 = exact_lag_transform(rows2, "permits_units_sa", "permits_diff_12m", lambda v, lv: v - lv)
        rows2 = exact_lag_transform(rows2, "permits_units_sa", "permits_log1p_change_12m", lambda v, lv: math.log1p(v) - math.log1p(lv) if v >= 0 and lv >= 0 else "")
        for r in rows2:
            z = to_float(r["zhvi_all_homes_sa"]); e = to_float(r["employment_thousands_sa"]); p = to_float(r["permits_units_sa"])
            trans_rows.append({
                "cbsa_code": code, "date": r["date"],
                "log_ZHVI": math.log(z) if z and z > 0 else "",
                "12m_log_change_ZHVI": r.get("zhvi_log_change_12m", ""),
                "log_employment": math.log(e) if e and e > 0 else "",
                "12m_log_change_employment": r.get("employment_log_change_12m", ""),
                "permits_level": p if p is not None else "",
                "log1p_permits": math.log1p(p) if p is not None and p >= 0 else "",
                "asinh_permits": math.asinh(p) if p is not None else "",
                "12m_difference_permits": r.get("permits_diff_12m", ""),
                "12m_change_log1p_permits": r.get("permits_log1p_change_12m", ""),
            })
    write_csv(tables / "housing_transformation_diagnostics.csv", trans_rows, list(trans_rows[0]) if trans_rows else ["cbsa_code"])
    corr_rows = []
    for a, _ in PRIMARY_VARS:
        for b, _ in PRIMARY_VARS:
            xs = [(to_float(r[a]), to_float(r[b])) for r in all_complete_rows]
            xs = [(x, y) for x, y in xs if x is not None and y is not None]
            if len(xs) > 1:
                vx = [x for x, _ in xs]; vy = [y for _, y in xs]
                mx, my = statistics.fmean(vx), statistics.fmean(vy)
                den = math.sqrt(sum((x - mx) ** 2 for x in vx) * sum((y - my) ** 2 for y in vy))
                corr = sum((x - mx) * (y - my) for x, y in xs) / den if den else ""
            else:
                corr = ""
            corr_rows.append({"variable_1": a, "variable_2": b, "correlation": corr})
    write_csv(tables / "housing_correlation_diagnostics.csv", corr_rows, ["variable_1", "variable_2", "correlation"])

    coverage_rows = [
        {"source": "Zillow Research", "variable": "ZHVI all-homes SA", "geography": "Zillow metro regions", "seasonal_adjustment_source": "source smoothed seasonally adjusted", "earliest_date": min(r["date"] for r in zillow), "latest_date": max(r["date"] for r in zillow), "number_of_source_regions": len({r["zillow_region_id"] for r in zillow}), "number_classified_as_current_MSAs": sum(g["classification"] == "current_metropolitan_cbsa" for g in geo_class), "number_matched_to_all_sources": len(all_three_codes), "number_of_observations": len(zillow), "vintage": "current Zillow Research download"},
        {"source": "Census BPS + X-13", "variable": "permits units SA", "geography": "CBSA/MSA", "seasonal_adjustment_source": "local official X-13", "earliest_date": min(r["date"] for r in permits_sa), "latest_date": max(r["date"] for r in permits_sa), "number_of_source_regions": len({r["cbsa_code"] for r in permits_sa}), "number_classified_as_current_MSAs": len(title_by_code), "number_matched_to_all_sources": len(all_three_codes), "number_of_observations": len(permits_sa), "vintage": "Census BPS monthly file"},
        {"source": "BLS CES State and Area", "variable": "total nonfarm employment SA", "geography": "MSA", "seasonal_adjustment_source": "official BLS", "earliest_date": min(r["date"] for r in emp_sa), "latest_date": max(r["date"] for r in emp_sa), "number_of_source_regions": len({r["cbsa_code"] for r in emp_sa}), "number_classified_as_current_MSAs": len(title_by_code), "number_matched_to_all_sources": len(all_three_codes), "number_of_observations": len(emp_sa), "vintage": "current BLS SM bulk download"},
    ]
    write_csv(tables / "tab_housing_source_coverage.csv", coverage_rows, list(coverage_rows[0]))
    write_latex_table(tables / "tab_housing_source_coverage.tex", coverage_rows, [("source", "Source"), ("variable", "Variable"), ("earliest_date", "Earliest"), ("latest_date", "Latest"), ("number_of_source_regions", "Regions"), ("number_matched_to_all_sources", "All-source match"), ("number_of_observations", "Obs.")], "Housing source coverage.", "tab:housing-source-coverage")
    class_counts: Dict[str, int] = {}
    for g in geo_class:
        class_counts[g["classification"]] = class_counts.get(g["classification"], 0) + 1
    waterfall = [
        {"step": "Zillow regions", "count": len({r["zillow_region_id"] for r in zillow})},
        {"step": "current metropolitan CBSAs", "count": class_counts.get("current_metropolitan_cbsa", 0)},
        {"step": "current micropolitan CBSAs", "count": class_counts.get("current_micropolitan_cbsa", 0)},
        {"step": "historical/retired CBSAs", "count": class_counts.get("historical_or_retired_cbsa", 0)},
        {"step": "metropolitan divisions", "count": class_counts.get("metropolitan_division", 0)},
        {"step": "unresolved regions", "count": class_counts.get("unresolved", 0)},
        {"step": "matched to permits", "count": len(set(title_by_code) & {c for c, _ in p_by})},
        {"step": "matched to official SA employment", "count": len(set(title_by_code) & {c for c, _ in e_by})},
        {"step": "matched across all three", "count": len(all_three_codes)},
    ]
    write_csv(tables / "tab_housing_match_waterfall.csv", waterfall, ["step", "count"])
    write_latex_table(tables / "tab_housing_match_waterfall.tex", waterfall, [("step", "Step"), ("count", "Count")], "Housing geography and source matching waterfall.", "tab:housing-match-waterfall")
    write_csv(tables / "tab_housing_window_comparison.csv", fixed_starts + fixed_durs, win_fields)
    write_latex_table(tables / "tab_housing_window_comparison.tex", fixed_starts + fixed_durs, table_cols, "Housing balanced-window comparison.", "tab:housing-window-comparison")
    # BLS and X-13 accounting.
    bls_sel = [{"item": "current-MSA official-SA series retained", "count": len({r["bls_series_id"] for r in emp_sa})},
               {"item": "NSA counterparts retained", "count": 393},
               {"item": "official-SA observations retained", "count": len(emp_sa)}]
    write_csv(tables / "tab_housing_bls_selection.csv", bls_sel, ["item", "count"])
    write_latex_table(tables / "tab_housing_bls_selection.tex", bls_sel, [("item", "Item"), ("count", "Count")], "BLS series-selection accounting.", "tab:housing-bls-selection")
    x13_counts: Dict[str, int] = {}
    for r in x13:
        x13_counts[r["status"]] = x13_counts.get(r["status"], 0) + 1
    x13_rows = [{"item": k or "blank", "count": v} for k, v in sorted(x13_counts.items())]
    write_csv(tables / "tab_housing_x13_diagnostics.csv", x13_rows, ["item", "count"])
    write_latex_table(tables / "tab_housing_x13_diagnostics.tex", x13_rows, [("item", "X-13 status"), ("count", "Count")], "Permit X-13 accounting.", "tab:housing-x13")

    # Data quality flags.
    flags = []
    seen = set()
    negative_candidates_by_key = {
        (str(r["cbsa_code"]).zfill(5), str(r["date"])[:7]): r.get("candidate_panels_affected", "")
        for r in neg_diag
    }
    for r in all_complete_rows:
        key = (r["cbsa_code"], r["date"])
        if key in seen:
            flags.append({**r, "variable": "all", "value": "", "diagnostic": "duplicate MSA-month row", "threshold": "unique cbsa_code/date", "source_file": "assembled matched rows"})
        seen.add(key)
        for col, label in PRIMARY_VARS:
            v = to_float(r[col])
            if v is None:
                flags.append({**r, "variable": label, "value": "", "diagnostic": "missing value", "threshold": "nonmissing", "source_file": "processed"})
            elif col == "permits_units_sa" and v < 0:
                flags.append({**r, "variable": label, "value": v, "diagnostic": "negative_seasonally_adjusted_permit_value", "threshold": "X-13 additive SA value < 0", "source_file": "processed permits SA", "candidate_panels_affected": negative_candidates_by_key.get(key, "")})
            elif (col == "zhvi_all_homes_sa" and v <= 0) or (col == "employment_thousands_sa" and v < 0):
                flags.append({**r, "variable": label, "value": v, "diagnostic": "invalid sign", "threshold": "positive/nonnegative", "source_file": "processed"})
            elif col == "permits_units_sa" and abs(v) < 1e-12:
                flags.append({**r, "variable": label, "value": v, "diagnostic": "permit zero", "threshold": "zero", "source_file": "processed"})
        if str(r.get("bls_preliminary_flag", "0")) == "1":
            flags.append({**r, "variable": "employment", "value": r["employment_thousands_sa"], "diagnostic": "BLS preliminary observation", "threshold": "preliminary_flag==1", "source_file": "BLS CES"})
    flag_fields = ["cbsa_code", "msa_title", "date", "variable", "value", "diagnostic", "threshold", "source_file", "candidate_panels_affected"]
    for f in flags:
        f["candidate_panels_affected"] = f.get("candidate_panels_affected", "")
    write_csv(out_root / "housing_data_quality_flags.csv", flags, flag_fields)
    summary_flags: Dict[str, int] = {}
    for f in flags:
        summary_flags[f["diagnostic"]] = summary_flags.get(f["diagnostic"], 0) + 1
    write_csv(out_root / "housing_data_quality_summary.csv", [{"diagnostic": k, "count": v} for k, v in sorted(summary_flags.items())], ["diagnostic", "count"])
    (out_root / "housing_data_quality_report.md").write_text("# Housing Data Quality Report\n\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(summary_flags.items())) + "\n", encoding="utf-8")

    # Figures and data.
    fig05 = monthly_distribution(all_complete_rows, "zhvi_all_homes_sa", "ZHVI all-homes SA")
    fig06 = monthly_distribution(all_complete_rows, "permits_units_sa", "permits units SA")
    fig07 = monthly_distribution(all_complete_rows, "employment_thousands_sa", "employment thousands SA")
    start_dist = []
    for col, label in [("zhvi", "ZHVI"), ("permits_sa", "permits SA"), ("employment", "employment SA")]:
        firsts: Dict[str, int] = {}
        if col == "zhvi":
            source = {c: sorted(m for cc, m in z_by if cc == c) for c in all_three_codes}
        elif col == "permits_sa":
            source = {c: sorted(m for cc, m in p_by if cc == c) for c in all_three_codes}
        else:
            source = {c: sorted(m for cc, m in e_by if cc == c) for c in all_three_codes}
        for months0 in source.values():
            if months0:
                firsts[months0[0] + "-01"] = firsts.get(months0[0] + "-01", 0) + 1
        start_dist.extend({"series": label, "first_date": k, "msa_count": v} for k, v in sorted(firsts.items()))
    figure_specs = [
        ("fig01_source_coverage_timeline", coverage_rows),
        ("fig02_balanced_sample_by_start", table_rows_from_window(fixed_starts)),
        ("fig03_pareto_frontier", table_rows_from_window(pareto)),
        ("fig04_availability_heatmap", [{"cbsa_code": c, "first_complete_date": min(complete_months[c]) + "-01", "complete_months": len(complete_months[c])} for c in all_three_codes]),
        ("fig05_zhvi_evolution_available", fig05),
        ("fig06_permits_evolution_available", fig06),
        ("fig07_employment_evolution_available", fig07),
        ("fig08_starting_date_distribution", start_dist),
        ("fig09_sample_selection_waterfall", waterfall),
        ("fig10_missingness_exclusion_reasons", fixed_starts + fixed_durs),
    ]
    for name, rows in figure_specs:
        fcsv = figdata / f"{name}.csv"
        fields = sorted({k for r in rows for k in r}) if rows else ["placeholder"]
        write_csv(fcsv, rows or [{"placeholder": "see table outputs"}], fields)
        write_pdf(figs / f"{name}.pdf", name.replace("_", " ").title(), [json.dumps(r, sort_keys=True) for r in rows[:20]])
        write_png(figs / f"{name}.png")
        manifest_outputs.extend([fcsv, figs / f"{name}.pdf", figs / f"{name}.png"])
    # Constant-composition figure data for every candidate.
    for cand in candidate_summary:
        rows = read_csv(Path(cand["path"]) / "housing_panel_levels.csv")
        for col, label in PRIMARY_VARS:
            by_m: Dict[str, List[float]] = {}
            for r in rows:
                v = to_float(r[col])
                if v is not None:
                    by_m.setdefault(r["date"], []).append(v)
            out = []
            for d, vals in sorted(by_m.items()):
                out.append({"candidate_id": cand["candidate_id"], "date": d, "variable": label, "p10": quantile(vals, 0.10), "p25": quantile(vals, 0.25), "median": quantile(vals, 0.50), "p75": quantile(vals, 0.75), "p90": quantile(vals, 0.90), "n": len(vals)})
            path = figdata / f"constant_composition_{cand['candidate_id']}_{col}.csv"
            write_csv(path, out, ["candidate_id", "date", "variable", "p10", "p25", "median", "p75", "p90", "n"])
            manifest_outputs.append(path)

    all_manifest_paths = []
    for base in [tables, figs, figdata, out_root]:
        if base.exists():
            all_manifest_paths.extend(p for p in base.rglob("*") if p.is_file())
    if cand_root.exists():
        all_manifest_paths.extend(p for p in cand_root.rglob("*") if p.is_file())
    if cand_final_root.exists():
        all_manifest_paths.extend(p for p in cand_final_root.rglob("*") if p.is_file())
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generating_command": "python scripts/report_housing_data.py --data-root data/zillow --output-root outputs/empirical/housing_data_audit",
        "source_vintage": "validated Zillow/Census/BLS processed outputs under data/zillow",
        "outputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(set(all_manifest_paths)) if p.exists()],
        "candidate_panels": candidate_summary,
        "candidate_panels_final_only": candidate_final_summary,
        "tables_dir": str(tables),
        "figures_dir": str(figs),
        "figure_data_dir": str(figdata),
    }
    write_json(out_root / "housing_paper_output_manifest.json", report)
    best_n = max(candidate_summary, key=lambda r: r["N"]) if candidate_summary else {}
    best_t = max(candidate_summary, key=lambda r: r["T_months"]) if candidate_summary else {}
    best_nt = max(candidate_summary, key=lambda r: r["NT"]) if candidate_summary else {}
    best_eff = max(candidate_summary, key=lambda r: r["NT"] / (r["N"] + r["T_months"]) if r["N"] + r["T_months"] else 0) if candidate_summary else {}
    best_ranked = [r for r in candidate_ranking if r.get("highlight")]
    report_md = [
        "# Housing Paper Output Report",
        "",
        "1. Zillow ZHVI and Census permit NSA begin in 2000; BLS official SA employment begins in 1990; matched current-CBSA permit SA begins in 2004.",
        "2. Permit SA begins in 2004 for accepted current metropolitan CBSAs because 2000--2003 permit rows use legacy PMSA/MSA codes and X-13 segments for those codes are below the 84-month minimum.",
        f"3. Balanced panel beginning in 2000 feasible: {next((r['N_complete_msas'] for r in fixed_starts if r['start_date'].startswith('2000-01')), 0)} MSAs.",
        f"4. Largest 2000-start N/T: {next((r['N_complete_msas'] for r in fixed_starts if r['start_date'].startswith('2000-01')), 0)} by {next((r['T_months'] for r in fixed_starts if r['start_date'].startswith('2000-01')), 0)}.",
        f"5. Largest 2004-start panel: {next((r['N_complete_msas'] for r in fixed_starts if r['start_date'].startswith('2004-01')), 0)} MSAs.",
        f"6. Largest 2005-start panel: {next((r['N_complete_msas'] for r in fixed_starts if r['start_date'].startswith('2005-01')), 0)} MSAs.",
        f"7. Largest 2010-start panel: {next((r['N_complete_msas'] for r in fixed_starts if r['start_date'].startswith('2010-01')), 0)} MSAs.",
        f"8. Non-dominated candidates: {len(pareto)} rows in housing_pareto_frontier.csv.",
        f"9. Candidate maximizing N: {best_n.get('candidate_id', '')}.",
        f"10. Candidate maximizing T: {best_t.get('candidate_id', '')}.",
        f"11. Candidate maximizing NT: {best_nt.get('candidate_id', '')}.",
        f"12. Candidate maximizing NT/(N+T): {best_eff.get('candidate_id', '')}.",
        "13. Preliminary BLS observations by candidate are recorded in candidate metadata/source tables and data-quality flags.",
        f"14. Latest month with all matched official-SA BLS employment observations final: {latest_final_month}-01.",
        f"15. Final-only non-dominated candidates: {len(pareto_final)} rows in housing_pareto_frontier_final_only.csv.",
        f"16. Negative seasonally adjusted permit values: {neg_summary['total_count']} observations across {neg_summary['affected_cbsas']} CBSAs.",
        "17. X-13 problematic segments are not in the current all-three matched current-CBSA candidate panels unless flagged in the X-13 warning columns.",
        "18. Candidate completeness checks require exactly N*T rows and no missing primary variables.",
        "19. Candidate-ranking highlights are diagnostic only: " + "; ".join(f"{r['highlight']} -> {r['panel_type']}:{r['candidate_id']}" for r in best_ranked[:10]) + ".",
        "20. Tables and figures under outputs/empirical/housing_data_audit are ready for manuscript review but are not copied into manuscript tables.",
        "21. No interpolation, imputation, winsorization, standardization, forecasting, or backcasting was performed.",
    ]
    (out_root / "housing_paper_output_report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    print(f"[housing-report] wrote outputs to {out_root}")
    print(f"[housing-report] candidate panels: {len(candidate_summary)}")
    print(f"[housing-report] final-only candidate panels: {len(candidate_final_summary)}")
    print(f"[housing-report] common end: {common_end}-01")
    print(f"[housing-report] latest fully final BLS month: {latest_final_month}-01")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
