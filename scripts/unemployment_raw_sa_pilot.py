#!/usr/bin/env python3
"""Official raw-data and seasonal-adjustment pilot for unemployment panels.

This script is deliberately isolated from the existing empirical estimator.  It
downloads or accepts official agency files, preserves raw bytes, runs an X-13
pilot for non-seasonally-adjusted covariates, and writes audit inventories under
``outputs/empirical/unemployment/raw_sa_pilot``.  It never interpolates,
backfills, forward-fills, winsorizes, standardizes, or runs the estimator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlrhcs.folds import make_folds  # noqa: E402
from dlrhcs.housing_data import (  # noqa: E402
    BPS_COLS,
    BPS_FOLDERS,
    CENSUS_BPS_PAGE,
    USER_AGENT,
    find_x13_executable,
    month_range,
    parse_bps_monthly_file,
    parse_float,
    read_bls_table,
    sha256_file,
)
from dlrhcs.paths import find_repo_root  # noqa: E402


OUTPUT_VERSION = "unemployment_raw_sa_pilot_v1"
LAUS_SA_PAGE = "https://www.bls.gov/lau/metrossa.htm"
LAUS_SA_TXT_URL = "https://www.bls.gov/web/metro/ssamatab1.txt"
BLS_SM_BASE = "https://download.bls.gov/pub/time.series/SM/"
CES_FILES = {
    "sm.area": "CES/SAE area metadata",
    "sm.seasonal": "CES/SAE seasonal metadata",
    "sm.industry": "CES/SAE industry metadata",
    "sm.data_type": "CES/SAE data-type metadata",
    "sm.series": "CES/SAE series metadata",
    "sm.data.56.TotalPrivate.Current": "CES/SAE total-private current data",
}
STARTS = ("2007-01", "2010-01", "2011-01")
TRANSFORMS = ("one_month_growth", "twelve_month_growth", "lagged_levels")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def month_index(ym: str) -> int:
    y, m = ym[:7].split("-")
    return int(y) * 12 + int(m) - 1


def month_from_index(idx: int) -> str:
    y, m0 = divmod(idx, 12)
    return f"{y:04d}-{m0 + 1:02d}"


def looks_like_denial_or_html(path: Path) -> bool:
    raw = path.read_bytes()[:1024].lstrip().lower()
    return raw.startswith(b"<!doctype") or raw.startswith(b"<html") or b"access denied" in raw


def validated_download(url: str, dest: Path, *, force: bool = False, min_size: int = 1) -> Dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size >= min_size and not looks_like_denial_or_html(dest):
        return {
            "url": url,
            "local_path": str(dest),
            "status": "accepted-existing",
            "size": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "error": "",
        }
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/plain, application/octet-stream, application/zip, */*",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            with part.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
        part.replace(dest)
        if dest.stat().st_size < min_size:
            raise RuntimeError(f"downloaded file is too small: {dest.stat().st_size} bytes")
        if looks_like_denial_or_html(dest):
            raise RuntimeError("downloaded body looks like HTML/access-denied page")
        return {
            "url": url,
            "local_path": str(dest),
            "status": "newly-downloaded",
            "size": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "error": "",
        }
    except Exception as exc:
        if part.exists():
            part.unlink()
        if dest.exists() and looks_like_denial_or_html(dest):
            dest.unlink()
        return {
            "url": url,
            "local_path": str(dest),
            "status": "download-failed",
            "size": "",
            "sha256": "",
            "error": str(exc),
        }


def copy_existing_official(src: Path, dest: Path, source_url: str, dataset: str) -> Dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return {
            "agency": "",
            "dataset": dataset,
            "url": source_url,
            "local_path": str(dest),
            "status": "missing-local-cache",
            "size": "",
            "sha256": "",
            "error": f"missing {src}",
        }
    if dest.exists() and sha256_file(dest) == sha256_file(src):
        status = "accepted-existing"
    else:
        shutil.copy2(src, dest)
        status = "copied-from-repo-official-cache"
    return {
        "agency": "",
        "dataset": dataset,
        "url": source_url,
        "local_path": str(dest),
        "status": status,
        "size": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "error": "",
    }


def parse_laus_sa_table(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Parse BLS LAUS Table 1 with a permissive fixed/text-table reader."""
    if not path.exists():
        return [], [{"stage": "laus_parse", "status": "missing", "message": str(path)}]
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if looks_like_denial_or_html(path):
        return [], [{"stage": "laus_parse", "status": "invalid_html", "message": str(path)}]
    rows: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        # Common format has an area code plus year/month/rate columns.  The
        # regex intentionally keys on the official numeric code, not names.
        code_m = re.search(r"\b(\d{5})\b", s)
        if not code_m:
            continue
        code = code_m.group(1)
        year_m = re.search(r"\b(19|20)\d{2}\b", s)
        mon_m = re.search(r"\b(M(?:0[1-9]|1[0-2])|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|0?[1-9]|1[0-2])\b", s, flags=re.I)
        nums = [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", s)]
        if not year_m or not mon_m or len(nums) < 2:
            continue
        year = int(year_m.group(0))
        mon_raw = mon_m.group(0)
        month = period_to_month(mon_raw)
        if month is None:
            continue
        # In Table 1 the unemployment rate is the final numeric field.
        rate = nums[-1]
        rows.append({
            "cbsa_code": code,
            "date": f"{year:04d}-{month:02d}-01",
            "unemployment_rate_sa": rate,
            "laus_source": path.name,
            "seasonal_adjustment_source": "official BLS smoothed SA metro table",
            "preliminary_flag": int("P" in s.upper().split()),
            "raw_line_number": lineno,
        })
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    duplicates = 0
    for row in rows:
        key = (str(row["cbsa_code"]), str(row["date"]))
        duplicates += int(key in by_key)
        by_key[key] = row
    if duplicates:
        warnings.append({"stage": "laus_parse", "status": "duplicate_keys_last_write_wins_for_audit", "count": duplicates})
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"])), warnings


def period_to_month(value: str) -> Optional[int]:
    value = str(value).strip()
    if re.fullmatch(r"M(0[1-9]|1[0-2])", value, flags=re.I):
        return int(value[1:])
    names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    if value.lower() in names:
        return names[value.lower()]
    try:
        m = int(value)
        return m if 1 <= m <= 12 else None
    except ValueError:
        return None


def parse_ces_metadata(raw_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], List[Dict[str, object]]]:
    _, area_rows = read_bls_table(raw_dir / "sm.area")
    _, series_rows = read_bls_table(raw_dir / "sm.series")
    area_names = {r.get("area_code", "").zfill(5): r.get("area_name", "") for r in area_rows}
    selected: Dict[str, Dict[str, str]] = {}
    for row in series_rows:
        area = str(row.get("area_code", "")).zfill(5)
        sid = row.get("series_id", "").strip()
        if not re.fullmatch(r"\d{5}", area) or area == "00000":
            continue
        if row.get("industry_code") != "05000000":
            continue
        if row.get("data_type_code") != "02":
            continue
        selected[sid] = {
            "series_id": sid,
            "cbsa_code": area,
            "area_name": area_names.get(area, ""),
            "seasonal": row.get("seasonal", "").upper(),
            "industry_code": row.get("industry_code", ""),
            "data_type_code": row.get("data_type_code", ""),
            "begin_year": row.get("begin_year", ""),
            "begin_period": row.get("begin_period", ""),
            "end_year": row.get("end_year", ""),
            "end_period": row.get("end_period", ""),
        }
    diagnostics = [{
        "dataset": "CES total-private average weekly hours",
        "series_count": len(selected),
        "source": "sm.series",
        "selected_by": "industry_code=05000000 and data_type_code=02",
        "seasonally_adjusted_series": sum(1 for r in selected.values() if r["seasonal"] == "S"),
        "not_seasonally_adjusted_series": sum(1 for r in selected.values() if r["seasonal"] == "U"),
    }]
    return selected, area_names, diagnostics


def parse_ces_data_file(path: Path, selected: Mapping[str, Mapping[str, str]]) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    _, data_rows = read_bls_table(path)
    out: List[Dict[str, object]] = []
    for raw in data_rows:
        sid = raw.get("series_id", "").strip()
        meta = selected.get(sid)
        if not meta:
            continue
        period = raw.get("period", "")
        if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
            continue
        value = parse_float(raw.get("value"))
        if value is None:
            continue
        foot = raw.get("footnote_codes", "")
        out.append({
            "cbsa_code": meta["cbsa_code"],
            "date": f"{int(raw['year']):04d}-{int(period[1:]):02d}-01",
            "ces_series_id": sid,
            "hours_raw": value,
            "seasonal": meta["seasonal"],
            "source_seasonally_adjusted": int(meta["seasonal"] == "S"),
            "preliminary_flag": int("P" in str(foot).upper()),
            "footnote_codes": foot,
            "area_name": meta["area_name"],
        })
    return sorted(out, key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"]))


def fetch_ces_api_fallback(selected: Mapping[str, Mapping[str, str]], raw_dir: Path,
                           start_year: int = 2007, end_year: Optional[int] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fetch selected CES series through the official BLS API and preserve JSON bytes."""
    end_year = end_year or time.gmtime().tm_year
    api_dir = raw_dir / "bls_api_ces_total_private_hours"
    api_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(selected)
    diagnostics: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    year_chunks: List[Tuple[int, int]] = []
    y = int(start_year)
    while y <= int(end_year):
        # The unauthenticated BLS API documents a ten-year window limit; keep
        # chunks at ten years so responses are complete rather than silently
        # truncated to the first decade.
        yy = min(y + 9, int(end_year))
        year_chunks.append((y, yy))
        y = yy + 1
    for ys, ye in year_chunks:
        for chunk_i, start in enumerate(range(0, len(ids), 25), start=1):
            chunk = ids[start:start + 25]
            dest = api_dir / f"ces_hours_{ys}_{ye}_chunk{chunk_i:03d}.json"
            payload = json.dumps({"seriesid": chunk, "startyear": str(ys), "endyear": str(ye)}).encode("utf-8")
            status = "accepted-existing"
            if not dest.exists() or looks_like_denial_or_html(dest):
                status = "newly-downloaded"
                req = urllib.request.Request(
                    "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = resp.read()
                obj_new = json.loads(body.decode("utf-8"))
                tmp = dest.with_name(dest.name + ".part")
                if obj_new.get("status") == "REQUEST_SUCCEEDED":
                    tmp.write_bytes(body)
                    tmp.replace(dest)
                else:
                    failed = dest.with_suffix(dest.suffix + ".failed")
                    tmp.write_bytes(body)
                    tmp.replace(failed)
                    dest = failed
            obj = json.loads(dest.read_text(encoding="utf-8"))
            diagnostics.append({
                "source_file": dest.name,
                "series_requested": len(chunk),
                "start_year": ys,
                "end_year": ye,
                "status": obj.get("status", ""),
                "message": "; ".join(obj.get("message", []) or []),
                "local_path": str(dest),
                "sha256": sha256_file(dest),
                "download_status": status,
            })
            rows.extend(parse_ces_api_response(obj, selected))
    # Preserve and reuse any successful official API responses already cached
    # under this pilot root.  This is important when a later rerun hits the BLS
    # daily unauthenticated quota after earlier successful responses were saved.
    for cached in sorted(api_dir.glob("ces_hours_*.json")):
        try:
            obj = json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("status") == "REQUEST_SUCCEEDED":
            rows.extend(parse_ces_api_response(obj, selected))
            diagnostics.append({
                "source_file": cached.name,
                "series_requested": "",
                "start_year": "",
                "end_year": "",
                "status": "REQUEST_SUCCEEDED",
                "message": "parsed preserved successful cached official BLS API response",
                "local_path": str(cached),
                "sha256": sha256_file(cached),
                "download_status": "accepted-existing-successful-cache",
            })
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        by_key[(str(row["cbsa_code"]), str(row["date"]), str(row["ces_series_id"]))] = row
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"])), diagnostics


def parse_ces_api_response(obj: Mapping[str, object], selected: Mapping[str, Mapping[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if obj.get("status") != "REQUEST_SUCCEEDED":
        return rows
    for series in (obj.get("Results", {}) or {}).get("series", []):
        sid = series.get("seriesID", "")
        meta = selected.get(sid)
        if not meta:
            continue
        for item in series.get("data", []):
            period = item.get("period", "")
            if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                continue
            val = parse_float(item.get("value"))
            if val is None:
                continue
            footnotes = item.get("footnotes", []) or []
            codes = "".join(str(f.get("code", "")) for f in footnotes if f.get("code"))
            rows.append({
                "cbsa_code": meta["cbsa_code"],
                "date": f"{int(item['year']):04d}-{int(period[1:]):02d}-01",
                "ces_series_id": sid,
                "hours_raw": val,
                "seasonal": meta["seasonal"],
                "source_seasonally_adjusted": int(meta["seasonal"] == "S"),
                "preliminary_flag": int("P" in codes.upper() or str(item.get("latest", "")).lower() == "true"),
                "footnote_codes": codes,
                "area_name": meta["area_name"],
            })
    return rows


def parse_bps_raw_dir(raw_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    out: List[Dict[str, object]] = []
    diag: List[Dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.txt")):
        if not re.match(r"(ma|cbsa)\d{4}c\.txt$", path.name.lower()):
            continue
        rows, micro = parse_bps_monthly_file(path.read_bytes(), path.name)
        out.extend(rows)
        diag.append({
            "source_file": path.name,
            "rows": len(rows),
            "micropolitan_rows_excluded": micro,
            "sha256": sha256_file(path),
        })
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    duplicate_count = 0
    for row in out:
        key = (str(row["cbsa_code"]), str(row["date"]))
        duplicate_count += int(key in by_key)
        if key not in by_key:
            by_key[key] = row
        else:
            by_key[key]["total_units"] = float(by_key[key]["total_units"]) + float(row["total_units"])
    diag.append({"source_file": "ALL", "rows": len(by_key), "duplicate_code_months_combined": duplicate_count})
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"])), diag


def contiguous_observed_segments(rows: Sequence[Mapping[str, object]], value_col: str) -> List[List[Tuple[str, float]]]:
    by_month = {str(r["date"])[:7]: parse_float(r.get(value_col)) for r in rows}
    months = sorted(m for m, v in by_month.items() if v is not None)
    if not months:
        return []
    segments: List[List[Tuple[str, float]]] = []
    cur: List[Tuple[str, float]] = []
    prev = None
    for m in months:
        if prev is not None and month_index(m) != month_index(prev) + 1:
            if cur:
                segments.append(cur)
            cur = []
        cur.append((m + "-01", float(by_month[m])))
        prev = m
    if cur:
        segments.append(cur)
    return segments


def _read_x13_saved_values(path: Path, n: int) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").splitlines():
        for token in line.split():
            val = parse_float(token)
            if val is not None:
                vals.append(float(val))
    return vals[-n:] if len(vals) >= n else []


def run_x13_series(
    exe: Path,
    variable: str,
    cbsa: str,
    title: str,
    segment: Sequence[Tuple[str, float]],
    mode: str,
    work_dir: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    y, m = segment[0][0][:7].split("-")
    stem = f"{variable}_{mode}_{cbsa}_{segment[0][0][:7].replace('-', '')}_{segment[-1][0][:7].replace('-', '')}"
    spc = work_dir / f"{stem}.spc"
    transform = "log" if mode == "log_additive" else "none"
    values = "\n".join(f"{float(v):.12g}" for _, v in segment)
    spec = f"""series {{
  title = "{variable} {cbsa}"
  start = {int(y)}.{int(m)}
  period = 12
  data = (
{values}
  )
}}
transform {{ function = {transform} }}
regression {{ aictest = (td easter) }}
automdl {{}}
x11 {{ mode = add save = (d10 d11) }}
"""
    spc.write_text(spec, encoding="utf-8")
    d10 = work_dir / f"{stem}.d10"
    d11 = work_dir / f"{stem}.d11"
    if d11.exists():
        proc_returncode, proc_stdout, proc_stderr = 0, "", ""
        reused_output = True
    else:
        proc = subprocess.run([str(exe), str(spc.with_suffix(""))], cwd=work_dir,
                              capture_output=True, text=True, timeout=180, check=False)
        proc_returncode, proc_stdout, proc_stderr = proc.returncode, proc.stdout, proc.stderr
        reused_output = False
    adj = _read_x13_saved_values(d11, len(segment))
    fac = _read_x13_saved_values(d10, len(segment))
    success = proc_returncode == 0 and len(adj) == len(segment)
    out_rows: List[Dict[str, object]] = []
    for k, (date, raw) in enumerate(segment):
        out_rows.append({
            "variable": variable,
            "cbsa_code": cbsa,
            "area_title": title,
            "date": date,
            "x13_mode": mode,
            "raw_value": raw,
            "adjusted_value": adj[k] if success else "",
            "seasonal_factor": fac[k] if len(fac) == len(segment) else "",
            "x13_status": "ok" if success else f"failed:{proc_returncode}",
            "x13_spec_id": stem,
            "x13_spec_path": str(spc),
        })
    text = "\n".join([proc_stdout, proc_stderr])
    diag = {
        "variable": variable,
        "cbsa_code": cbsa,
        "area_title": title,
        "x13_mode": mode,
        "x13_spec_id": stem,
        "x13_status": "ok" if success else f"failed:{proc_returncode}",
        "returncode": proc_returncode,
        "reused_existing_x13_output": int(reused_output),
        "n_observed": len(segment),
        "segment_start": segment[0][0],
        "segment_end": segment[-1][0],
        "x13_spec_path": str(spc),
        "x13_transformation": transform,
        "x13_x11_mode": "add",
        "warning_or_error": int(bool(re.search(r"warning|error|fail", text, flags=re.I)) or not success),
        "warning_excerpt": "\n".join([ln for ln in text.splitlines() if re.search(r"warning|error|fail", ln, re.I)])[:2000],
        "outlier_excerpt": "\n".join([ln for ln in text.splitlines() if re.search(r"outlier|AO|LS|TC", ln)])[:2000],
        "model_diagnostics_excerpt": text[-2000:],
    }
    return out_rows, diag


def run_x13_pilot(ces_rows: Sequence[Mapping[str, object]], bps_rows: Sequence[Mapping[str, object]],
                  out_root: Path, min_months: int = 84) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    exe = find_x13_executable(ROOT / "data" / "zillow")
    if exe is None:
        diag = [{"variable": "", "cbsa_code": "", "x13_mode": "", "x13_status": "x13_executable_missing"}]
        return [], diag
    all_adjusted: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    work_dir = out_root / "x13_work"
    grouped_ces: Dict[str, List[Mapping[str, object]]] = {}
    for row in ces_rows:
        if int(row.get("source_seasonally_adjusted", 0)):
            continue
        grouped_ces.setdefault(str(row["cbsa_code"]).zfill(5), []).append(row)
    for cbsa, rows in sorted(grouped_ces.items()):
        for segment in contiguous_observed_segments(rows, "hours_raw"):
            if len(segment) < min_months:
                diagnostics.append({"variable": "ces_hours", "cbsa_code": cbsa, "x13_status": "segment_too_short", "n_observed": len(segment)})
                continue
            title = str(rows[0].get("area_name", ""))
            for mode in ("additive_level", "log_additive"):
                adjusted, diag = run_x13_series(exe, "ces_hours", cbsa, title, segment, mode, work_dir)
                all_adjusted.extend(adjusted)
                diagnostics.append(diag)
    grouped_bps: Dict[str, List[Mapping[str, object]]] = {}
    for row in bps_rows:
        grouped_bps.setdefault(str(row["cbsa_code"]).zfill(5), []).append(row)
    for cbsa, rows in sorted(grouped_bps.items()):
        for segment in contiguous_observed_segments(rows, "total_units"):
            if len(segment) < min_months:
                diagnostics.append({"variable": "bps_permits", "cbsa_code": cbsa, "x13_status": "segment_too_short", "n_observed": len(segment)})
                continue
            title = str(rows[0].get("cbsa_title", ""))
            adjusted, diag = run_x13_series(exe, "bps_permits", cbsa, title, segment, "additive_level", work_dir)
            all_adjusted.extend(adjusted)
            diagnostics.append(diag)
    return all_adjusted, diagnostics


def by_code_month(rows: Sequence[Mapping[str, object]], value_col: str, status_filter: Optional[Mapping[str, str]] = None) -> Dict[Tuple[str, str], Mapping[str, object]]:
    out = {}
    for r in rows:
        if status_filter:
            ok = True
            for k, v in status_filter.items():
                ok &= str(r.get(k, "")) == v
            if not ok:
                continue
        val = parse_float(r.get(value_col))
        if val is None:
            continue
        out[(str(r["cbsa_code"]).zfill(5), str(r["date"])[:7])] = r
    return out


def transformed_maps(adjusted: Sequence[Mapping[str, object]]) -> Dict[str, Dict[Tuple[str, str], float]]:
    hours_log: Dict[Tuple[str, str], float] = {}
    permits_asinh: Dict[Tuple[str, str], float] = {}
    for row in adjusted:
        if str(row.get("x13_status")) != "ok":
            continue
        key = (str(row["cbsa_code"]).zfill(5), str(row["date"])[:7])
        val = parse_float(row.get("adjusted_value"))
        if val is None:
            continue
        if row.get("variable") == "ces_hours" and row.get("x13_mode") == "log_additive" and val > 0:
            hours_log[key] = math.log(val)
        elif row.get("variable") == "bps_permits" and row.get("x13_mode") == "additive_level":
            permits_asinh[key] = math.asinh(val)
    return {"hours_log_sa": hours_log, "permits_asinh_sa": permits_asinh}


def lagged_value(values: Mapping[Tuple[str, str], float], code: str, month: str, lag: int) -> Optional[float]:
    m = month_from_index(month_index(month) - lag)
    return values.get((code, m))


def transform_value(values: Mapping[Tuple[str, str], float], code: str, month: str, kind: str, variable: str) -> Optional[float]:
    if kind == "lagged_levels":
        return values.get((code, month))
    if kind == "one_month_growth":
        now = values.get((code, month))
        old = lagged_value(values, code, month, 1)
        if now is None or old is None:
            return None
        return 100.0 * (now - old) if variable == "hours" else now - old
    if kind == "twelve_month_growth":
        now = values.get((code, month))
        old = lagged_value(values, code, month, 12)
        if now is None or old is None:
            return None
        return 100.0 * (now - old) if variable == "hours" else now - old
    raise ValueError(kind)


def condition_diagnostics(x1: np.ndarray, x2: np.ndarray) -> Dict[str, object]:
    X = np.column_stack([x1, x2]).astype(float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if X.shape[0] < 3:
        return {"pooled_correlation": "", "vif": "", "auxiliary_r2": "", "gram_min_eigenvalue": "", "gram_max_eigenvalue": "", "condition_number": ""}
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) if np.std(X[:, 0]) > 0 and np.std(X[:, 1]) > 0 else float("nan")
    r2 = corr * corr if math.isfinite(corr) else float("nan")
    Xm = X - X.mean(axis=0)
    gram = (Xm.T @ Xm) / max(Xm.shape[0], 1)
    eig = np.linalg.eigvalsh(gram)
    cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    return {
        "pooled_correlation": corr,
        "vif": 1.0 / (1.0 - r2) if math.isfinite(r2) and r2 < 1.0 else float("inf"),
        "auxiliary_r2": r2,
        "gram_min_eigenvalue": float(eig[0]),
        "gram_max_eigenvalue": float(eig[-1]),
        "condition_number": cond,
    }


def two_way_demeaned_corr(panel_rows: Sequence[Mapping[str, object]]) -> float:
    vals = [(r["cbsa_code"], r["date"][:7], float(r["hours_x"]), float(r["permits_x"])) for r in panel_rows]
    if len(vals) < 3:
        return float("nan")
    codes = sorted({v[0] for v in vals})
    months = sorted({v[1] for v in vals})
    ci = {c: i for i, c in enumerate(codes)}
    mi = {m: i for i, m in enumerate(months)}
    A = np.full((len(codes), len(months), 2), np.nan)
    for c, m, h, p in vals:
        A[ci[c], mi[m], :] = [h, p]
    overall = np.nanmean(A, axis=(0, 1))
    code_mean = np.nanmean(A, axis=1, keepdims=True)
    month_mean = np.nanmean(A, axis=0, keepdims=True)
    D = A - code_mean - month_mean + overall
    x = D[:, :, 0].ravel()
    y = D[:, :, 1].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def within_msa_corr_summary(panel_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for r in panel_rows:
        grouped.setdefault(str(r["cbsa_code"]), []).append(r)
    vals = []
    for rows in grouped.values():
        x = np.array([float(r["hours_x"]) for r in rows], float)
        y = np.array([float(r["permits_x"]) for r in rows], float)
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "within_msa_corr_mean": float(np.mean(vals)) if vals else "",
        "within_msa_corr_min": float(np.min(vals)) if vals else "",
        "within_msa_corr_max": float(np.max(vals)) if vals else "",
    }


def fold_condition_diagnostics(panel_rows: Sequence[Mapping[str, object]], q: int = 1, J: int = 10) -> Dict[str, object]:
    codes = sorted({str(r["cbsa_code"]) for r in panel_rows})
    months = sorted({str(r["date"])[:7] for r in panel_rows})
    if not codes or not months:
        return {"fold_condition_min": "", "fold_condition_median": "", "fold_condition_max": ""}
    ci = {c: i for i, c in enumerate(codes)}
    mi = {m: i for i, m in enumerate(months)}
    X = np.full((len(months), len(codes), 2), np.nan)
    for r in panel_rows:
        X[mi[str(r["date"])[:7]], ci[str(r["cbsa_code"])], :] = [float(r["hours_x"]), float(r["permits_x"])]
    folds = make_folds(len(months), len(codes), min(J, len(months) * len(codes)), q, r=0, rng=np.random.default_rng(123))
    conds = []
    for f in folds:
        z = X[f.train]
        z = z[np.isfinite(z).all(axis=1)]
        if z.shape[0] < 3:
            continue
        d = condition_diagnostics(z[:, 0], z[:, 1])
        c = parse_float(d.get("condition_number"))
        if c is not None and math.isfinite(c):
            conds.append(c)
    return {
        "fold_condition_min": float(np.min(conds)) if conds else "",
        "fold_condition_median": float(np.median(conds)) if conds else "",
        "fold_condition_max": float(np.max(conds)) if conds else "",
    }


def build_candidate_inventories(
    laus_rows: Sequence[Mapping[str, object]],
    adjusted_rows: Sequence[Mapping[str, object]],
    x13_diag: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    laus = by_code_month(laus_rows, "unemployment_rate_sa")
    maps = transformed_maps(adjusted_rows)
    hours_map = maps["hours_log_sa"]
    permits_map = maps["permits_asinh_sa"]
    x13_failed_codes = {
        str(d.get("cbsa_code", "")).zfill(5) for d in x13_diag
        if str(d.get("x13_status", "")).startswith("failed") or str(d.get("x13_status", "")).startswith("segment_too_short")
    }
    months_by_code: Dict[str, List[str]] = {}
    for code, month in laus:
        months_by_code.setdefault(code, []).append(month)
    all_codes = sorted({c for c, _ in laus} | {c for c, _ in hours_map} | {c for c, _ in permits_map})
    inventory: List[Dict[str, object]] = []
    panel_rows_all: List[Dict[str, object]] = []
    col_rows: List[Dict[str, object]] = []
    for start in STARTS:
        for kind in TRANSFORMS:
            candidate_rows: List[Dict[str, object]] = []
            dropped: List[Dict[str, object]] = []
            latest_candidates = [
                max([m for _, m in laus] or [start]),
                max([m for _, m in hours_map] or [start]),
                max([m for _, m in permits_map] or [start]),
            ]
            latest = min(latest_candidates)
            raw_months = month_range(start, latest)
            transform_lag = {"lagged_levels": 0, "one_month_growth": 1, "twelve_month_growth": 12}[kind]
            usable_months = raw_months[transform_lag + 1:]  # one-month model lag after transformations
            for code in all_codes:
                missing_reasons = []
                code_rows = []
                for m in usable_months:
                    y = laus.get((code, m))
                    hx = transform_value(hours_map, code, month_from_index(month_index(m) - 1), kind, "hours")
                    px = transform_value(permits_map, code, month_from_index(month_index(m) - 1), kind, "permits")
                    if y is None or hx is None or px is None:
                        if y is None:
                            missing_reasons.append("missing_laus_sa")
                        if hx is None:
                            missing_reasons.append("missing_transformed_hours_after_lag")
                        if px is None:
                            missing_reasons.append("missing_transformed_permits_after_lag")
                        continue
                    code_rows.append({
                        "candidate_start": start,
                        "transformation": kind,
                        "cbsa_code": code,
                        "date": m + "-01",
                        "unemployment_rate_sa": y["unemployment_rate_sa"],
                        "hours_x": hx,
                        "permits_x": px,
                        "model_lag_applied": 1,
                    })
                if len(code_rows) == len(usable_months) and code_rows:
                    candidate_rows.extend(code_rows)
                else:
                    dropped.append({
                        "candidate_start": start,
                        "transformation": kind,
                        "cbsa_code": code,
                        "dropped_reason": ";".join(sorted(set(missing_reasons))) or "no_usable_rows",
                    })
            codes_kept = sorted({r["cbsa_code"] for r in candidate_rows})
            missing_obs = len(all_codes) * len(usable_months) - len(candidate_rows)
            inv = {
                "candidate_start": start,
                "transformation": kind,
                "common_msas": len(codes_kept),
                "raw_months": len(raw_months),
                "usable_months_after_transformations_and_one_month_lag": len(usable_months) if codes_kept else 0,
                "missing_observations": missing_obs,
                "preliminary_observations": sum(int(r.get("preliminary_flag", 0)) for r in laus_rows if str(r.get("cbsa_code", "")).zfill(5) in codes_kept),
                "x13_failures": len(x13_failed_codes & set(codes_kept)),
                "geography_mismatches": len([d for d in dropped if "missing_laus_sa" in d["dropped_reason"]]),
                "dropped_msas": len({d["cbsa_code"] for d in dropped}),
            }
            if candidate_rows:
                x1 = np.array([float(r["hours_x"]) for r in candidate_rows], float)
                x2 = np.array([float(r["permits_x"]) for r in candidate_rows], float)
                inv.update(condition_diagnostics(x1, x2))
                inv["two_way_demeaned_correlation"] = two_way_demeaned_corr(candidate_rows)
                inv.update(within_msa_corr_summary(candidate_rows))
                inv.update(fold_condition_diagnostics(candidate_rows))
            inventory.append(inv)
            panel_rows_all.extend(candidate_rows)
            for d in dropped:
                col_rows.append(d)
    return inventory, panel_rows_all, col_rows


def write_latex_simple(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[Tuple[str, str]], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(h for _, h in columns) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for key, _ in columns:
            v = row.get(key, "")
            if isinstance(v, float):
                if math.isfinite(v):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append("--")
            else:
                vals.append(str(v) if str(v) else "--")
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-root", default="outputs/empirical/unemployment/raw_sa_pilot")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--min-x13-months", type=int, default=84)
    args = ap.parse_args(argv)

    root = find_repo_root(Path(args.repo_root).resolve())
    out_root = root / args.out_root
    raw_root = out_root / "raw"
    tables = out_root / "tables"
    raw_root.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    manifest.append({
        "agency": "U.S. Bureau of Labor Statistics",
        "dataset": "LAUS smoothed seasonally adjusted metro Table 1",
        "official_source_page": LAUS_SA_PAGE,
        **validated_download(LAUS_SA_TXT_URL, raw_root / "bls_laus" / "ssamatab1.txt", force=args.force_download, min_size=1000),
        "seasonal_adjustment_status": "official smoothed seasonally adjusted; no local adjustment",
    })
    repo_bls = root / "data" / "zillow" / "raw" / "bls_ces"
    for filename, dataset in CES_FILES.items():
        src = repo_bls / filename
        dest = raw_root / "bls_ces" / filename
        if filename == "sm.data.56.TotalPrivate.Current":
            rec = validated_download(urllib.parse.urljoin(BLS_SM_BASE, filename), dest,
                                     force=args.force_download, min_size=1000)
            if rec["status"] == "download-failed":
                rec = copy_existing_official(src, dest, urllib.parse.urljoin(BLS_SM_BASE, filename), dataset)
        else:
            rec = copy_existing_official(src, dest, urllib.parse.urljoin(BLS_SM_BASE, filename), dataset)
        rec.update({"agency": "U.S. Bureau of Labor Statistics", "official_source_page": "https://www.bls.gov/sae/", "seasonal_adjustment_status": "metadata-defined"})
        manifest.append(rec)
    bps_src = root / "data" / "zillow" / "raw" / "census_bps"
    bps_dest = raw_root / "census_bps"
    bps_dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(bps_src.glob("*.txt")):
        if re.match(r"(ma|cbsa)\d{4}c\.txt$", src.name.lower()):
            dest = bps_dest / src.name
            if not dest.exists() or sha256_file(dest) != sha256_file(src):
                shutil.copy2(src, dest)
            manifest.append({
                "agency": "U.S. Census Bureau",
                "dataset": "BPS monthly MSA/CBSA permits",
                "official_source_page": CENSUS_BPS_PAGE,
                "url": "repo official raw cache",
                "local_path": str(dest),
                "status": "copied-from-repo-official-cache",
                "size": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "seasonal_adjustment_status": "not seasonally adjusted",
                "error": "",
            })

    laus_rows, laus_warnings = parse_laus_sa_table(raw_root / "bls_laus" / "ssamatab1.txt")
    ces_meta: Dict[str, Dict[str, str]] = {}
    ces_rows: List[Dict[str, object]] = []
    ces_meta_diag: List[Dict[str, object]] = []
    try:
        ces_meta, _, ces_meta_diag = parse_ces_metadata(raw_root / "bls_ces")
        ces_rows = parse_ces_data_file(raw_root / "bls_ces" / "sm.data.56.TotalPrivate.Current", ces_meta)
        if not ces_rows and ces_meta:
            api_rows, api_diag = fetch_ces_api_fallback(ces_meta, raw_root)
            ces_rows = api_rows
            ces_meta_diag.extend(api_diag)
            for d in api_diag:
                manifest.append({
                    "agency": "U.S. Bureau of Labor Statistics",
                    "dataset": "CES/SAE total-private average weekly hours API response",
                    "official_source_page": "https://www.bls.gov/developers/",
                    "url": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                    "local_path": d.get("local_path", ""),
                    "status": d.get("download_status", ""),
                    "size": Path(str(d.get("local_path", ""))).stat().st_size if d.get("local_path") else "",
                    "sha256": d.get("sha256", ""),
                    "seasonal_adjustment_status": "metadata-defined",
                    "error": d.get("message", ""),
                })
    except Exception as exc:
        ces_meta_diag.append({"dataset": "CES total-private average weekly hours", "status": "parse_failed", "error": str(exc)})
    bps_rows, bps_diag = parse_bps_raw_dir(raw_root / "census_bps")
    x13_rows, x13_diag = run_x13_pilot(ces_rows, bps_rows, out_root, min_months=args.min_x13_months)
    inventory, panel_rows, dropped_rows = build_candidate_inventories(laus_rows, x13_rows, x13_diag)

    write_csv(out_root / "raw_download_manifest.csv", manifest,
              ["agency", "dataset", "official_source_page", "url", "local_path", "status", "size", "sha256", "seasonal_adjustment_status", "error"])
    write_csv(out_root / "laus_sa_long.csv", laus_rows,
              ["cbsa_code", "date", "unemployment_rate_sa", "seasonal_adjustment_source", "preliminary_flag", "laus_source", "raw_line_number"])
    write_csv(out_root / "ces_hours_raw_long.csv", ces_rows,
              ["cbsa_code", "date", "ces_series_id", "hours_raw", "seasonal", "source_seasonally_adjusted", "preliminary_flag", "footnote_codes", "area_name"])
    write_csv(out_root / "bps_permits_raw_long.csv", bps_rows,
              ["cbsa_code", "date", "total_units", "cbsa_title", "source_file", "source_vintage"])
    write_csv(out_root / "x13_adjusted_long.csv", x13_rows,
              ["variable", "cbsa_code", "area_title", "date", "x13_mode", "raw_value", "adjusted_value", "seasonal_factor", "x13_status", "x13_spec_id", "x13_spec_path"])
    write_csv(out_root / "x13_diagnostics.csv", x13_diag,
              ["variable", "cbsa_code", "area_title", "x13_mode", "x13_spec_id", "x13_status", "returncode", "reused_existing_x13_output", "n_observed", "segment_start", "segment_end", "x13_transformation", "x13_x11_mode", "warning_or_error", "warning_excerpt", "outlier_excerpt", "model_diagnostics_excerpt"])
    write_csv(out_root / "common_panel_inventory.csv", inventory,
              ["candidate_start", "transformation", "common_msas", "raw_months", "usable_months_after_transformations_and_one_month_lag", "missing_observations", "preliminary_observations", "x13_failures", "geography_mismatches", "dropped_msas"])
    write_csv(out_root / "candidate_panels_long.csv", panel_rows,
              ["candidate_start", "transformation", "cbsa_code", "date", "unemployment_rate_sa", "hours_x", "permits_x", "model_lag_applied"])
    write_csv(out_root / "dropped_msas_reasons.csv", dropped_rows,
              ["candidate_start", "transformation", "cbsa_code", "dropped_reason"])
    col_fields = [
        "candidate_start", "transformation", "common_msas", "usable_months_after_transformations_and_one_month_lag",
        "pooled_correlation", "two_way_demeaned_correlation", "within_msa_corr_mean", "within_msa_corr_min",
        "within_msa_corr_max", "vif", "auxiliary_r2", "gram_min_eigenvalue", "gram_max_eigenvalue",
        "condition_number", "fold_condition_min", "fold_condition_median", "fold_condition_max",
    ]
    write_csv(out_root / "collinearity_audit.csv", inventory, col_fields)
    write_csv(tables / "tab_unemp_raw_inventory.csv", [
        {"source": "LAUS SA", "rows": len(laus_rows), "codes": len({r["cbsa_code"] for r in laus_rows}), "months": len({r["date"][:7] for r in laus_rows}), "status": manifest[0]["status"]},
        {"source": "CES hours", "rows": len(ces_rows), "codes": len({r["cbsa_code"] for r in ces_rows}), "months": len({r["date"][:7] for r in ces_rows}), "status": "parsed" if ces_rows else "missing"},
        {"source": "BPS permits", "rows": len(bps_rows), "codes": len({r["cbsa_code"] for r in bps_rows}), "months": len({r["date"][:7] for r in bps_rows}), "status": "parsed" if bps_rows else "missing"},
        {"source": "X-13 adjusted covariates", "rows": len(x13_rows), "codes": len({r["cbsa_code"] for r in x13_rows}), "months": len({r["date"][:7] for r in x13_rows}), "status": "parsed" if x13_rows else "missing"},
    ], ["source", "rows", "codes", "months", "status"])
    write_csv(tables / "tab_unemp_candidate_inventory.csv", inventory,
              ["candidate_start", "transformation", "common_msas", "raw_months", "usable_months_after_transformations_and_one_month_lag", "missing_observations", "preliminary_observations", "x13_failures", "geography_mismatches", "dropped_msas"])
    write_csv(tables / "tab_unemp_collinearity.csv", inventory, col_fields)
    write_latex_simple(tables / "tab_unemp_raw_inventory.tex", read_csv_rows(tables / "tab_unemp_raw_inventory.csv"),
                       [("source", "Source"), ("rows", "Rows"), ("codes", "MSAs"), ("months", "Months"), ("status", "Status")],
                       "Unemployment raw-data inventory.", "tab:unemp-raw-inventory")
    write_latex_simple(tables / "tab_unemp_candidate_inventory.tex", inventory,
                       [("candidate_start", "Start"), ("transformation", "Transform"), ("common_msas", "MSAs"), ("raw_months", "Raw months"), ("usable_months_after_transformations_and_one_month_lag", "Usable months"), ("missing_observations", "Missing")],
                       "Unemployment common-panel inventory.", "tab:unemp-candidate-inventory")
    write_latex_simple(tables / "tab_unemp_collinearity.tex", inventory,
                       [("candidate_start", "Start"), ("transformation", "Transform"), ("pooled_correlation", "Pooled corr."), ("two_way_demeaned_correlation", "TW corr."), ("vif", "VIF"), ("condition_number", "Cond.")],
                       "Unemployment covariate collinearity audit.", "tab:unemp-collinearity")
    write_json(out_root / "pilot_metadata.json", {
        "output_version": OUTPUT_VERSION,
        "created_utc": utc_now(),
        "requirements": {
            "monthly_frequency": True,
            "five_digit_cbsa_codes": True,
            "name_matching_primary_merge": False,
            "interpolation": False,
            "backfill": False,
            "forward_fill": False,
            "winsorization": False,
            "standardization": False,
            "estimator_run": False,
        },
        "laus_parse_warnings": laus_warnings,
        "ces_metadata_diagnostics": ces_meta_diag,
        "bps_parse_diagnostics": bps_diag[-5:],
        "x13_executable": str(find_x13_executable(ROOT / "data" / "zillow") or ""),
        "x13_policy": {
            "ces_hours": ["additive adjustment on level", "log-additive adjustment"],
            "bps_permits": ["additive adjustment on raw permit count"],
            "manual_msa_tuning": False,
        },
    })
    print(f"raw-data pilot written to {out_root}")
    print(f"LAUS rows={len(laus_rows)} CES rows={len(ces_rows)} BPS rows={len(bps_rows)} X13 rows={len(x13_rows)}")
    if not laus_rows:
        print("WARNING: LAUS SA file was not parsed; see raw_download_manifest.csv and pilot_metadata.json.")
    return 0 if laus_rows and ces_rows and bps_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
