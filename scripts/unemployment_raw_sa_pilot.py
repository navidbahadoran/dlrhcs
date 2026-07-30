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
import tempfile
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
from dlrhcs.paths import find_repo_root, repo_relative  # noqa: E402


OUTPUT_VERSION = "unemployment_raw_sa_pilot_v1"
LAUS_SA_PAGE = "https://www.bls.gov/lau/metrossa.htm"
LAUS_SA_TXT_URL = "https://www.bls.gov/web/metro/ssamatab1.txt"
BLS_SM_BASE = "https://download.bls.gov/pub/time.series/SM/"
BLS_SM_LOWER_BASE = "https://download.bls.gov/pub/time.series/sm/"
CES_HISTORY_FILES = (
    "sm.txt",
    "sm.data.1.AllData",
    "sm.series",
    "sm.area",
    "sm.industry",
    "sm.data_type",
    "sm.seasonal",
    "sm.period",
    "sm.footnote",
    "sm.data.56.TotalPrivate.Current",
)
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
CES_HISTORY_VERSION = "ces_history_completion_v1"
CES_FINAL_SOURCE_VERSION = "ces_final_source_validation_v1"
CES_FINAL_REQUIRED_METADATA_FILES = (
    "sm.series",
    "sm.area",
    "sm.industry",
    "sm.data_type",
    "sm.seasonal",
    "sm.footnote",
)
CES_FINAL_OPTIONAL_MAPPING_FILES = ("sm.period", "sm.txt")
CANONICAL_CES_MONTHLY_PERIODS = {f"M{i:02d}": i for i in range(1, 13)}
CANONICAL_CES_ANNUAL_PERIOD = "M13"


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
    if looks_like_denial_or_html(path):
        raise ValueError(f"{path.name} appears to be HTML or an access-denied page, not an official BLS tab-delimited file")
    def append_cells(cells: Sequence[str], lineno: object) -> None:
        if len(cells) == len(header) - 1 and optional_last:
            cells = [*cells, ""]
        if len(cells) != len(header):
            raise ValueError(
                f"{path.name}: malformed tab-delimited row at line {lineno}: "
                f"expected {len(header)} fields, observed {len(cells)}"
            )
        period_kind, month = validate_ces_period_code(cells[index["period"]])
        if period_kind == "annual_average":
            return
        sid = cells[index["series_id"]].strip()
        meta = selected.get(sid)
        if not meta:
            return
        value = parse_float(cells[index["value"]])
        if value is None:
            return
        foot = cells[index["footnote_codes"]].strip() if "footnote_codes" in index else ""
        out.append({
            "cbsa_code": meta["cbsa_code"],
            "date": f"{int(cells[index['year']].strip()):04d}-{int(month or 0):02d}-01",
            "ces_series_id": sid,
            "hours_raw": value,
            "seasonal": meta["seasonal"],
            "source_seasonally_adjusted": int(meta["seasonal"] == "S"),
            "preliminary_flag": int("P" in str(foot).upper()),
            "footnote_codes": foot,
            "area_name": meta["area_name"],
            "source_file": path.name,
            "source_sha256": source_hash,
        })

    out: List[Dict[str, object]] = []
    source_hash = sha256_file(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        header_line = fh.readline()
    if not header_line:
        return []
    header = [p.strip() for p in header_line.rstrip("\r\n").split("\t")]
    index = {name: i for i, name in enumerate(header)}
    required = ("series_id", "year", "period", "value")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{path.name}: missing required CES data columns {missing}")
    optional_last = bool(header and header[-1].strip().lower() == "footnote_codes")

    rg_path = shutil.which("rg")
    if rg_path and selected:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False) as pf:
            pattern_path = Path(pf.name)
            for sid in sorted(selected):
                pf.write(sid + "\n")
                pf.write(sid.ljust(30) + "\n")
        try:
            proc = subprocess.run(
                [rg_path, "--fixed-strings", "--text", "--no-heading", "--file", str(pattern_path), str(path)],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        finally:
            try:
                pattern_path.unlink()
            except OSError:
                pass
        if proc.returncode not in {0, 1}:
            raise ValueError(f"{path.name}: ripgrep selected-row extraction failed: {proc.stderr.strip()}")
        for idx, line in enumerate(proc.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            append_cells(line.rstrip("\r\n").split("\t"), f"rg_match_{idx}")
        return sorted(out, key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"]))

    selected_padded = {sid.encode("ascii").ljust(30) for sid in selected}
    with path.open("rb") as fh:
        _ = fh.readline()
        for lineno, line in enumerate(fh, start=2):
            if len(line) < 31:
                if not line.strip():
                    continue
                raise ValueError(f"{path.name}: malformed tab-delimited row at line {lineno}: missing tab delimiter")
            first_tab = line.find(b"\t")
            if first_tab < 0:
                raise ValueError(f"{path.name}: malformed tab-delimited row at line {lineno}: missing tab delimiter")
            first_field = line[:first_tab]
            if first_field not in selected_padded:
                continue
            append_cells(
                [c.decode("ascii", errors="replace") for c in line.rstrip(b"\r\n").split(b"\t")],
                lineno,
            )
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


def parse_ces_api_response(
    obj: Mapping[str, object],
    selected: Mapping[str, Mapping[str, str]],
    *,
    source_file: str = "",
    source_sha256: str = "",
) -> List[Dict[str, object]]:
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
                "source_file": source_file,
                "source_sha256": source_sha256,
            })
    return rows


def select_ces_hours_series(raw_dir: Path) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, object]]]:
    selected_all, _, diag = parse_ces_metadata(raw_dir)
    selected = {sid: meta for sid, meta in selected_all.items() if str(meta.get("seasonal", "")).upper() == "U"}
    diag.append({
        "dataset": "CES total-private average weekly hours",
        "source": "sm.series",
        "selected_by": "industry_code=05000000, data_type_code=02, seasonal=U, five-digit nonzero area_code",
        "selected_series_count": len(selected),
        "excluded_source_seasonally_adjusted": len(selected_all) - len(selected),
    })
    return selected, diag


def bls_api_version(url: str) -> str:
    m = re.search(r"/publicAPI/(v\d+)/", url)
    return m.group(1) if m else ""


def batch_series_ids(series_ids: Sequence[str], batch_size: int = 50) -> List[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ids = list(series_ids)
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def build_bls_api_payload(series_ids: Sequence[str], start_year: int, end_year: int, api_key: str = "") -> Dict[str, object]:
    payload: Dict[str, object] = {
        "seriesid": list(series_ids),
        "startyear": str(int(start_year)),
        "endyear": str(int(end_year)),
    }
    if api_key:
        payload["registrationkey"] = api_key
    return payload


def classify_bls_api_response(obj: Mapping[str, object], http_status: object = "") -> str:
    status = str(obj.get("status", "")).upper()
    message = " ".join(str(m) for m in (obj.get("message", []) or [])).lower()
    if str(http_status) and str(http_status) not in {"200", "200.0"}:
        return "http_error"
    if status == "REQUEST_SUCCEEDED":
        return "success"
    if "daily threshold" in message or "query limit" in message or "quota" in message:
        return "quota"
    if "invalid series" in message or "series does not exist" in message or "no data available" in message:
        return "absent_series"
    if status:
        return status.lower()
    return "unknown"


def detect_ten_year_truncation(obj: Mapping[str, object], requested_start_year: int, requested_end_year: int) -> bool:
    requested_span = int(requested_end_year) - int(requested_start_year) + 1
    messages = " ".join(str(m) for m in (obj.get("message", []) or [])).lower()
    if "year range" in messages and ("reduced" in messages or "10" in messages or "ten" in messages):
        return True
    years: List[int] = []
    for series in (obj.get("Results", {}) or {}).get("series", []):
        for item in series.get("data", []):
            try:
                years.append(int(item.get("year")))
            except Exception:
                pass
    if requested_span > 10 and years and (max(years) - min(years) + 1) <= 10:
        return True
    return False


def _copy_without_overwrite(src: Path, dest: Path, source_url: str, dataset: str) -> Dict[str, object]:
    if not src.exists():
        return {
            "agency": "U.S. Bureau of Labor Statistics",
            "dataset": dataset,
            "url": source_url,
            "local_path": str(dest),
            "status": "missing-local-cache",
            "size": "",
            "sha256": "",
            "error": f"missing {src}",
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    if dest.exists():
        dest_hash = sha256_file(dest)
        return {
            "agency": "U.S. Bureau of Labor Statistics",
            "dataset": dataset,
            "url": source_url,
            "local_path": str(dest),
            "status": "accepted-existing" if dest_hash == src_hash else "existing-file-differs-not-overwritten",
            "size": dest.stat().st_size,
            "sha256": dest_hash,
            "error": "" if dest_hash == src_hash else f"source cache hash is {src_hash}",
        }
    shutil.copy2(src, dest)
    return {
        "agency": "U.S. Bureau of Labor Statistics",
        "dataset": dataset,
        "url": source_url,
        "local_path": str(dest),
        "status": "copied-from-repo-official-cache",
        "size": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "error": "",
    }


def freeze_ces_history_sources(root: Path, out_raw: Path, *, force_download: bool = False) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    repo_bls = root / "data" / "zillow" / "raw" / "bls_ces"
    flat_dir = out_raw / "bls_ces_flat"
    for filename in CES_HISTORY_FILES:
        url = urllib.parse.urljoin(BLS_SM_LOWER_BASE, filename)
        dest = flat_dir / filename
        rec = validated_download(url, dest, force=force_download, min_size=10)
        rec.update({
            "agency": "U.S. Bureau of Labor Statistics",
            "dataset": "CES State and Area flat file",
            "official_source_page": BLS_SM_LOWER_BASE,
        })
        if rec["status"] == "download-failed":
            local = _copy_without_overwrite(repo_bls / filename, dest, url, "CES State and Area flat file")
            if local["status"] not in {"missing-local-cache"}:
                rec = local
            else:
                rec["fallback_status"] = local["status"]
                rec["fallback_error"] = local["error"]
        records.append(rec)
    return records


def discover_ces_state_partition_files(sm_txt: Path) -> List[str]:
    if not sm_txt.exists() or looks_like_denial_or_html(sm_txt):
        return []
    text = sm_txt.read_text(encoding="utf-8", errors="replace")
    files: List[str] = []
    seen = set()
    for match in re.finditer(r"\b(sm\.data\.(\d+)[a-z]?\.[A-Za-z0-9]+)\b", text):
        filename = match.group(1)
        idx = int(match.group(2))
        if 1 <= idx <= 53 and filename not in seen:
            seen.add(filename)
            files.append(filename)
    return files


def freeze_ces_state_partitions(root: Path, out_raw: Path, partition_files: Sequence[str], *, force_download: bool = False) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    repo_bls = root / "data" / "zillow" / "raw" / "bls_ces"
    flat_dir = out_raw / "bls_ces_flat"
    for filename in partition_files:
        url = urllib.parse.urljoin(BLS_SM_LOWER_BASE, filename)
        dest = flat_dir / filename
        rec = validated_download(url, dest, force=force_download, min_size=10)
        rec.update({
            "agency": "U.S. Bureau of Labor Statistics",
            "dataset": "CES State and Area state historical partition",
            "official_source_page": BLS_SM_LOWER_BASE,
            "discovered_from_sm_txt": 1,
        })
        if rec["status"] == "download-failed":
            local = _copy_without_overwrite(repo_bls / filename, dest, url, "CES State and Area state historical partition")
            if local["status"] not in {"missing-local-cache"}:
                rec = local
                rec["discovered_from_sm_txt"] = 1
            else:
                rec["fallback_status"] = local["status"]
                rec["fallback_error"] = local["error"]
        records.append(rec)
    return records


def parse_ces_data_with_provenance(path: Path, selected: Mapping[str, Mapping[str, str]]) -> List[Dict[str, object]]:
    rows = parse_ces_data_file(path, selected)
    source_hash = sha256_file(path)
    for row in rows:
        row["source_file"] = path.name
        row["source_sha256"] = row.get("source_sha256") or source_hash
    return rows


def read_cached_ces_api_rows(api_dirs: Sequence[Path], selected: Mapping[str, Mapping[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    seen: set[Path] = set()
    for api_dir in api_dirs:
        if not api_dir.exists():
            continue
        for path in sorted(api_dir.glob("ces_hours_*.json")):
            if path in seen:
                continue
            seen.add(path)
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                manifest.append({
                    "source_file": path.name,
                    "status": "parse_failed",
                    "classification": "parse_failed",
                    "message": str(exc),
                    "local_path": str(path),
                    "sha256": sha256_file(path),
                })
                continue
            cls = classify_bls_api_response(obj)
            source_hash = sha256_file(path)
            manifest.append({
                "source_file": path.name,
                "status": obj.get("status", ""),
                "classification": cls,
                "message": "; ".join(obj.get("message", []) or []),
                "local_path": str(path),
                "sha256": source_hash,
                "api_version": bls_api_version("https://api.bls.gov/publicAPI/v2/timeseries/data/"),
                "ten_year_truncated": int(detect_ten_year_truncation(obj, 2007, 2026)),
            })
            rows.extend(parse_ces_api_response(obj, selected, source_file=path.name, source_sha256=source_hash))
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        by_key[(str(row["ces_series_id"]), str(row["date"]), str(row["cbsa_code"]))] = row
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"])), manifest


def fetch_registered_ces_api_history(
    selected: Mapping[str, Mapping[str, str]],
    api_dir: Path,
    api_key: str,
    *,
    start_year: int = 2007,
    end_year: int = 2026,
    batch_size: int = 50,
    max_retries: int = 2,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    api_dir.mkdir(parents=True, exist_ok=True)
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    rows: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    for batch_no, chunk in enumerate(batch_series_ids(sorted(selected), batch_size), start=1):
        dest = api_dir / f"ces_hours_{start_year}_{end_year}_batch{batch_no:03d}.json"
        failed = api_dir / f"ces_hours_{start_year}_{end_year}_batch{batch_no:03d}.failed.json"
        http_status = ""
        if dest.exists() or failed.exists():
            path = dest if dest.exists() else failed
            status_action = "accepted-existing-cache"
            body = path.read_bytes()
        elif not api_key:
            manifest.append({
                "batch_no": batch_no,
                "series_requested": len(chunk),
                "start_year": start_year,
                "end_year": end_year,
                "status": "skipped_no_registered_key",
                "classification": "missing_api_key",
                "message": "BLS_API_KEY/BLS_KEY not provided",
                "http_status": "",
                "api_version": bls_api_version(url),
                "local_path": "",
                "sha256": "",
                "attempts": 0,
                "ten_year_truncated": "",
            })
            continue
        else:
            payload = json.dumps(build_bls_api_payload(chunk, start_year, end_year, api_key)).encode("utf-8")
            body = b""
            status_action = "newly-downloaded"
            last_error = ""
            for attempt in range(1, max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        http_status = str(getattr(resp, "status", ""))
                        body = resp.read()
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt == max_retries:
                        body = json.dumps({"status": "HTTP_ERROR", "message": [last_error]}).encode("utf-8")
            obj_tmp = json.loads(body.decode("utf-8", errors="replace"))
            path = dest if classify_bls_api_response(obj_tmp, http_status) == "success" else failed
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(body)
            tmp.replace(path)
        obj = json.loads(body.decode("utf-8", errors="replace"))
        cls = classify_bls_api_response(obj, http_status)
        path = dest if dest.exists() else failed
        source_hash = sha256_file(path) if path.exists() else hashlib.sha256(body).hexdigest()
        manifest.append({
            "batch_no": batch_no,
            "series_requested": len(chunk),
            "start_year": start_year,
            "end_year": end_year,
            "status": obj.get("status", ""),
            "classification": cls,
            "message": "; ".join(obj.get("message", []) or []),
            "http_status": http_status,
            "api_version": bls_api_version(url),
            "local_path": str(path),
            "sha256": source_hash,
            "download_status": status_action,
            "attempts": 0 if status_action == "accepted-existing-cache" else max_retries,
            "ten_year_truncated": int(detect_ten_year_truncation(obj, start_year, end_year)),
        })
        rows.extend(parse_ces_api_response(obj, selected, source_file=path.name, source_sha256=source_hash))
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        by_key[(str(row["ces_series_id"]), str(row["date"]), str(row["cbsa_code"]))] = row
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"])), manifest


def ces_series_coverage(
    rows: Sequence[Mapping[str, object]],
    selected: Mapping[str, Mapping[str, str]],
    *,
    requested_start: str = "2007-01",
    requested_end: str = "2026-12",
) -> List[Dict[str, object]]:
    by_sid: Dict[str, List[Mapping[str, object]]] = {sid: [] for sid in selected}
    for row in rows:
        sid = str(row.get("ces_series_id", ""))
        if sid in by_sid:
            by_sid[sid].append(row)
    out: List[Dict[str, object]] = []
    for sid, meta in sorted(selected.items(), key=lambda kv: (kv[1].get("cbsa_code", ""), kv[0])):
        rr = by_sid.get(sid, [])
        months = [str(r.get("date", ""))[:7] for r in rr]
        counts: Dict[str, int] = {}
        for m in months:
            counts[m] = counts.get(m, 0) + 1
        duplicate_months = sorted(m for m, c in counts.items() if c > 1)
        first = min(months) if months else ""
        last = max(months) if months else ""
        expected = month_range(first, last) if first and last else []
        missing = [m for m in expected if m not in counts]
        vals = [parse_float(r.get("hours_raw")) for r in rr]
        source_files = sorted({str(r.get("source_file", "")) for r in rr if r.get("source_file")})
        source_hashes = sorted({str(r.get("source_sha256", "")) for r in rr if r.get("source_sha256")})
        area_name = str(meta.get("area_name", ""))
        out.append({
            "series_id": sid,
            "cbsa_code": meta.get("cbsa_code", ""),
            "area_title": area_name,
            "area_type": "metropolitan_division" if "metropolitan division" in area_name.lower() else "metropolitan_area",
            "seasonal": meta.get("seasonal", ""),
            "industry_code": meta.get("industry_code", ""),
            "data_type_code": meta.get("data_type_code", ""),
            "metadata_begin": f"{meta.get('begin_year', '')}-{str(meta.get('begin_period', ''))[-2:]}",
            "metadata_end": f"{meta.get('end_year', '')}-{str(meta.get('end_period', ''))[-2:]}",
            "requested_start": requested_start,
            "requested_end": requested_end,
            "first_month": first,
            "last_month": last,
            "observed_months": ";".join(sorted(counts)),
            "observation_count": len(rr),
            "missing_month_count": len(missing),
            "missing_months": ";".join(missing[:24]) + (";..." if len(missing) > 24 else ""),
            "duplicate_month_count": sum(counts[m] - 1 for m in duplicate_months),
            "duplicate_months": ";".join(duplicate_months),
            "preliminary_count": sum(int(r.get("preliminary_flag", 0)) for r in rr),
            "zero_count": sum(1 for v in vals if v == 0),
            "negative_count": sum(1 for v in vals if v is not None and v < 0),
            "nonfinite_count": sum(1 for v in vals if v is None or not math.isfinite(v)),
            "source_files": ";".join(source_files),
            "source_hashes": ";".join(source_hashes),
            "coverage_complete_requested_window": int(bool(rr) and first <= requested_start and last >= requested_end and not missing),
        })
    return out


def summarize_x13_coverage(
    adjusted_rows: Sequence[Mapping[str, object]],
    x13_diag: Sequence[Mapping[str, object]],
    ces_rows: Sequence[Mapping[str, object]],
    bps_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    raw_counts = {
        "ces_hours": len({str(r.get("cbsa_code", "")).zfill(5) for r in ces_rows}),
        "bps_permits": len({str(r.get("cbsa_code", "")).zfill(5) for r in bps_rows}),
    }
    out: List[Dict[str, object]] = []
    for variable in ("ces_hours", "bps_permits"):
        adj = [r for r in adjusted_rows if r.get("variable") == variable and str(r.get("x13_status", "")) == "ok"]
        diag = [d for d in x13_diag if d.get("variable") == variable]
        failed = [d for d in diag if str(d.get("x13_status", "")).startswith("failed") or str(d.get("x13_status", "")).startswith("segment_too_short")]
        statuses: Dict[str, int] = {}
        for d in diag:
            st = str(d.get("x13_status", ""))
            statuses[st] = statuses.get(st, 0) + 1
        out.append({
            "variable": variable,
            "raw_series_count": raw_counts.get(variable, 0),
            "successful_adjusted_series_count": len({str(r.get("cbsa_code", "")).zfill(5) for r in adj}),
            "failed_series_count": len({str(d.get("cbsa_code", "")).zfill(5) for d in failed}),
            "first_adjusted_month": min([str(r.get("date", ""))[:7] for r in adj] or [""]),
            "last_adjusted_month": max([str(r.get("date", ""))[:7] for r in adj] or [""]),
            "adjusted_observations": len(adj),
            "warning_or_error_count": sum(1 for d in diag if str(d.get("warning_or_error", ""))),
            "status_counts": ";".join(f"{k}:{v}" for k, v in sorted(statuses.items())),
            "exact_failure_reasons": ";".join(sorted({str(d.get("x13_status", "")) for d in failed})),
        })
    return out


def build_revised_candidate_inventory(
    inventory: Sequence[Mapping[str, object]],
    panel_rows: Sequence[Mapping[str, object]],
    *,
    ces_history_complete: bool,
    ces_loss_reason: str,
) -> List[Dict[str, object]]:
    rows_by_key: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in panel_rows:
        rows_by_key.setdefault((str(row.get("candidate_start")), str(row.get("transformation"))), []).append(row)
    out: List[Dict[str, object]] = []
    for inv in inventory:
        key = (str(inv.get("candidate_start")), str(inv.get("transformation")))
        rr = rows_by_key.get(key, [])
        dates = sorted({str(r.get("date", ""))[:7] for r in rr})
        requested = key[0]
        out.append({
            "requested_start": requested,
            "transformation": key[1],
            "effective_common_start": dates[0] if dates else "",
            "effective_common_end": dates[-1] if dates else "",
            "raw_common_months": inv.get("raw_months", ""),
            "usable_start": dates[0] if dates else "",
            "usable_end": dates[-1] if dates else "",
            "usable_months_after_transformations_and_one_month_lag": len(dates),
            "common_msa_count": inv.get("common_msas", ""),
            "missing_observations": inv.get("missing_observations", ""),
            "preliminary_observations": inv.get("preliminary_observations", ""),
            "x13_failures": inv.get("x13_failures", ""),
            "geography_mismatches": inv.get("geography_mismatches", ""),
            "dropped_msas": inv.get("dropped_msas", ""),
            "losses_from_laus": "",
            "losses_from_ces": ces_loss_reason,
            "losses_from_bps": "",
            "losses_from_x13": inv.get("x13_failures", ""),
            "losses_from_transform_and_lag_months": {"lagged_levels": 1, "one_month_growth": 2, "twelve_month_growth": 13}.get(key[1], ""),
            "freeze_allowed": int(ces_history_complete),
        })
    return out


def _read_existing_pilot_csv(root: Path, filename: str) -> List[Dict[str, str]]:
    path = root / filename
    return read_csv_rows(path) if path.exists() else []


def _final_source_file_requirement(filename: str) -> str:
    if filename in CES_FINAL_REQUIRED_METADATA_FILES:
        return "required_metadata"
    if filename in CES_FINAL_OPTIONAL_MAPPING_FILES:
        return "optional_mapping"
    if filename.startswith("sm.data") and filename != "sm.data_type":
        return "required_data_alternative"
    return "extra"


def _final_source_present_files(source_root: Path) -> List[Path]:
    if not source_root.exists():
        return []
    return sorted(p for p in source_root.rglob("*") if p.is_file())


def source_file_manifest(source_root: Path, repo_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not source_root.exists():
        return rows
    present_files = _final_source_present_files(source_root)
    present_names = {p.name for p in present_files}
    data_present = any(p.name.startswith("sm.data") and p.name != "sm.data_type" for p in present_files)
    for path in present_files:
        rel_repo = repo_relative(path, repo_root)
        rel_source = str(path.relative_to(source_root)).replace("\\", "/")
        endpoint = urllib.parse.urljoin(BLS_SM_LOWER_BASE, path.name)
        rows.append({
            "filename": path.name,
            "relative_path": rel_repo,
            "source_relative_path": rel_source,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_agency": "U.S. Bureau of Labor Statistics",
            "official_endpoint": endpoint,
            "access_date_utc": utc_now(),
            "release_or_last_modified_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
            "file_requirement": _final_source_file_requirement(path.name),
            "present": True,
            "missing": False,
            "content_looks_html": looks_like_denial_or_html(path),
        })
    for filename in (*CES_FINAL_REQUIRED_METADATA_FILES, *CES_FINAL_OPTIONAL_MAPPING_FILES):
        if filename not in present_names:
            rows.append({
                "filename": filename,
                "relative_path": "",
                "source_relative_path": "",
                "byte_size": "",
                "sha256": "",
                "source_agency": "U.S. Bureau of Labor Statistics",
                "official_endpoint": urllib.parse.urljoin(BLS_SM_LOWER_BASE, filename),
                "access_date_utc": utc_now(),
                "release_or_last_modified_date": "",
                "file_requirement": _final_source_file_requirement(filename),
                "present": False,
                "missing": True,
                "content_looks_html": False,
            })
    if not data_present:
        rows.append({
            "filename": "sm.data.*",
            "relative_path": "",
            "source_relative_path": "",
            "byte_size": "",
            "sha256": "",
            "source_agency": "U.S. Bureau of Labor Statistics",
            "official_endpoint": urllib.parse.urljoin(BLS_SM_LOWER_BASE, "sm.data.1.AllData"),
            "access_date_utc": utc_now(),
            "release_or_last_modified_date": "",
            "file_requirement": "required_data_alternative",
            "present": False,
            "missing": True,
            "content_looks_html": False,
        })
    return rows


def find_source_file(source_root: Path, filename: str) -> Optional[Path]:
    matches = sorted(p for p in source_root.rglob(filename) if p.is_file())
    return matches[0] if matches else None


def read_bls_table_from_source(source_root: Path, filename: str) -> Tuple[List[str], List[Dict[str, str]]]:
    path = find_source_file(source_root, filename)
    if path is None:
        return [], []
    if looks_like_denial_or_html(path):
        raise ValueError(f"{filename} appears to be HTML or an access-denied page, not an official BLS tab-delimited file")
    return read_bls_table(path)


def validate_final_source_file_presence(source_root: Path) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    for filename in CES_FINAL_REQUIRED_METADATA_FILES:
        path = find_source_file(source_root, filename)
        if path is None:
            reasons.append(f"missing_required_file:{filename}")
        elif looks_like_denial_or_html(path):
            reasons.append(f"html_or_denial_required_file:{filename}")
    data_files = [
        p for p in _final_source_present_files(source_root)
        if p.name.startswith("sm.data") and p.name != "sm.data_type"
    ]
    if not data_files:
        reasons.append("missing_required_data_file:sm.data.1.AllData_or_sm.data_partitions")
    for path in data_files:
        if looks_like_denial_or_html(path):
            reasons.append(f"html_or_denial_data_file:{path.name}")
    return len(reasons) == 0, reasons


def _period_name_to_month(value: str) -> Optional[int]:
    return period_to_month(value)


def _period_row_month(row: Mapping[str, str]) -> Optional[int]:
    for key in ("period_name", "period_abbr", "period"):
        month = _period_name_to_month(row.get(key, ""))
        if month is not None:
            return month
    return None


def _period_row_is_annual_average(row: Mapping[str, str]) -> bool:
    text = " ".join(str(v).lower() for v in row.values())
    return "annual" in text


def resolve_ces_period_mapping(source_root: Path) -> Dict[str, object]:
    period_path = find_source_file(source_root, "sm.period")
    sm_txt_path = find_source_file(source_root, "sm.txt")
    info: Dict[str, object] = {
        "sm_period_optional": True,
        "sm_period_supplied": period_path is not None,
        "period_mapping_source": "official_sm_txt_documentation",
        "accepted_monthly_periods": ",".join(CANONICAL_CES_MONTHLY_PERIODS),
        "excluded_annual_period": CANONICAL_CES_ANNUAL_PERIOD,
        "sm_txt_official_endpoint": urllib.parse.urljoin(BLS_SM_LOWER_BASE, "sm.txt"),
        "sm_txt_supplied": sm_txt_path is not None,
        "sm_txt_sha256": sha256_file(sm_txt_path) if sm_txt_path is not None and not looks_like_denial_or_html(sm_txt_path) else "",
    }
    if period_path is None:
        return info
    if looks_like_denial_or_html(period_path):
        raise ValueError("sm.period appears to be HTML or an access-denied page, not an official BLS tab-delimited file")
    _, rows = read_bls_table(period_path)
    by_period = {str(row.get("period", "")).strip().upper(): row for row in rows}
    for code, month in CANONICAL_CES_MONTHLY_PERIODS.items():
        row = by_period.get(code)
        if row is None:
            raise ValueError(f"sm.period is missing required monthly period code {code}")
        observed = _period_row_month(row)
        if observed != month:
            raise ValueError(f"sm.period maps {code} to {observed}, expected month {month}")
    annual = by_period.get(CANONICAL_CES_ANNUAL_PERIOD)
    if annual is None or not _period_row_is_annual_average(annual):
        raise ValueError("sm.period must identify M13 as the annual average")
    info.update({
        "period_mapping_source": "official_sm_period",
        "sm_period_sha256": sha256_file(period_path),
        "sm_period_relative_path": str(period_path.relative_to(source_root)).replace("\\", "/"),
    })
    return info


def validate_ces_period_code(period: str) -> Tuple[str, Optional[int]]:
    code = str(period).strip().upper()
    if code in CANONICAL_CES_MONTHLY_PERIODS:
        return "monthly", CANONICAL_CES_MONTHLY_PERIODS[code]
    if code == CANONICAL_CES_ANNUAL_PERIOD:
        return "annual_average", None
    raise ValueError(
        f"unexpected CES period code {period!r}; only M01-M12 monthly observations and M13 annual average are documented"
    )


def git_provenance(root: Path) -> Dict[str, object]:
    def run_git(args: Sequence[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args],
                cwd=str(root),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except Exception:
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "git_commit": commit or None,
        "git_dirty": bool(status),
        "git_status_short": status,
    }


def x13_identity() -> Dict[str, object]:
    exe = find_x13_executable(ROOT / "data" / "zillow")
    return {
        "x13_executable": str(exe) if exe else "",
        "x13_executable_sha256": sha256_file(exe) if exe else "",
        "x13_settings": {
            "ces_hours": "prespecified local X-13 policy; additive level and log-additive variants where supported",
            "bps_permits": "prespecified local X-13 additive adjustment on raw permit counts",
            "manual_series_tuning": False,
        },
    }


def generated_output_hashes(out_root: Path) -> List[Dict[str, object]]:
    expected_names = {
        "CES_FINAL_SOURCE_VALIDATION.md",
        "ces_final_series_coverage.csv",
        "x13_final_diagnostics.csv",
        "candidate_panel_comparison.csv",
        "collinearity_final.csv",
        "recommended_panel_specification.json",
        "final_dropped_msas_reasons.csv",
    }
    rows: List[Dict[str, object]] = []
    for path in sorted(p for p in out_root.iterdir() if p.is_file()):
        if path.name not in expected_names:
            continue
        rows.append({
            "filename": path.name,
            "relative_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def final_dropped_msa_reasons(coverage: Sequence[Mapping[str, object]], coverage_reasons: Sequence[str]) -> List[Dict[str, object]]:
    by_sid: Dict[str, List[str]] = {}
    for reason in coverage_reasons:
        sid, _, detail = str(reason).partition(":")
        by_sid.setdefault(sid, []).append(detail or str(reason))
    rows: List[Dict[str, object]] = []
    for row in coverage:
        sid = str(row.get("series_id", ""))
        reasons = by_sid.get(sid, [])
        if not reasons:
            continue
        rows.append({
            "series_id": sid,
            "cbsa_code": row.get("cbsa_code", ""),
            "area_title": row.get("area_title", ""),
            "area_type": row.get("area_type", ""),
            "first_month": row.get("first_month", ""),
            "last_month": row.get("last_month", ""),
            "observation_count": row.get("observation_count", ""),
            "missing_month_count": row.get("missing_month_count", ""),
            "duplicate_month_count": row.get("duplicate_month_count", ""),
            "preliminary_count": row.get("preliminary_count", ""),
            "zero_count": row.get("zero_count", ""),
            "negative_count": row.get("negative_count", ""),
            "nonfinite_count": row.get("nonfinite_count", ""),
            "drop_reason": ";".join(sorted(set(reasons))),
        })
    return rows


def parse_ces_metadata_from_source(source_root: Path) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, object]]]:
    _, area_rows = read_bls_table_from_source(source_root, "sm.area")
    _, series_rows = read_bls_table_from_source(source_root, "sm.series")
    area_names = {r.get("area_code", "").zfill(5): r.get("area_name", "") for r in area_rows}
    selected: Dict[str, Dict[str, str]] = {}
    excluded = {"not_five_digit_msa": 0, "metropolitan_division": 0, "wrong_industry": 0, "wrong_data_type": 0, "seasonally_adjusted": 0}
    for row in series_rows:
        area = str(row.get("area_code", "")).zfill(5)
        sid = row.get("series_id", "").strip()
        area_name = area_names.get(area, "")
        if not re.fullmatch(r"\d{5}", area) or area == "00000":
            excluded["not_five_digit_msa"] += 1
            continue
        if "metropolitan division" in area_name.lower():
            excluded["metropolitan_division"] += 1
            continue
        if row.get("industry_code") != "05000000":
            excluded["wrong_industry"] += 1
            continue
        if row.get("data_type_code") != "02":
            excluded["wrong_data_type"] += 1
            continue
        if row.get("seasonal", "").upper() != "U":
            excluded["seasonally_adjusted"] += 1
            continue
        selected[sid] = {
            "series_id": sid,
            "cbsa_code": area,
            "area_name": area_name,
            "seasonal": row.get("seasonal", "").upper(),
            "industry_code": row.get("industry_code", ""),
            "data_type_code": row.get("data_type_code", ""),
            "begin_year": row.get("begin_year", ""),
            "begin_period": row.get("begin_period", ""),
            "end_year": row.get("end_year", ""),
            "end_period": row.get("end_period", ""),
        }
    diagnostics = [{
        "dataset": "CES total-private average weekly hours final source",
        "source": "sm.series and sm.area",
        "selected_by": "MSA geography, industry_code=05000000, data_type_code=02, seasonal=U",
        "selected_series_count": len(selected),
        **{f"excluded_{k}": v for k, v in excluded.items()},
    }]
    return selected, diagnostics


def parse_ces_source_data_files(source_root: Path, selected: Mapping[str, Mapping[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    data_paths = [
        p for p in _final_source_present_files(source_root)
        if p.name.startswith("sm.data") and p.name != "sm.data_type"
    ]
    for path in data_paths:
        parsed = parse_ces_data_with_provenance(path, selected)
        rows.extend(parsed)
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        by_key[(str(row.get("ces_series_id")), str(row.get("date"))[:7], str(row.get("cbsa_code")))] = dict(row)
    return sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"]))


def validate_complete_ces_history(
    coverage: Sequence[Mapping[str, object]],
    *,
    required_start: str = "2007-01",
    required_end: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not coverage:
        return False, ["no_selected_ces_series"]
    last_months = [str(r.get("last_month", "")) for r in coverage if r.get("last_month")]
    if required_end is None:
        required_end = min(last_months) if last_months else ""
    if not required_end:
        return False, ["no_observed_ces_months"]
    for row in coverage:
        sid = str(row.get("series_id", ""))
        first = str(row.get("first_month", ""))
        last = str(row.get("last_month", ""))
        if not first or first > required_start:
            reasons.append(f"{sid}:starts_after_{required_start}")
        if not last or last < required_end:
            reasons.append(f"{sid}:ends_before_{required_end}")
        if int(row.get("missing_month_count") or 0) > 0:
            reasons.append(f"{sid}:missing_months")
        if int(row.get("duplicate_month_count") or 0) > 0:
            reasons.append(f"{sid}:duplicate_months")
        if int(row.get("nonfinite_count") or 0) > 0:
            reasons.append(f"{sid}:nonfinite_values")
        if int(row.get("zero_count") or 0) > 0:
            reasons.append(f"{sid}:zero_values")
        if int(row.get("negative_count") or 0) > 0:
            reasons.append(f"{sid}:negative_values")
    return len(reasons) == 0, reasons


def validate_ces_source_integrity(coverage: Sequence[Mapping[str, object]]) -> Tuple[bool, str, List[str]]:
    if not coverage:
        return False, "failed_invalid_metadata", ["no_selected_ces_series"]
    reasons: List[str] = []
    statuses = []
    for row in coverage:
        sid = str(row.get("series_id", ""))
        if not row.get("first_month") or not row.get("last_month"):
            reasons.append(f"{sid}:no_observed_months")
            statuses.append("failed_source_truncated")
        if int(row.get("missing_month_count") or 0) > 0:
            reasons.append(f"{sid}:internal_gaps")
            statuses.append("failed_internal_gaps")
        if int(row.get("duplicate_month_count") or 0) > 0:
            reasons.append(f"{sid}:duplicate_months")
            statuses.append("failed_duplicate_observations")
        if int(row.get("nonfinite_count") or 0) > 0:
            reasons.append(f"{sid}:nonfinite_values")
            statuses.append("failed_invalid_metadata")
        if int(row.get("zero_count") or 0) > 0:
            reasons.append(f"{sid}:zero_values")
            statuses.append("failed_invalid_metadata")
        if int(row.get("negative_count") or 0) > 0:
            reasons.append(f"{sid}:negative_values")
            statuses.append("failed_invalid_metadata")
    if reasons:
        priority = [
            "failed_unknown_period_codes",
            "failed_duplicate_observations",
            "failed_internal_gaps",
            "failed_source_truncated",
            "failed_invalid_metadata",
        ]
        for status in priority:
            if status in statuses:
                return False, status, reasons
        return False, "failed_invalid_metadata", reasons
    starts_after = [r for r in coverage if str(r.get("first_month", "")) > "2007-01"]
    status = "source_valid_candidate_filtering_required" if starts_after else "source_valid"
    return True, status, []


def latest_common_final_ces_month(rows: Sequence[Mapping[str, object]]) -> Tuple[str, str, List[Dict[str, object]]]:
    by_month: Dict[str, Dict[str, int]] = {}
    for row in rows:
        month = str(row.get("date", ""))[:7]
        if not month:
            continue
        d = by_month.setdefault(month, {"ces_observations": 0, "ces_preliminary_observations": 0})
        d["ces_observations"] += 1
        d["ces_preliminary_observations"] += int(row.get("preliminary_flag", 0))
    audit = [
        {
            "month": month,
            "ces_observations": vals["ces_observations"],
            "ces_preliminary_observations": vals["ces_preliminary_observations"],
            "laus_preliminary_observations": "",
            "bps_preliminary_observations": "",
        }
        for month, vals in sorted(by_month.items())
    ]
    observed = max(by_month) if by_month else ""
    final_months = [m for m, vals in by_month.items() if vals["ces_preliminary_observations"] == 0]
    final = max(final_months) if final_months else ""
    return final, observed, audit


def candidate_ces_eligibility(
    coverage: Sequence[Mapping[str, object]],
    *,
    endpoint: str,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, object]] = {}
    total = len(coverage)
    for start in STARTS:
        eligible = 0
        for row in coverage:
            reasons = []
            first = str(row.get("first_month", ""))
            last = str(row.get("last_month", ""))
            if not first or first > start:
                reasons.append("ces_starts_after_requested_start")
            if not last or (endpoint and last < endpoint):
                reasons.append("ces_ends_before_candidate_endpoint")
            observed = {m for m in str(row.get("observed_months", "")).split(";") if m}
            if endpoint and observed and not (first and first > start):
                required = month_range(start, endpoint)
                missing_required = [m for m in required if m not in observed]
                if missing_required:
                    reasons.append("ces_missing_required_candidate_months")
            if int(row.get("missing_month_count") or 0) > 0:
                reasons.append("ces_internal_missing_months")
            if int(row.get("duplicate_month_count") or 0) > 0:
                reasons.append("ces_duplicate_months")
            if int(row.get("nonfinite_count") or 0) > 0:
                reasons.append("ces_nonfinite_values")
            if int(row.get("zero_count") or 0) > 0:
                reasons.append("ces_zero_hours")
            if int(row.get("negative_count") or 0) > 0:
                reasons.append("ces_negative_hours")
            ok = not reasons
            eligible += int(ok)
            rows.append({
                "requested_start": start,
                "candidate_endpoint": endpoint,
                "eligible": int(ok),
                "series_id": row.get("series_id", ""),
                "cbsa_code": row.get("cbsa_code", ""),
                "area_title": row.get("area_title", ""),
                "official_first_month": first,
                "official_last_month": last,
                "exclusion_reason": ";".join(reasons),
            })
        summary[start] = {
            "requested_start": start,
            "candidate_endpoint": endpoint,
            "total_selected_source_msas": total,
            "eligible_ces_msas": eligible,
            "excluded_ces_msas": total - eligible,
            "candidate_available": eligible > 0 and bool(endpoint),
        }
    return rows, summary


def candidate_rows_by_key(panel_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], List[Mapping[str, object]]]:
    out: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in panel_rows:
        out.setdefault((str(row.get("candidate_start")), str(row.get("transformation"))), []).append(row)
    return out


def final_candidate_comparison(
    inventory: Sequence[Mapping[str, object]],
    panel_rows: Sequence[Mapping[str, object]],
    dropped_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    panel_by = candidate_rows_by_key(panel_rows)
    dropped_by: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in dropped_rows:
        key = (str(row.get("candidate_start")), str(row.get("transformation")))
        reason = str(row.get("dropped_reason", ""))
        d = dropped_by.setdefault(key, {"losses_from_laus": 0, "losses_from_ces": 0, "losses_from_bps": 0})
        if "missing_laus_sa" in reason:
            d["losses_from_laus"] += 1
        if "missing_transformed_hours" in reason:
            d["losses_from_ces"] += 1
        if "missing_transformed_permits" in reason:
            d["losses_from_bps"] += 1
    comparison: List[Dict[str, object]] = []
    collinearity: List[Dict[str, object]] = []
    for inv in inventory:
        key = (str(inv.get("candidate_start")), str(inv.get("transformation")))
        rr = panel_by.get(key, [])
        dates = sorted({str(r.get("date", ""))[:7] for r in rr})
        dropped = dropped_by.get(key, {"losses_from_laus": 0, "losses_from_ces": 0, "losses_from_bps": 0})
        common = int(inv.get("common_msas") or 0)
        usable = int(inv.get("usable_months_after_transformations_and_one_month_lag") or 0)
        raw = int(inv.get("raw_months") or 0)
        base = {
            "requested_start": key[0],
            "transformation": key[1],
            "effective_common_start": dates[0] if dates else "",
            "effective_common_end": dates[-1] if dates else "",
            "usable_start": dates[0] if dates else "",
            "usable_end": dates[-1] if dates else "",
            "common_msa_count": common,
            "raw_common_months": raw,
            "raw_months": raw,
            "usable_months": usable,
            "NT": common * usable,
            "NT_over_N_plus_T": (common * usable / (common + usable)) if (common + usable) else "",
            "missing_observations": inv.get("missing_observations", ""),
            "preliminary_observations": inv.get("preliminary_observations", ""),
            "x13_failures": inv.get("x13_failures", ""),
            **dropped,
        }
        for name in (
            "hours_mean", "hours_std", "hours_q01", "hours_q05", "hours_q50", "hours_q95", "hours_q99",
            "hours_within_msa_std_mean", "hours_persistence_mean", "hours_zero_change_fraction",
            "hours_extreme_change_count", "permits_mean", "permits_std", "permits_q01", "permits_q05",
            "permits_q50", "permits_q95", "permits_q99", "permits_within_msa_std_mean",
            "permits_persistence_mean", "permits_zero_change_fraction", "permits_extreme_change_count",
        ):
            base[name] = inv.get(name, "")
        comparison.append(base)
        col = {
            **base,
            "pooled_correlation": inv.get("pooled_correlation", ""),
            "two_way_demeaned_correlation": inv.get("two_way_demeaned_correlation", ""),
            "within_msa_corr_mean": inv.get("within_msa_corr_mean", ""),
            "within_msa_corr_min": inv.get("within_msa_corr_min", ""),
            "within_msa_corr_max": inv.get("within_msa_corr_max", ""),
            "within_msa_corr_q05": inv.get("within_msa_corr_q05", ""),
            "within_msa_corr_q25": inv.get("within_msa_corr_q25", ""),
            "within_msa_corr_q50": inv.get("within_msa_corr_q50", ""),
            "within_msa_corr_q75": inv.get("within_msa_corr_q75", ""),
            "within_msa_corr_q95": inv.get("within_msa_corr_q95", ""),
            "max_abs_within_msa_corr": inv.get("max_abs_within_msa_corr", ""),
            "vif": inv.get("vif", ""),
            "auxiliary_r2": inv.get("auxiliary_r2", ""),
            "gram_min_eigenvalue": inv.get("gram_min_eigenvalue", ""),
            "gram_max_eigenvalue": inv.get("gram_max_eigenvalue", ""),
            "condition_number": inv.get("condition_number", ""),
            "fold_condition_min": inv.get("fold_condition_min", ""),
            "fold_condition_median": inv.get("fold_condition_median", ""),
            "fold_condition_max": inv.get("fold_condition_max", ""),
            "fold_min_eigenvalue_min": inv.get("fold_min_eigenvalue_min", ""),
            "fold_min_eigenvalue_median": inv.get("fold_min_eigenvalue_median", ""),
            "fold_min_eigenvalue_max": inv.get("fold_min_eigenvalue_max", ""),
            "pre_covid_condition_number": inv.get("pre_covid_condition_number", ""),
            "pre_covid_gram_min_eigenvalue": inv.get("pre_covid_gram_min_eigenvalue", ""),
            "covid_post_condition_number": inv.get("covid_post_condition_number", ""),
            "covid_post_gram_min_eigenvalue": inv.get("covid_post_gram_min_eigenvalue", ""),
            "flag_abs_tw_corr_ge_0p80": inv.get("flag_abs_tw_corr_ge_0p80", ""),
            "flag_vif_ge_5": inv.get("flag_vif_ge_5", ""),
            "flag_condition_gt_30": inv.get("flag_condition_gt_30", ""),
            "flag_rank_deficient": inv.get("flag_rank_deficient", ""),
            "flag_near_zero_fold_min_eigenvalue": inv.get("flag_near_zero_fold_min_eigenvalue", ""),
        }
        collinearity.append(col)
    return comparison, collinearity


def recommend_candidate_panel(comparison: Sequence[Mapping[str, object]], collinearity: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    col_by = {(str(r.get("requested_start")), str(r.get("transformation"))): r for r in collinearity}
    feasible = []
    for row in comparison:
        key = (str(row.get("requested_start")), str(row.get("transformation")))
        col = col_by.get(key, {})
        cond = parse_float(col.get("condition_number"))
        vif = parse_float(col.get("vif"))
        tw = parse_float(col.get("two_way_demeaned_correlation"))
        common = int(row.get("common_msa_count") or 0)
        usable = int(row.get("usable_months") or 0)
        if common <= 0 or usable <= 0:
            continue
        flagged = int(
            (cond is not None and math.isfinite(cond) and cond > 30.0)
            or (vif is not None and math.isfinite(vif) and vif >= 5.0)
            or (tw is not None and math.isfinite(tw) and abs(tw) >= 0.80)
        )
        transform_priority = {"twelve_month_growth": 2, "one_month_growth": 1, "lagged_levels": 0}.get(key[1], 0)
        feasible.append((
            -flagged,
            common * usable,
            transform_priority,
            common,
            usable,
            -(cond if cond is not None and math.isfinite(cond) else 1e99),
            row,
            col,
        ))
    if not feasible:
        return {"status": "no_feasible_candidate", "freeze_panel": False}
    _, _, _, _, _, _, row, col = max(feasible, key=lambda x: x[:6])
    return {
        "status": "recommended_for_review",
        "freeze_panel": False,
        "requested_start": row.get("requested_start"),
        "transformation": row.get("transformation"),
        "effective_common_start": row.get("effective_common_start"),
        "effective_common_end": row.get("effective_common_end"),
        "common_msa_count": row.get("common_msa_count"),
        "usable_months": row.get("usable_months"),
        "NT": row.get("NT"),
        "condition_number": col.get("condition_number", ""),
        "vif": col.get("vif", ""),
        "two_way_demeaned_correlation": col.get("two_way_demeaned_correlation", ""),
        "rationale": "prefers candidate specifications without prespecified collinearity or conditioning flags, then maximizes balanced-panel NT; no immutable panel is created automatically",
    }


def annotate_candidate_comparison_with_ces_eligibility(
    comparison: Sequence[Mapping[str, object]],
    eligibility_summary: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in comparison:
        out = dict(row)
        start = str(row.get("requested_start", ""))
        elig = eligibility_summary.get(start, {})
        out["ces_eligible_msa_count"] = elig.get("eligible_ces_msas", "")
        out["ces_excluded_msa_count"] = elig.get("excluded_ces_msas", "")
        out["ces_start_date_losses"] = elig.get("excluded_ces_msas", "")
        rows.append(out)
    return rows


def run_final_source_validation(args: argparse.Namespace) -> int:
    root = find_repo_root(Path(args.repo_root).resolve())
    source_root = root / args.ces_source_root
    out_root = root / args.final_out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = source_file_manifest(source_root, root)
    validation: Dict[str, object] = {
        "output_version": CES_FINAL_SOURCE_VERSION,
        "created_utc": utc_now(),
        "source_root": repo_relative(source_root, root),
        "estimator_run": False,
        "immutable_panel_created": False,
        "manuscript_edited": False,
        "interpolation": False,
        "imputation": False,
        "backfill": False,
        "forward_fill": False,
        "winsorization": False,
        "standardization": False,
        "candidate_specifications": [
            {"requested_start": s, "transformation": t}
            for s in STARTS
            for t in TRANSFORMS
        ],
        **git_provenance(root),
        **x13_identity(),
    }
    if not source_root.exists():
        validation.update({
            "status": "failed_missing_ces_source_root",
            "continuation_allowed": False,
            "reason": f"missing {repo_relative(source_root, root)}",
        })
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text(
            "# CES Final Source Validation\n\n"
            f"Validation stopped: missing `{repo_relative(source_root, root)}`. No X-13 run, candidate panel rebuild, immutable panel creation, or estimator run occurred.\n",
            encoding="utf-8",
        )
        print(f"final CES source validation stopped: {validation['reason']}")
        return 2

    files_ok, file_reasons = validate_final_source_file_presence(source_root)
    try:
        period_mapping = resolve_ces_period_mapping(source_root)
    except ValueError as exc:
        validation.update({
            "status": "failed_period_mapping_validation",
            "continuation_allowed": False,
            "period_mapping_error": str(exc),
        })
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text(
            "# CES Final Source Validation\n\n"
            f"Validation stopped: {exc}. No X-13 run, candidate panel rebuild, immutable panel creation, or estimator run occurred.\n",
            encoding="utf-8",
        )
        print(f"final CES source validation stopped: {exc}")
        return 2
    validation.update(period_mapping)
    validation.update({
        "required_ces_files": list(CES_FINAL_REQUIRED_METADATA_FILES) + ["sm.data.1.AllData_or_official_sm.data_partitions"],
        "optional_ces_files": list(CES_FINAL_OPTIONAL_MAPPING_FILES),
        "missing_required_ces_files": file_reasons,
    })
    if not files_ok:
        validation.update({
            "status": "failed_missing_ces_source",
            "continuation_allowed": False,
            "coverage_failure_count": len(file_reasons),
            "coverage_failure_examples": file_reasons[:50],
        })
        write_csv(out_root / "ces_final_series_coverage.csv", [], [
            "series_id", "cbsa_code", "area_title", "area_type", "seasonal", "industry_code", "data_type_code",
            "metadata_begin", "metadata_end", "requested_start", "requested_end", "first_month", "last_month",
            "observation_count", "missing_month_count", "missing_months", "duplicate_month_count", "duplicate_months",
            "preliminary_count", "zero_count", "negative_count", "nonfinite_count", "source_files", "source_hashes",
            "coverage_complete_requested_window",
        ])
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text(
            "# CES Final Source Validation\n\n"
            "Validation stopped because required official CES data or metadata files are missing or are not valid BLS tab-delimited files. "
            "The optional `sm.period` mapping file is not required when the canonical BLS `M01`--`M12` monthly and `M13` annual-average rule is used. "
            "No X-13 run, candidate panel rebuild, immutable panel creation, or estimator run occurred.\n",
            encoding="utf-8",
        )
        print(f"final CES source validation stopped: {len(file_reasons)} required source issue(s)")
        return 2

    try:
        selected, metadata_diag = parse_ces_metadata_from_source(source_root)
        ces_rows = parse_ces_source_data_files(source_root, selected)
    except ValueError as exc:
        validation.update({
            "status": "failed_ces_source_parse_validation",
            "continuation_allowed": False,
            "parse_error": str(exc),
        })
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text(
            "# CES Final Source Validation\n\n"
            f"Validation stopped: {exc}. No X-13 run, candidate panel rebuild, immutable panel creation, or estimator run occurred.\n",
            encoding="utf-8",
        )
        print(f"final CES source validation stopped: {exc}")
        return 2
    final_endpoint, preliminary_endpoint, preliminary_audit = latest_common_final_ces_month(ces_rows)
    coverage = ces_series_coverage(ces_rows, selected, requested_start="2007-01", requested_end=final_endpoint or "2026-12")
    source_valid, source_status, source_reasons = validate_ces_source_integrity(coverage)
    eligibility_rows, eligibility_summary = candidate_ces_eligibility(coverage, endpoint=final_endpoint)
    current_eligibility_rows, current_eligibility_summary = candidate_ces_eligibility(coverage, endpoint=preliminary_endpoint)
    write_csv(out_root / "ces_final_series_coverage.csv", coverage, [
        "series_id", "cbsa_code", "area_title", "area_type", "seasonal", "industry_code", "data_type_code",
        "metadata_begin", "metadata_end", "requested_start", "requested_end", "first_month", "last_month",
        "observation_count", "missing_month_count", "missing_months", "duplicate_month_count", "duplicate_months",
        "preliminary_count", "zero_count", "negative_count", "nonfinite_count", "source_files", "source_hashes",
        "coverage_complete_requested_window",
    ])
    write_csv(out_root / "candidate_ces_eligibility.csv", eligibility_rows, [
        "requested_start", "candidate_endpoint", "eligible", "series_id", "cbsa_code", "area_title",
        "official_first_month", "official_last_month", "exclusion_reason",
    ])
    write_csv(out_root / "preliminary_observation_audit.csv", preliminary_audit, [
        "month", "ces_observations", "ces_preliminary_observations", "laus_preliminary_observations",
        "bps_preliminary_observations",
    ])
    dropped = [r for r in eligibility_rows if not int(r.get("eligible") or 0)]
    write_csv(out_root / "final_dropped_msas_reasons.csv", dropped, [
        "requested_start", "candidate_endpoint", "series_id", "cbsa_code", "area_title",
        "official_first_month", "official_last_month", "exclusion_reason",
    ])
    write_csv(out_root / "current_vintage_candidate_ces_eligibility.csv", current_eligibility_rows, [
        "requested_start", "candidate_endpoint", "eligible", "series_id", "cbsa_code", "area_title",
        "official_first_month", "official_last_month", "exclusion_reason",
    ])
    candidate_available = any(bool(v.get("candidate_available")) for v in eligibility_summary.values())
    validation.update({
        "status": source_status,
        "source_validation_status": source_status,
        "source_validation_passed": source_valid,
        "candidate_2007_available": bool(eligibility_summary.get("2007-01", {}).get("candidate_available")),
        "candidate_2010_available": bool(eligibility_summary.get("2010-01", {}).get("candidate_available")),
        "candidate_2011_available": bool(eligibility_summary.get("2011-01", {}).get("candidate_available")),
        "candidate_2007_N": eligibility_summary.get("2007-01", {}).get("eligible_ces_msas", 0),
        "candidate_2010_N": eligibility_summary.get("2010-01", {}).get("eligible_ces_msas", 0),
        "candidate_2011_N": eligibility_summary.get("2011-01", {}).get("eligible_ces_msas", 0),
        "final_only_endpoint": final_endpoint,
        "current_vintage_endpoint": preliminary_endpoint,
        "current_vintage_candidate_eligibility": current_eligibility_summary,
        "x13_allowed": bool(source_valid and candidate_available),
        "continuation_allowed": bool(source_valid and candidate_available),
        "selected_series_count": len(selected),
        "ces_rows": len(ces_rows),
        "observed_first_month": min([str(r.get("first_month", "")) for r in coverage if r.get("first_month")] or [""]),
        "observed_last_month": max([str(r.get("last_month", "")) for r in coverage if r.get("last_month")] or [""]),
        "coverage_failure_count": len(source_reasons),
        "coverage_failure_examples": source_reasons[:50],
        "candidate_eligibility": eligibility_summary,
        "metadata_diagnostics": metadata_diag,
    })
    if not source_valid or not candidate_available:
        (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text(
            "# CES Final Source Validation\n\n"
            "Validation stopped because the official CES source failed source-integrity checks or no candidate start has an adequate CES-eligible MSA set. "
            "No X-13 run, candidate panel rebuild, immutable panel creation, or estimator run occurred.\n",
            encoding="utf-8",
        )
        validation["generated_output_hashes"] = generated_output_hashes(out_root)
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        print(f"final CES source validation stopped: source_valid={int(source_valid)} candidate_available={int(candidate_available)}")
        return 2

    prior_root = root / args.out_root
    laus_rows = _read_existing_pilot_csv(prior_root, "laus_sa_long.csv")
    bps_rows = _read_existing_pilot_csv(prior_root, "bps_permits_raw_long.csv")
    if not laus_rows or not bps_rows:
        validation.update({"status": "failed_missing_existing_laus_or_bps_inputs", "continuation_allowed": False})
        write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
        return 2
    ces_rows_final = [
        row for row in ces_rows
        if str(row.get("date", ""))[:7] <= final_endpoint and not int(row.get("preliminary_flag", 0))
    ]
    x13_rows, x13_diag = run_x13_pilot(ces_rows_final, bps_rows, out_root, min_months=args.min_x13_months)
    inventory, panel_rows, dropped_rows = build_candidate_inventories(laus_rows, x13_rows, x13_diag)
    comparison, collinearity = final_candidate_comparison(inventory, panel_rows, dropped_rows)
    comparison = annotate_candidate_comparison_with_ces_eligibility(comparison, eligibility_summary)
    recommendation = recommend_candidate_panel(comparison, collinearity)
    write_csv(out_root / "x13_final_diagnostics.csv", x13_diag, [
        "variable", "cbsa_code", "area_title", "x13_mode", "x13_spec_id", "x13_status", "returncode",
        "reused_existing_x13_output", "n_observed", "segment_start", "segment_end", "x13_transformation",
        "x13_x11_mode", "warning_or_error", "warning_excerpt", "outlier_excerpt", "model_diagnostics_excerpt",
    ])
    write_csv(out_root / "x13_final_adjusted_long.csv", x13_rows, [
        "variable", "cbsa_code", "area_title", "date", "x13_mode", "raw_value", "adjusted_value",
        "seasonal_factor", "x13_status", "x13_spec_id", "x13_spec_path",
    ])
    write_csv(out_root / "candidate_panel_comparison.csv", comparison, [
        "requested_start", "transformation", "effective_common_start", "effective_common_end", "usable_start",
        "usable_end", "common_msa_count", "raw_common_months", "raw_months", "usable_months", "NT", "NT_over_N_plus_T",
        "ces_eligible_msa_count", "ces_excluded_msa_count", "ces_start_date_losses", "losses_from_laus",
        "losses_from_ces", "losses_from_bps", "x13_failures", "preliminary_observations", "missing_observations",
        "hours_mean", "hours_std", "hours_q01", "hours_q05", "hours_q50", "hours_q95", "hours_q99",
        "hours_within_msa_std_mean", "hours_persistence_mean", "hours_zero_change_fraction",
        "hours_extreme_change_count", "permits_mean", "permits_std", "permits_q01", "permits_q05",
        "permits_q50", "permits_q95", "permits_q99", "permits_within_msa_std_mean",
        "permits_persistence_mean", "permits_zero_change_fraction", "permits_extreme_change_count",
    ])
    write_csv(out_root / "collinearity_final.csv", collinearity, [
        "requested_start", "transformation", "common_msa_count", "usable_months", "NT", "NT_over_N_plus_T",
        "pooled_correlation", "two_way_demeaned_correlation", "within_msa_corr_mean", "within_msa_corr_min",
        "within_msa_corr_max", "within_msa_corr_q05", "within_msa_corr_q25", "within_msa_corr_q50",
        "within_msa_corr_q75", "within_msa_corr_q95", "max_abs_within_msa_corr", "vif", "auxiliary_r2",
        "gram_min_eigenvalue", "gram_max_eigenvalue", "condition_number", "fold_condition_min",
        "fold_condition_median", "fold_condition_max", "fold_min_eigenvalue_min",
        "fold_min_eigenvalue_median", "fold_min_eigenvalue_max", "pre_covid_condition_number",
        "pre_covid_gram_min_eigenvalue", "covid_post_condition_number", "covid_post_gram_min_eigenvalue",
        "flag_abs_tw_corr_ge_0p80", "flag_vif_ge_5", "flag_condition_gt_30", "flag_rank_deficient",
        "flag_near_zero_fold_min_eigenvalue",
    ])
    validation.update({
        "status": "validated_recommended_panel_specification",
        "continuation_allowed": True,
        "source_validation_passed": True,
        "x13_allowed": True,
        "x13_rows": len(x13_rows),
        "candidate_count": len(comparison),
        "recommended_panel": recommendation,
    })
    write_json(out_root / "recommended_panel_specification.json", recommendation)
    md = [
        "# CES Final Source Validation",
        "",
        f"Selected CES series: {len(selected)}.",
        f"CES observations: {len(ces_rows)}.",
        f"Candidate specifications compared: {len(comparison)}.",
        "",
        "The recommended panel is for review only. No immutable estimation panel was created and the estimator was not run.",
        "",
        f"Recommended requested start: `{recommendation.get('requested_start', '')}`.",
        f"Recommended transformation: `{recommendation.get('transformation', '')}`.",
        f"Common MSAs: `{recommendation.get('common_msa_count', '')}`.",
        f"Usable months: `{recommendation.get('usable_months', '')}`.",
    ]
    (out_root / "CES_FINAL_SOURCE_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    validation["generated_output_hashes"] = generated_output_hashes(out_root)
    write_json(out_root / "final_source_manifest.json", {"files": manifest, "validation": validation})
    print(f"final CES source validation written to {out_root}")
    print(f"recommendation: {recommendation.get('requested_start')} {recommendation.get('transformation')}")
    return 0


def run_ces_history_completion(args: argparse.Namespace) -> int:
    root = find_repo_root(Path(args.repo_root).resolve())
    prior_root = root / args.out_root
    out_root = root / args.history_out_root
    raw_root = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    source_manifest = freeze_ces_history_sources(root, raw_root, force_download=args.force_download)
    sm_txt = raw_root / "bls_ces_flat" / "sm.txt"
    state_partition_files = discover_ces_state_partition_files(sm_txt)
    all_data = raw_root / "bls_ces_flat" / "sm.data.1.AllData"
    if state_partition_files and (not all_data.exists() or looks_like_denial_or_html(all_data)):
        source_manifest.extend(
            freeze_ces_state_partitions(root, raw_root, state_partition_files, force_download=args.force_download)
        )
    selected: Dict[str, Dict[str, str]] = {}
    metadata_diag: List[Dict[str, object]] = []
    try:
        selected, metadata_diag = select_ces_hours_series(raw_root / "bls_ces_flat")
    except Exception as exc:
        metadata_diag.append({"dataset": "CES total-private average weekly hours", "status": "metadata_parse_failed", "error": str(exc)})

    ces_rows: List[Dict[str, object]] = []
    data_sources_used: List[str] = []
    for filename in ("sm.data.1.AllData", "sm.data.56.TotalPrivate.Current"):
        path = raw_root / "bls_ces_flat" / filename
        if path.exists() and not looks_like_denial_or_html(path):
            parsed = parse_ces_data_with_provenance(path, selected)
            if parsed:
                ces_rows.extend(parsed)
                data_sources_used.append(filename)
    for filename in state_partition_files:
        path = raw_root / "bls_ces_flat" / filename
        if path.exists() and not looks_like_denial_or_html(path):
            parsed = parse_ces_data_with_provenance(path, selected)
            if parsed:
                ces_rows.extend(parsed)
                data_sources_used.append(filename)

    api_key = args.bls_api_key or os.environ.get("BLS_API_KEY") or os.environ.get("BLS_KEY") or ""
    api_rows, api_manifest = fetch_registered_ces_api_history(
        selected,
        raw_root / "bls_api_ces_total_private_hours",
        api_key,
        start_year=2007,
        end_year=2026,
        batch_size=50,
        max_retries=2,
    )
    if api_rows:
        ces_rows.extend(api_rows)
        data_sources_used.append("registered_bls_api_or_cache")

    cached_rows, cached_manifest = read_cached_ces_api_rows(
        [prior_root / "raw" / "bls_api_ces_total_private_hours"],
        selected,
    )
    if cached_rows:
        ces_rows.extend(cached_rows)
        data_sources_used.append("prior_official_api_cache")
    api_manifest.extend(cached_manifest)

    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in ces_rows:
        by_key[(str(row.get("ces_series_id")), str(row.get("date"))[:7], str(row.get("cbsa_code")))] = dict(row)
    ces_rows = sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"], r["ces_series_id"]))

    coverage = ces_series_coverage(ces_rows, selected, requested_start="2007-01", requested_end="2026-12")
    complete_series = [r for r in coverage if int(r.get("coverage_complete_requested_window") or 0)]
    first_months = [str(r.get("first_month", "")) for r in coverage if r.get("first_month")]
    last_months = [str(r.get("last_month", "")) for r in coverage if r.get("last_month")]
    classifications = {str(r.get("classification", "")) for r in api_manifest if r.get("classification")}
    ten_year_detected = any(str(r.get("ten_year_truncated", "")) in {"1", "True", "true"} for r in api_manifest)
    missing_key = not bool(api_key)
    flat_failures = [r for r in source_manifest if r.get("status") == "download-failed"]
    flat_data_failures = [
        r for r in flat_failures
        if Path(str(r.get("local_path", ""))).name in {"sm.data.1.AllData", "sm.data.56.TotalPrivate.Current"}
    ]
    state_partition_successes = [
        r for r in source_manifest
        if str(r.get("dataset", "")) == "CES State and Area state historical partition"
        and r.get("status") not in {"download-failed", "missing-local-cache"}
    ]
    ces_history_complete = bool(selected) and len(complete_series) == len(selected)
    loss_reasons = []
    if flat_data_failures:
        loss_reasons.append("official_flat_data_download_unavailable")
    if missing_key:
        loss_reasons.append("registered_bls_api_key_absent")
    if "quota" in classifications:
        loss_reasons.append("api_quota_or_daily_threshold")
    if ten_year_detected:
        loss_reasons.append("api_ten_year_truncation_detected")
    if not ces_rows:
        loss_reasons.append("no_ces_history_rows_available")
    ces_loss_reason = ";".join(loss_reasons) or ("complete" if ces_history_complete else "incomplete_coverage")

    laus_rows = _read_existing_pilot_csv(prior_root, "laus_sa_long.csv")
    adjusted_rows = _read_existing_pilot_csv(prior_root, "x13_adjusted_long.csv")
    x13_diag = _read_existing_pilot_csv(prior_root, "x13_diagnostics.csv")
    bps_rows = _read_existing_pilot_csv(prior_root, "bps_permits_raw_long.csv")
    if laus_rows and adjusted_rows:
        inventory, panel_rows, _ = build_candidate_inventories(laus_rows, adjusted_rows, x13_diag)
    else:
        inventory, panel_rows = [], []
    revised_inventory = build_revised_candidate_inventory(
        inventory,
        panel_rows,
        ces_history_complete=ces_history_complete,
        ces_loss_reason=ces_loss_reason,
    )
    x13_coverage = summarize_x13_coverage(adjusted_rows, x13_diag, ces_rows, bps_rows)

    write_csv(out_root / "ces_source_manifest.csv", source_manifest, [
        "agency", "dataset", "official_source_page", "url", "local_path", "status", "size", "sha256", "error",
        "discovered_from_sm_txt", "fallback_status", "fallback_error",
    ])
    write_csv(out_root / "ces_series_coverage.csv", coverage, [
        "series_id", "cbsa_code", "area_title", "area_type", "seasonal", "industry_code", "data_type_code",
        "metadata_begin", "metadata_end", "requested_start", "requested_end", "first_month", "last_month",
        "observation_count", "missing_month_count", "missing_months", "duplicate_month_count", "duplicate_months",
        "preliminary_count", "zero_count", "negative_count", "nonfinite_count", "source_files", "source_hashes",
        "coverage_complete_requested_window",
    ])
    write_csv(out_root / "ces_api_batch_manifest.csv", api_manifest, [
        "batch_no", "source_file", "series_requested", "start_year", "end_year", "status", "classification", "message",
        "http_status", "api_version", "local_path", "sha256", "download_status", "attempts", "ten_year_truncated",
    ])
    write_csv(out_root / "candidate_inventory_revised.csv", revised_inventory, [
        "requested_start", "transformation", "effective_common_start", "effective_common_end", "raw_common_months",
        "usable_start", "usable_end", "usable_months_after_transformations_and_one_month_lag", "common_msa_count",
        "missing_observations", "preliminary_observations", "x13_failures", "geography_mismatches", "dropped_msas",
        "losses_from_laus", "losses_from_ces", "losses_from_bps", "losses_from_x13",
        "losses_from_transform_and_lag_months", "freeze_allowed",
    ])
    write_csv(out_root / "x13_coverage_by_variable.csv", x13_coverage, [
        "variable", "raw_series_count", "successful_adjusted_series_count", "failed_series_count", "first_adjusted_month",
        "last_adjusted_month", "adjusted_observations", "warning_or_error_count", "status_counts", "exact_failure_reasons",
    ])
    log_lines = [
        f"created_utc={utc_now()}",
        f"official_flat_base={BLS_SM_LOWER_BASE}",
        f"flat_download_files={','.join(CES_HISTORY_FILES)}",
        f"state_partition_files_discovered={len(state_partition_files)}",
        f"state_partition_files_available={len(state_partition_successes)}",
        f"selected_ces_series={len(selected)}",
        f"api_batch_size=50",
        f"registered_api_key_present={int(bool(api_key))}",
        f"data_sources_used={';'.join(data_sources_used)}",
        f"ces_rows={len(ces_rows)}",
        f"first_month={min(first_months) if first_months else ''}",
        f"last_month={max(last_months) if last_months else ''}",
        f"history_complete={int(ces_history_complete)}",
        f"loss_reasons={ces_loss_reason}",
    ]
    (out_root / "ces_download_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    validation = {
        "output_version": CES_HISTORY_VERSION,
        "created_utc": utc_now(),
        "requirements": {
            "official_bls_only": True,
            "preserve_response_bytes": True,
            "interpolation": False,
            "backfill": False,
            "forward_fill": False,
            "winsorization": False,
            "standardization": False,
            "manual_source_edits": False,
            "estimator_run": False,
        },
        "selected_series_count": len(selected),
        "ces_rows": len(ces_rows),
        "observed_first_month": min(first_months) if first_months else "",
        "observed_last_month": max(last_months) if last_months else "",
        "complete_series_count": len(complete_series),
        "ces_history_complete": ces_history_complete,
        "candidate_freeze_allowed": ces_history_complete,
        "candidate_freeze_refusal_reason": "" if ces_history_complete else ces_loss_reason,
        "metadata_diagnostics": metadata_diag,
        "api_classifications": sorted(classifications),
        "ten_year_truncation_detected": ten_year_detected,
        "registered_api_key_present": bool(api_key),
        "flat_required_file_download_failures": len(flat_failures),
        "flat_data_file_download_failures": len(flat_data_failures),
        "state_partition_files_discovered": len(state_partition_files),
        "state_partition_files_available": len(state_partition_successes),
    }
    write_json(out_root / "ces_history_validation.json", validation)
    md = [
        "# CES History Completion Audit",
        "",
        f"Output version: `{CES_HISTORY_VERSION}`.",
        "",
        "This audit uses official BLS State and Area CES metadata and data only. It does not interpolate, backfill, forward-fill, winsorize, standardize, manually edit source values, or run the estimator.",
        "",
        f"- Selected not-seasonally-adjusted total-private average-weekly-hours series: {len(selected)}",
        f"- Parsed CES observations: {len(ces_rows)}",
        f"- Observed CES history: {min(first_months) if first_months else 'none'} to {max(last_months) if last_months else 'none'}",
        f"- Complete requested 2007--2026 coverage: {'yes' if ces_history_complete else 'no'}",
        f"- Candidate freeze allowed: {'yes' if ces_history_complete else 'no'}",
        f"- Limitation reason: {ces_loss_reason}",
        "",
        "The previous 126-month CES cache is diagnosed as an official API/cache coverage limitation when the BLS API response indicates a ten-year reduction or when only ten calendar years are present after a 2007--2026 request.",
    ]
    (out_root / "CES_HISTORY_COMPLETION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CES history completion audit written to {out_root}")
    print(f"selected CES series={len(selected)} rows={len(ces_rows)} complete={ces_history_complete}")
    if not ces_history_complete:
        print(f"CES history is not freeze-ready: {ces_loss_reason}")
    return 0


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
        "within_msa_corr_q05": float(np.quantile(vals, 0.05)) if vals else "",
        "within_msa_corr_q25": float(np.quantile(vals, 0.25)) if vals else "",
        "within_msa_corr_q50": float(np.quantile(vals, 0.50)) if vals else "",
        "within_msa_corr_q75": float(np.quantile(vals, 0.75)) if vals else "",
        "within_msa_corr_q95": float(np.quantile(vals, 0.95)) if vals else "",
        "max_abs_within_msa_corr": float(np.max(np.abs(vals))) if vals else "",
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
    mineigs = []
    for f in folds:
        z = X[f.train]
        z = z[np.isfinite(z).all(axis=1)]
        if z.shape[0] < 3:
            continue
        d = condition_diagnostics(z[:, 0], z[:, 1])
        c = parse_float(d.get("condition_number"))
        if c is not None and math.isfinite(c):
            conds.append(c)
        lam = parse_float(d.get("gram_min_eigenvalue"))
        if lam is not None and math.isfinite(lam):
            mineigs.append(lam)
    return {
        "fold_condition_min": float(np.min(conds)) if conds else "",
        "fold_condition_median": float(np.median(conds)) if conds else "",
        "fold_condition_max": float(np.max(conds)) if conds else "",
        "fold_min_eigenvalue_min": float(np.min(mineigs)) if mineigs else "",
        "fold_min_eigenvalue_median": float(np.median(mineigs)) if mineigs else "",
        "fold_min_eigenvalue_max": float(np.max(mineigs)) if mineigs else "",
    }


def candidate_value_diagnostics(panel_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not panel_rows:
        return {}
    rows = sorted(panel_rows, key=lambda r: (str(r["cbsa_code"]), str(r["date"])))
    out: Dict[str, object] = {}
    for col, prefix in (("hours_x", "hours"), ("permits_x", "permits")):
        vals = np.array([float(r[col]) for r in rows], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.update({
            f"{prefix}_mean": float(np.mean(vals)),
            f"{prefix}_std": sd,
            f"{prefix}_q01": float(np.quantile(vals, 0.01)),
            f"{prefix}_q05": float(np.quantile(vals, 0.05)),
            f"{prefix}_q50": float(np.quantile(vals, 0.50)),
            f"{prefix}_q95": float(np.quantile(vals, 0.95)),
            f"{prefix}_q99": float(np.quantile(vals, 0.99)),
            f"{prefix}_zero_change_fraction": float(np.mean(np.isclose(vals, 0.0, atol=1e-12))),
            f"{prefix}_extreme_change_count": int(np.sum(np.abs(vals - np.mean(vals)) > 5.0 * sd)) if sd > 0 else 0,
        })
    by_code: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_code.setdefault(str(row["cbsa_code"]), []).append(row)
    for col, prefix in (("hours_x", "hours"), ("permits_x", "permits")):
        within_sds = []
        persistence = []
        for rr in by_code.values():
            vals = np.array([float(r[col]) for r in sorted(rr, key=lambda x: str(x["date"]))], float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                within_sds.append(float(np.std(vals, ddof=1)))
            if vals.size >= 3 and np.std(vals[:-1]) > 0 and np.std(vals[1:]) > 0:
                persistence.append(float(np.corrcoef(vals[:-1], vals[1:])[0, 1]))
        out[f"{prefix}_within_msa_std_mean"] = float(np.mean(within_sds)) if within_sds else ""
        out[f"{prefix}_persistence_mean"] = float(np.mean(persistence)) if persistence else ""
    return out


def panel_condition_subset(panel_rows: Sequence[Mapping[str, object]], start: str = "", end: str = "") -> Dict[str, object]:
    rr = [
        r for r in panel_rows
        if (not start or str(r.get("date", ""))[:7] >= start)
        and (not end or str(r.get("date", ""))[:7] <= end)
    ]
    if len(rr) < 3:
        return {"condition_number": "", "gram_min_eigenvalue": ""}
    x1 = np.array([float(r["hours_x"]) for r in rr], float)
    x2 = np.array([float(r["permits_x"]) for r in rr], float)
    d = condition_diagnostics(x1, x2)
    return {
        "condition_number": d.get("condition_number", ""),
        "gram_min_eigenvalue": d.get("gram_min_eigenvalue", ""),
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
                inv.update(candidate_value_diagnostics(candidate_rows))
                pre = panel_condition_subset(candidate_rows, end="2019-12")
                post = panel_condition_subset(candidate_rows, start="2020-01")
                inv["pre_covid_condition_number"] = pre.get("condition_number", "")
                inv["pre_covid_gram_min_eigenvalue"] = pre.get("gram_min_eigenvalue", "")
                inv["covid_post_condition_number"] = post.get("condition_number", "")
                inv["covid_post_gram_min_eigenvalue"] = post.get("gram_min_eigenvalue", "")
                tw = parse_float(inv.get("two_way_demeaned_correlation"))
                vif = parse_float(inv.get("vif"))
                cond = parse_float(inv.get("condition_number"))
                eig = parse_float(inv.get("gram_min_eigenvalue"))
                fold_eig = parse_float(inv.get("fold_min_eigenvalue_min"))
                inv["flag_abs_tw_corr_ge_0p80"] = int(tw is not None and math.isfinite(tw) and abs(tw) >= 0.80)
                inv["flag_vif_ge_5"] = int(vif is not None and math.isfinite(vif) and vif >= 5.0)
                inv["flag_condition_gt_30"] = int(cond is not None and math.isfinite(cond) and cond > 30.0)
                inv["flag_rank_deficient"] = int(eig is not None and eig <= 0.0)
                inv["flag_near_zero_fold_min_eigenvalue"] = int(fold_eig is not None and fold_eig < 1e-10)
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
    ap.add_argument("--history-out-root", default="outputs/empirical/unemployment/raw_sa_pilot/ces_history_completion")
    ap.add_argument("--final-out-root", default="outputs/empirical/unemployment/raw_sa_pilot/final_source_validation")
    ap.add_argument("--ces-source-root", default="data/unemployment/raw/ces_metro")
    ap.add_argument("--complete-ces-history", action="store_true",
                    help="run isolated CES history completion audit without regenerating the main pilot")
    ap.add_argument("--validate-final-ces-source", action="store_true",
                    help="validate supplied official CES metro source files and rebuild final candidate comparisons")
    ap.add_argument("--bls-api-key", default="",
                    help="registered BLS API key for CES history completion; defaults to BLS_API_KEY/BLS_KEY")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--min-x13-months", type=int, default=84)
    args = ap.parse_args(argv)

    if args.complete_ces_history:
        return run_ces_history_completion(args)
    if args.validate_final_ces_source:
        return run_final_source_validation(args)

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
