"""Audit and acquisition helpers for the monthly MSA housing panel.

This module is intentionally standard-library only.  The empirical estimator
still uses the legacy top/bottom Zillow loaders; these helpers create a separate
audit-ready all-homes data pipeline under ``data/zillow`` without interpolation,
imputation, winsorization, or standardization.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


PARSER_VERSION = "housing_audit_v1"
USER_AGENT = "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"

ZILLOW_DATA_PAGE = "https://www.zillow.com/research/data/"
ZILLOW_ALL_HOMES_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
CENSUS_BPS_PAGE = "https://www.census.gov/construction/bps/index.html"
BPS_FOLDERS = (
    "https://www2.census.gov/econ/bps/Metro%20%28ending%202023%29/",
    "https://www2.census.gov/econ/bps/CBSA%20%28beginning%20Jan%202024%29/",
)
CENSUS_DELINEATION_PAGE = (
    "https://www.census.gov/geographies/reference-files/time-series/demo/"
    "metro-micro/delineation-files.html"
)
CBSA_DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2023/delineation-files/list1_2023.xlsx"
)
BLS_SAE_PAGE = "https://www.bls.gov/sae/"
BLS_SM_BASE = "https://download.bls.gov/pub/time.series/SM/"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_SERIES_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.series")
BLS_AREA_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.area")
BLS_SEASONAL_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.seasonal")
BLS_INDUSTRY_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.industry")
BLS_FOOTNOTE_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.footnote")
BLS_TNF_DATA_URL = urllib.parse.urljoin(BLS_SM_BASE, "sm.data.54.TotalNonFarm.All")
X13_PAGE = "https://www.census.gov/data/software/x13as.X-13ARIMA-SEATS.html"
BLS_BULK_FILES = (
    ("sm.area", BLS_AREA_URL, "CES/SAE area metadata", 1000),
    ("sm.seasonal", BLS_SEASONAL_URL, "CES/SAE seasonal metadata", 10),
    ("sm.industry", BLS_INDUSTRY_URL, "CES/SAE industry metadata", 100),
    ("sm.footnote", BLS_FOOTNOTE_URL, "CES/SAE footnote metadata", 10),
    ("sm.series", BLS_SERIES_URL, "CES/SAE series metadata", 1000),
    ("sm.data.54.TotalNonFarm.All", BLS_TNF_DATA_URL, "CES/SAE total nonfarm data", 1000),
)

DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?$")
BPS_COLS = [
    "period", "csa_code", "cbsa_code", "header_code", "cbsa_title",
    "imp_101_bldgs", "imp_101_units", "imp_101_value",
    "imp_103_bldgs", "imp_103_units", "imp_103_value",
    "imp_104_bldgs", "imp_104_units", "imp_104_value",
    "imp_105_bldgs", "imp_105_units", "imp_105_value",
    "rep_101_bldgs", "rep_101_units", "rep_101_value",
    "rep_103_bldgs", "rep_103_units", "rep_103_value",
    "rep_104_bldgs", "rep_104_units", "rep_104_value",
    "rep_105_bldgs", "rep_105_units", "rep_105_value",
]


@dataclass
class DownloadRecord:
    agency: str
    dataset: str
    official_source_page: str
    exact_download_url: str
    download_timestamp_utc: str
    http_status: Optional[int]
    local_path: str
    status: str
    file_size: Optional[int]
    sha256: Optional[str]
    release_vintage: str
    parser_version: str
    seasonal_adjustment_status: str
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs(data_root: Path) -> Dict[str, Path]:
    dirs = {
        "root": data_root,
        "raw_zillow": data_root / "raw" / "zillow",
        "raw_bps": data_root / "raw" / "census_bps",
        "raw_bls": data_root / "raw" / "bls_ces",
        "raw_geo": data_root / "raw" / "geography",
        "processed": data_root / "processed",
        "audit": data_root / "audit",
        "x13": data_root / "tools" / "x13",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False,
                                    dir=str(path.parent)) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    tmp.replace(path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False,
                                    dir=str(path.parent)) as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
        tmp = Path(fh.name)
    tmp.replace(path)


def read_simple_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    _, rows = read_csv_rows(path)
    return rows


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x) for x in value)
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path, max_rows: Optional[int] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        if path.suffix.lower() == ".csv":
            dialect = csv.excel
        elif path.suffix.lower() == ".tsv":
            dialect = csv.excel_tab
        else:
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        header = list(reader.fieldnames or [])
        rows = []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
    return header, rows


def looks_like_date_column(name: str) -> bool:
    return bool(DATE_RE.match(str(name).strip()))


def normalize_month(value: str) -> Optional[str]:
    value = str(value).strip()
    if not value:
        return None
    value = value.replace("/", "-")
    parts = value.split("-")
    if len(parts) < 2:
        return None
    try:
        y, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if y < 1800 or not (1 <= m <= 12):
        return None
    return f"{y:04d}-{m:02d}"


def month_to_int(ym: str) -> int:
    y, m = ym[:7].split("-")
    return int(y) * 12 + int(m) - 1


def int_to_month(n: int) -> str:
    y, m0 = divmod(int(n), 12)
    return f"{y:04d}-{m0 + 1:02d}"


def month_range(start: str, end: str) -> List[str]:
    a, b = month_to_int(start), month_to_int(end)
    return [int_to_month(x) for x in range(a, b + 1)]


def parse_float(value) -> Optional[float]:
    try:
        s = str(value).strip().replace(",", "")
        if s in {"", ".", "-", "(NA)", "NA", "nan"}:
            return None
        return float(s)
    except Exception:
        return None


def detect_missing_months(months: Iterable[str]) -> List[str]:
    vals = sorted({m[:7] for m in months if normalize_month(m)})
    if not vals:
        return []
    full = set(month_range(vals[0], vals[-1]))
    return sorted(full.difference(vals))


def duplicate_key_count(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> int:
    seen = set()
    dup = 0
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            dup += 1
        seen.add(key)
    return dup


def contiguous_segments(months: Sequence[str]) -> List[Tuple[str, str, int]]:
    vals = sorted({m[:7] for m in months if normalize_month(m)})
    if not vals:
        return []
    nums = [month_to_int(m) for m in vals]
    out = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n != prev + 1:
            out.append((int_to_month(start), int_to_month(prev), prev - start + 1))
            start = n
        prev = n
    out.append((int_to_month(start), int_to_month(prev), prev - start + 1))
    return out


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\bmetropolitan statistical area\b|\bmicropolitan statistical area\b", "", text)
    text = re.sub(r"\bmsa\b|\bcbsa\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def http_download(url: str, dest: Path, *, force: bool = False, tries: int = 3,
                  timeout: int = 90) -> DownloadRecord:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return DownloadRecord("", "", "", url, utc_now(), None, str(dest), "accepted-existing",
                              dest.stat().st_size, sha256_file(dest), "existing", PARSER_VERSION,
                              "unknown")
    last_error = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))
                with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as fh:
                    shutil.copyfileobj(resp, fh)
                    tmp = Path(fh.name)
            tmp.replace(dest)
            return DownloadRecord("", "", "", url, utc_now(), status, str(dest), "newly-downloaded",
                                  dest.stat().st_size, sha256_file(dest), "downloaded", PARSER_VERSION,
                                  "unknown")
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 10))
    return DownloadRecord("", "", "", url, utc_now(), None, str(dest), "download-failed",
                          None, None, "unknown", PARSER_VERSION, "unknown", str(last_error))


def _response_body_prefix(exc) -> str:
    try:
        data = exc.read(300)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _looks_like_html(path: Path) -> bool:
    try:
        prefix = path.open("rb").read(512).lstrip().lower()
    except Exception:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html") or b"access denied" in prefix[:300]


def _tab_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline().rstrip("\n")
    return [x.strip() for x in first.split("\t")]


def validate_bls_bulk_file(path: Path, expected_header: Sequence[str], min_size: int) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    size = path.stat().st_size
    if size < int(min_size):
        return False, f"implausibly small file: {size} bytes"
    if _looks_like_html(path):
        return False, "response looks like HTML/access-denied page"
    header = _tab_header(path)
    missing = [h for h in expected_header if h not in header]
    if missing:
        return False, f"missing expected tab-delimited headers: {missing}; observed={header[:12]}"
    return True, "ok"


def _bls_expected_header(filename: str) -> List[str]:
    if filename == "sm.area":
        return ["area_code", "area_name"]
    if filename == "sm.seasonal":
        return ["seasonal_code", "seasonal_text"]
    if filename == "sm.industry":
        return ["industry_code", "industry_name"]
    if filename == "sm.footnote":
        return ["footnote_code", "footnote_text"]
    if filename == "sm.series":
        return ["series_id", "area_code", "industry_code", "data_type_code", "seasonal"]
    return ["series_id", "year", "period", "value"]


def bls_python_get(url: str, dest: Path, min_size: int, expected_header: Sequence[str],
                   *, tries: int = 4, timeout: int = 120) -> Tuple[Optional[DownloadRecord], List[Dict[str, object]]]:
    diagnostics = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    for attempt in range(1, tries + 1):
        info = {
            "url": url,
            "method": "GET",
            "headers": {
                "User-Agent": USER_AGENT,
                "Accept": "text/plain, application/octet-stream, */*",
                "Connection": "close",
            },
            "attempt": attempt,
            "used_head": False,
            "transport": "python_http",
            "url_case": "uppercase /SM/" if "/SM/" in url else "not uppercase /SM/",
        }
        try:
            req = urllib.request.Request(url, headers=info["headers"], method="GET")
            h = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                info["response_status"] = int(getattr(resp, "status", 200))
                info["response_headers"] = dict(resp.headers.items())
                info["redirect_history"] = [resp.geturl()] if resp.geturl() != url else []
                with part.open("wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        h.update(chunk)
                        total += len(chunk)
                        fh.write(chunk)
            info["bytes_streamed"] = total
            info["sha256_streamed"] = h.hexdigest()
            part.replace(dest)
            ok, reason = validate_bls_bulk_file(dest, expected_header, min_size)
            info["validation"] = reason
            info["content_looks_html"] = _looks_like_html(dest)
            diagnostics.append(info)
            if ok and sha256_file(dest) == h.hexdigest() and dest.stat().st_size == total:
                rec = DownloadRecord("U.S. Bureau of Labor Statistics", "", BLS_SAE_PAGE, url,
                                     utc_now(), int(info["response_status"]), str(dest),
                                     "newly-downloaded", dest.stat().st_size, sha256_file(dest),
                                     "current BLS SM bulk download", PARSER_VERSION,
                                     "metadata-defined")
                rec.__dict__["transport"] = "python_http"
                return rec, diagnostics
            if dest.exists():
                dest.unlink()
        except urllib.error.HTTPError as exc:
            info["response_status"] = exc.code
            info["response_headers"] = dict(exc.headers.items()) if exc.headers else {}
            info["response_body_prefix"] = _response_body_prefix(exc)
            info["validation"] = str(exc)
            diagnostics.append(info)
        except Exception as exc:
            info["validation"] = str(exc)
            diagnostics.append(info)
        if part.exists():
            part.unlink()
        time.sleep(min(2 ** attempt, 10))
    return None, diagnostics


def bls_curl_get(url: str, dest: Path, min_size: int, expected_header: Sequence[str]) -> Tuple[Optional[DownloadRecord], Dict[str, object]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    cmd = [
        "curl.exe", "-L", "--fail", "--retry", "4", "--retry-delay", "3",
        "-A", USER_AGENT,
        "-H", "Accept: text/plain, application/octet-stream, */*",
        "-H", "Connection: close",
        "-o", str(part),
        url,
    ]
    diag = {
        "url": url,
        "method": "GET",
        "headers": {"User-Agent": USER_AGENT, "Accept": "text/plain, application/octet-stream, */*", "Connection": "close"},
        "used_head": False,
        "transport": "curl_fallback",
        "shell": False,
        "command": cmd,
        "url_case": "uppercase /SM/" if "/SM/" in url else "not uppercase /SM/",
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        diag["returncode"] = proc.returncode
        diag["stdout"] = proc.stdout[-1000:]
        diag["stderr"] = proc.stderr[-1000:]
        if proc.returncode != 0:
            if part.exists():
                part.unlink()
            diag["validation"] = "curl failed"
            return None, diag
        part.replace(dest)
        ok, reason = validate_bls_bulk_file(dest, expected_header, min_size)
        diag["validation"] = reason
        diag["content_looks_html"] = _looks_like_html(dest)
        if ok:
            rec = DownloadRecord("U.S. Bureau of Labor Statistics", "", BLS_SAE_PAGE, url,
                                 utc_now(), None, str(dest), "newly-downloaded",
                                 dest.stat().st_size, sha256_file(dest),
                                 "current BLS SM bulk download", PARSER_VERSION,
                                 "metadata-defined")
            rec.__dict__["transport"] = "curl_fallback"
            return rec, diag
        dest.unlink(missing_ok=True)
        return None, diag
    except Exception as exc:
        if part.exists():
            part.unlink()
        diag["validation"] = str(exc)
        return None, diag


def bls_api_fallback_get(series_ids: Sequence[str], dest: Path, start_year: int, end_year: int) -> Tuple[Optional[DownloadRecord], Dict[str, object]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    diag = {
        "url": BLS_API_URL,
        "method": "POST",
        "headers": {"User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "application/json"},
        "used_head": False,
        "transport": "bls_api_fallback",
        "series_count": len(series_ids),
        "start_year": start_year,
        "end_year": end_year,
        "validation": "",
    }
    rows: List[Dict[str, str]] = []
    try:
        # The public BLS API has request-size/year-span limits. Chunking keeps
        # this fallback deterministic and avoids changing the primary bulk path.
        year_chunks = []
        y = int(start_year)
        while y <= int(end_year):
            yy = min(y + 19, int(end_year))
            year_chunks.append((y, yy))
            y = yy + 1
        ids = list(dict.fromkeys(series_ids))
        for start, end in year_chunks:
            for i in range(0, len(ids), 50):
                payload = json.dumps({
                    "seriesid": ids[i:i + 50],
                    "startyear": str(start),
                    "endyear": str(end),
                }).encode("utf-8")
                req = urllib.request.Request(
                    BLS_API_URL,
                    data=payload,
                    headers=diag["headers"],
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    diag["response_status"] = int(getattr(resp, "status", 200))
                    body = resp.read()
                obj = json.loads(body.decode("utf-8"))
                if obj.get("status") != "REQUEST_SUCCEEDED":
                    diag["validation"] = f"API status {obj.get('status')}: {obj.get('message')}"
                    raise RuntimeError(diag["validation"])
                for series in obj.get("Results", {}).get("series", []):
                    sid = series.get("seriesID", "")
                    for item in series.get("data", []):
                        period = item.get("period", "")
                        if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                            continue
                        footnotes = item.get("footnotes", []) or []
                        codes = "".join(str(f.get("code", "")) for f in footnotes if f.get("code"))
                        rows.append({
                            "series_id": sid,
                            "year": str(item.get("year", "")),
                            "period": period,
                            "value": str(item.get("value", "")),
                            "footnote_codes": codes,
                        })
        if not rows:
            diag["validation"] = "API returned no monthly observations"
            return None, diag
        with part.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["series_id", "year", "period", "value", "footnote_codes"],
                               delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in sorted(rows, key=lambda r: (r["series_id"], r["year"], r["period"])):
                w.writerow(row)
        part.replace(dest)
        ok, reason = validate_bls_bulk_file(dest, _bls_expected_header("sm.data.54.TotalNonFarm.All"), 1)
        diag["validation"] = reason
        if not ok:
            dest.unlink(missing_ok=True)
            return None, diag
        rec = DownloadRecord("U.S. Bureau of Labor Statistics", "CES/SAE total nonfarm data",
                             BLS_SAE_PAGE, BLS_API_URL, utc_now(), None, str(dest),
                             "newly-downloaded", dest.stat().st_size, sha256_file(dest),
                             "current BLS API fallback", PARSER_VERSION, "metadata-defined")
        rec.__dict__["transport"] = "bls_api_fallback"
        return rec, diag
    except Exception as exc:
        if part.exists():
            part.unlink()
        diag["validation"] = str(exc)
        return None, diag


def download_bls_bulk_file(filename: str, url: str, dataset: str, min_size: int,
                           dest_dir: Path, force: bool,
                           diagnostics: List[Dict[str, object]]) -> DownloadRecord:
    dest = dest_dir / filename
    expected = _bls_expected_header(filename)
    if dest.exists() and not force:
        ok, reason = validate_bls_bulk_file(dest, expected, min_size)
        diagnostics.append({
            "url": url,
            "method": "GET",
            "transport": "cache_reuse",
            "used_head": False,
            "response_status": "",
            "response_headers": {},
            "redirect_history": [],
            "validation": reason,
            "url_case": "uppercase /SM/" if "/SM/" in url else "not uppercase /SM/",
            "response_body_prefix": "",
        })
        if ok:
            rec = DownloadRecord("U.S. Bureau of Labor Statistics", dataset, BLS_SAE_PAGE,
                                 url, utc_now(), None, str(dest), "accepted-existing",
                                 dest.stat().st_size, sha256_file(dest),
                                 "current BLS SM bulk download", PARSER_VERSION,
                                 "metadata-defined")
            rec.__dict__["transport"] = "cache_reuse"
            return rec
        dest.unlink()
    rec, py_diag = bls_python_get(url, dest, min_size, expected)
    diagnostics.extend(py_diag)
    if rec is not None:
        rec.dataset = dataset
        return rec
    rec, curl_diag = bls_curl_get(url, dest, min_size, expected)
    diagnostics.append(curl_diag)
    if rec is not None:
        rec.dataset = dataset
        return rec
    return DownloadRecord("U.S. Bureau of Labor Statistics", dataset, BLS_SAE_PAGE,
                          url, utc_now(), None, str(dest), "download-failed",
                          None, None, "current BLS SM bulk download", PARSER_VERSION,
                          "metadata-defined", str(diagnostics[-1].get("validation", "download failed")))


def write_bls_download_diagnosis(data_root: Path, diagnostics: Sequence[Dict[str, object]],
                                 old_log: str = "") -> None:
    lines = [
        "# BLS CES Download Diagnosis",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Previous failure",
        "- Previous code used `download.bls.gov` with lowercase `/sm/` URLs.",
        "- The failed URL recorded in `download_log.txt` was `https://download.bls.gov/pub/time.series/sm/sm.series`.",
        "- The code used HTTPS GET through Python `urllib.request` with a descriptive User-Agent.",
        "- The pipeline code did not use FTP, the BLS API, browser automation, or `www.bls.gov` for the bulk file.",
        "- The previous response was HTTP 403 Forbidden and no valid text body was saved.",
        "",
        "## Repaired source",
        "- Repaired code uses the official case-sensitive bulk directory `https://download.bls.gov/pub/time.series/SM/`.",
        "- The downloader uses GET only; no HEAD request is issued by the pipeline.",
        "- Python streamed GET is attempted first, followed by `curl.exe` fallback with `shell=False`.",
        "- If metadata files succeed but the large TotalNonFarm bulk file fails, the code can fall back to the BLS public API and write an equivalent tab-delimited cache.",
        "",
        "## Attempts",
    ]
    for i, d in enumerate(diagnostics, start=1):
        lines.append(f"### Attempt {i}: {d.get('transport', '')} {d.get('url', '')}")
        lines.append(f"- method: {d.get('method', '')}")
        lines.append(f"- used HEAD: {d.get('used_head', False)}")
        lines.append(f"- URL case: {d.get('url_case', '')}")
        lines.append(f"- request headers: `{json.dumps(d.get('headers', {}), sort_keys=True)}`")
        lines.append(f"- redirect history: `{json.dumps(d.get('redirect_history', []), sort_keys=True)}`")
        lines.append(f"- response status: {d.get('response_status', d.get('returncode', ''))}")
        lines.append(f"- response headers: `{json.dumps(d.get('response_headers', {}), sort_keys=True)[:2000]}`")
        prefix = str(d.get("response_body_prefix", "") or "")
        if prefix:
            lines.append(f"- response body prefix: `{prefix[:300].replace(chr(10), ' ')}`")
        lines.append(f"- validation: {d.get('validation', '')}")
        lines.append(f"- content looked like HTML/access denied: {d.get('content_looks_html', '')}")
        lines.append("")
    atomic_write_text(data_root / "audit" / "bls_download_diagnosis.md", "\n".join(lines) + "\n")


def classify_zillow_file(path: Path, header: Sequence[str]) -> Dict[str, object]:
    name = path.name.lower()
    has_dates = any(looks_like_date_column(h) for h in header)
    lower_header = {str(h).lower() for h in header}
    is_metro = "regiontype" in lower_header or "region type" in lower_header
    top_bottom = any(tok in name for tok in ("top", "bottom", "tier_0.67", "tier_0.0"))
    all_homes = ("sfrcondo" in name and "tier_0.33_0.67" in name) or "all_home" in name
    sm_sa = ("_sm_sa_" in name) or ("smoothed" in name and "season" in name)
    return {
        "is_zillow_like": "zhvi" in name or {"regionid", "regionname"}.issubset(lower_header),
        "is_all_homes_metro_sa": bool(has_dates and is_metro and all_homes and sm_sa and not top_bottom),
        "is_top_or_bottom_tier": bool(top_bottom),
        "seasonal_adjustment_status": "smoothed seasonally adjusted" if sm_sa else "not verifiable",
        "housing_category": "all homes SFR condo/co-op" if all_homes else ("top/bottom tier" if top_bottom else "unknown"),
    }


def parse_zillow_all_homes(path: Path, source_vintage: str = "downloaded") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    header, rows = read_csv_rows(path)
    cls = classify_zillow_file(path, header)
    if not cls["is_all_homes_metro_sa"]:
        raise ValueError(f"not a Zillow all-homes metro smoothed SA file: {path}")
    date_cols = [h for h in header if looks_like_date_column(h)]
    out = []
    for row in rows:
        region_type = row.get("RegionType") or row.get("Region Type") or ""
        if region_type.strip().lower() != "msa":
            continue
        for col in date_cols:
            value = parse_float(row.get(col))
            out.append({
                "zillow_region_id": row.get("RegionID", ""),
                "zillow_region_name": row.get("RegionName", ""),
                "region_type": region_type,
                "state_name": row.get("StateName", ""),
                "date": (normalize_month(col) or col[:7]) + "-01",
                "zhvi_all_homes_sa": value,
                "source_file": str(path),
                "source_vintage": source_vintage,
            })
    summary = {
        "n_metros": len({r["zillow_region_id"] for r in out}),
        "earliest_date": min((r["date"] for r in out), default=""),
        "latest_date": max((r["date"] for r in out), default=""),
        "missing_months": detect_missing_months([r["date"][:7] for r in out]),
        "duplicate_keys": duplicate_key_count(out, ["zillow_region_id", "date"]),
    }
    return out, summary


def inventory_existing_data(data_root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(data_root).as_posix()
        stat = path.stat()
        rec = {
            "relative_path": rel,
            "file_type": path.suffix.lower().lstrip(".") or "unknown",
            "file_size": stat.st_size,
            "sha256": sha256_file(path),
            "modification_time": _dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "row_count": "",
            "column_count": "",
            "column_names": "",
            "geographic_level": "unknown",
            "frequency": "unknown",
            "earliest_date": "",
            "latest_date": "",
            "duplicate_keys": "",
            "missing_months": "",
            "apparent_source": "unknown",
            "apparent_vintage": "unknown",
            "seasonal_adjustment_status": "unknown",
            "transformations_applied": "unknown",
            "interpolation_may_have_occurred": "unknown",
        }
        if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            try:
                header, sample = read_csv_rows(path, max_rows=100000)
                rec["row_count"] = len(sample)
                rec["column_count"] = len(header)
                rec["column_names"] = ";".join(header[:80])
                date_cols = [h for h in header if looks_like_date_column(h)]
                date_values = []
                if date_cols:
                    date_values = [normalize_month(h) for h in date_cols if normalize_month(h)]
                elif "date" in {h.lower() for h in header}:
                    dcol = next(h for h in header if h.lower() == "date")
                    date_values = [normalize_month(r.get(dcol, "")) for r in sample]
                    date_values = [d for d in date_values if d]
                if date_values:
                    rec["frequency"] = "monthly" if len(set(d[-2:] for d in date_values)) > 1 else "unknown"
                    rec["earliest_date"] = min(date_values)
                    rec["latest_date"] = max(date_values)
                    rec["missing_months"] = len(detect_missing_months(date_values))
                lower_cols = {h.lower() for h in header}
                if {"regionid", "regionname"}.issubset(lower_cols):
                    cls = classify_zillow_file(path, header)
                    rec["apparent_source"] = "Zillow Research"
                    rec["geographic_level"] = "Metro/MSA" if any(str(r.get("RegionType", "")).lower() == "msa" for r in sample) else "unknown"
                    rec["seasonal_adjustment_status"] = cls["seasonal_adjustment_status"]
                    rec["transformations_applied"] = cls["housing_category"]
                if "permits_units_growth_12m" in lower_cols or "population_growth_12m" in lower_cols:
                    rec["apparent_source"] = "project-built covariates"
                    rec["transformations_applied"] = "12-month growth; population/GDP monthly construction"
                    rec["interpolation_may_have_occurred"] = "yes"
                if "cbsa_code" in lower_cols:
                    rec["geographic_level"] = "CBSA/MSA"
                keys = []
                if "cbsa_code" in lower_cols and any(h.lower() == "date" for h in header):
                    keys = [next(h for h in header if h.lower() == "cbsa_code"), next(h for h in header if h.lower() == "date")]
                elif "RegionID" in header and date_cols:
                    keys = ["RegionID"]
                if keys and not date_cols:
                    rec["duplicate_keys"] = duplicate_key_count(sample, keys)
            except Exception as exc:
                rec["parser_error"] = str(exc)
        rows.append(rec)
    return rows


def audit_existing_code(root: Path, data_root: Path) -> str:
    entries = [
        {
            "path": "scripts/zillow_abc.py",
            "current purpose": "Runs legacy housing A/B/C/D empirical specifications.",
            "data inputs": "data/zillow/zillow_metro_top.csv; data/zillow/zillow_metro_bottom.csv; metro_monthly_covariates_2000_present.csv; cbsa_county_crosswalk_2023.csv",
            "data outputs": "outputs/empirical/zillow_*.json",
            "Zillow series used": "top-tier and bottom-tier ZHVI stacked as separate units",
            "covariates used": "permits, population, GDP in spec C",
            "whether interpolation occurs": "not in this script; upstream covariate file documents population interpolation and GDP monthly carry/lagging",
            "whether seasonal adjustment occurs": "no local adjustment here; relies on Zillow tier files",
            "whether winsorization occurs": "not in this script; covariate loader winsorizes",
            "whether standardization occurs": "yes via load_zillow and covariate loader",
            "whether the code can be retained": "yes, preserve for old top/bottom reproducibility",
            "required revision": "do not change for this audit; future all-homes spec should call a separate loader",
        },
        {
            "path": "dlrhcs/empirical.py",
            "current purpose": "Legacy empirical loaders, AR(2) construction, targets, and run_ar2.",
            "data inputs": "Zillow tier CSVs; metro unemployment CSVs",
            "data outputs": "in-memory model panels and result dictionaries",
            "Zillow series used": "top/bottom tiers in load_zillow",
            "covariates used": "optional predetermined covariates from caller",
            "whether interpolation occurs": "no direct interpolation",
            "whether seasonal adjustment occurs": "no",
            "whether winsorization occurs": "no direct winsorization",
            "whether standardization occurs": "yes, load_zillow standardizes transformed series",
            "whether the code can be retained": "yes",
            "required revision": "add future all-homes model loader only after data audit approval",
        },
        {
            "path": "dlrhcs/covariates.py",
            "current purpose": "Legacy covariate matching and transformation for Zillow/unemployment specs.",
            "data inputs": "metro_monthly_covariates_2000_present.csv; cbsa_county_crosswalk_2023.csv",
            "data outputs": "covariate matrices aligned to panel units",
            "Zillow series used": "region names from legacy Zillow tier loader",
            "covariates used": "permits_units_growth_12m, population_growth_12m, real_gdp_growth_1y",
            "whether interpolation occurs": "not directly; docstring states upstream annual-to-monthly interpolation for GDP/population",
            "whether seasonal adjustment occurs": "no",
            "whether winsorization occurs": "yes, 1st/99th percentile clipping",
            "whether standardization occurs": "yes, z-scoring",
            "whether the code can be retained": "yes for legacy outputs only",
            "required revision": "future headline all-homes specification should not use this three-covariate loader",
        },
        {
            "path": "data/zillow/zillow-covariate.py",
            "current purpose": "Legacy upstream covariate builder.",
            "data inputs": "Census BPS, BEA county GDP/population, Census CBSA delineation",
            "data outputs": "metro_monthly_covariates_2000_present.csv and intermediates",
            "Zillow series used": "none directly",
            "covariates used": "permits, population, GDP",
            "whether interpolation occurs": "yes for annual population; GDP is repeated/lagged into months",
            "whether seasonal adjustment occurs": "no local X-13 adjustment",
            "whether winsorization occurs": "no",
            "whether standardization occurs": "no",
            "whether the code can be retained": "yes as legacy provenance, but not for new headline data",
            "required revision": "replace with no-interpolation monthly-only pipeline",
        },
        {
            "path": "scripts/build_metro_panel.py",
            "current purpose": "Builds legacy annual unemployment panel, not housing.",
            "data inputs": "BLS LAUS raw files under data/metro",
            "data outputs": "data/metro/metro_unemployment.csv",
            "Zillow series used": "none",
            "covariates used": "none",
            "whether interpolation occurs": "no",
            "whether seasonal adjustment occurs": "uses BLS annual average, no monthly seasonal adjustment",
            "whether winsorization occurs": "no",
            "whether standardization occurs": "downstream only",
            "whether the code can be retained": "yes; unrelated to new housing acquisition",
            "required revision": "none for this task",
        },
        {
            "path": "tests/test_core.py",
            "current purpose": "Core unit tests for simulation/estimation/reporting helpers.",
            "data inputs": "synthetic fixtures only",
            "data outputs": "none",
            "Zillow series used": "none before this task",
            "covariates used": "none",
            "whether interpolation occurs": "no",
            "whether seasonal adjustment occurs": "no",
            "whether winsorization occurs": "no",
            "whether standardization occurs": "no",
            "whether the code can be retained": "yes",
            "required revision": "add fixture tests for housing audit helpers",
        },
    ]
    lines = ["# Existing Housing Code Audit", "", f"Generated: {utc_now()}", ""]
    for entry in entries:
        lines.append(f"## {entry['path']}")
        for key, value in entry.items():
            if key != "path":
                lines.append(f"- **{key}:** {value}")
        lines.append("")
    return "\n".join(lines)


def archive_invalid_existing(data_root: Path, inventory: Sequence[Dict[str, object]]) -> Tuple[Path, List[Dict[str, object]]]:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = data_root / f"archive_existing_{stamp}"
    archived = []
    legacy_names = {
        "zillow_metro_top.csv": "top-tier Zillow file, not all-homes",
        "zillow_metro_bottom.csv": "bottom-tier Zillow file, not all-homes",
        "metro_monthly_covariates_2000_present.csv": "legacy covariate file includes population/GDP monthly construction and transformed growth rates",
        "cbsa_county_crosswalk_2023.csv": "source URL/checksum provenance not sufficient for acceptance as clean raw input",
    }
    for name, reason in legacy_names.items():
        src = data_root / name
        if src.exists():
            dst = archive_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            archived.append({"relative_path": name, "archive_path": str(dst), "reason": reason})
    return archive_dir, archived


def write_inventory(data_root: Path, inventory: Sequence[Dict[str, object]]) -> None:
    audit = data_root / "audit"
    fields = [
        "relative_path", "file_type", "file_size", "sha256", "modification_time",
        "row_count", "column_count", "column_names", "geographic_level", "frequency",
        "earliest_date", "latest_date", "duplicate_keys", "missing_months",
        "apparent_source", "apparent_vintage", "seasonal_adjustment_status",
        "transformations_applied", "interpolation_may_have_occurred", "parser_error",
    ]
    atomic_write_csv(audit / "existing_data_inventory.csv", inventory, fields)
    lines = ["# Existing Zillow Data Inventory", "", f"Generated: {utc_now()}", ""]
    for row in inventory:
        lines.append(f"## {row['relative_path']}")
        for key in fields[1:]:
            val = row.get(key, "")
            if val != "":
                lines.append(f"- **{key}:** {val}")
        lines.append("")
    atomic_write_text(audit / "existing_data_inventory.md", "\n".join(lines))


def write_code_audit(root: Path, data_root: Path) -> None:
    atomic_write_text(data_root / "audit" / "existing_code_audit.md", audit_existing_code(root, data_root))


def discover_zillow_url() -> str:
    # Zillow occasionally changes paths; keep a verified current canonical path
    # and allow future extension to parse the data page when direct paths move.
    return ZILLOW_ALL_HOMES_URL


def fetch_zillow(dirs: Dict[str, Path], force: bool, manifest: List[DownloadRecord]) -> Optional[Path]:
    url = discover_zillow_url()
    dest = dirs["raw_zillow"] / Path(url).name
    rec = http_download(url, dest, force=force)
    rec.agency = "Zillow Research"
    rec.dataset = "ZHVI all homes metro smoothed seasonally adjusted"
    rec.official_source_page = ZILLOW_DATA_PAGE
    rec.release_vintage = "current download"
    rec.seasonal_adjustment_status = "source smoothed seasonally adjusted"
    manifest.append(rec)
    return dest if rec.status != "download-failed" else None


def parse_bps_monthly_file(content: bytes, source_file: str) -> Tuple[List[Dict[str, object]], int]:
    text = content.decode("latin1", errors="replace").splitlines()
    rows = []
    micro_count = 0
    for raw in csv.reader(text):
        if len(raw) < len(BPS_COLS):
            continue
        row = dict(zip(BPS_COLS, raw))
        period = re.sub(r"\D", "", row.get("period", ""))
        if len(period) < 6:
            continue
        ym = f"{period[:4]}-{period[4:6]}"
        cbsa = re.sub(r"\D", "", row.get("cbsa_code", "")).zfill(5)
        if cbsa == "99999" or not cbsa.strip("0"):
            continue
        header = row.get("header_code", "").strip()
        # Census header code 5 is the micropolitan section in these files.
        is_micro = header == "5" or "micropolitan" in row.get("cbsa_title", "").lower()
        if is_micro:
            micro_count += 1
            continue
        total = 0.0
        any_units = False
        for col in ("imp_101_units", "imp_103_units", "imp_104_units", "imp_105_units"):
            val = parse_float(row.get(col))
            if val is not None:
                total += val
                any_units = True
        if not any_units:
            continue
        rows.append({
            "cbsa_code": cbsa,
            "cbsa_title": row.get("cbsa_title", "").strip(),
            "metropolitan_or_micropolitan_flag": "metropolitan",
            "year": int(period[:4]),
            "month": int(period[4:6]),
            "date": ym + "-01",
            "total_units": int(total) if float(total).is_integer() else total,
            "source_vintage": "Census BPS monthly file",
            "source_file": source_file,
        })
    return rows, micro_count


def list_bps_urls() -> List[Tuple[str, str]]:
    urls = []
    for folder in BPS_FOLDERS:
        req = urllib.request.Request(folder, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=90) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        for href in re.findall(r'href="([^"]+\.txt)"', html, flags=re.I):
            low = href.lower()
            if re.match(r"ma\d{4}c\.txt$", low) or re.match(r"cbsa\d{4}c\.txt$", low):
                urls.append((urllib.parse.urljoin(folder, href), href))
    return sorted(set(urls))


def fetch_bps(dirs: Dict[str, Path], force: bool, manifest: List[DownloadRecord],
              warnings: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    all_rows = []
    micro_total = 0
    try:
        urls = list_bps_urls()
    except Exception as exc:
        warnings.append({"stage": "bps_list", "message": str(exc), "severity": "error"})
        return [], 0
    for url, filename in urls:
        dest = dirs["raw_bps"] / filename
        rec = http_download(url, dest, force=force)
        rec.agency = "U.S. Census Bureau"
        rec.dataset = "Building Permits Survey monthly CBSA/MSA file"
        rec.official_source_page = CENSUS_BPS_PAGE
        rec.release_vintage = filename
        rec.seasonal_adjustment_status = "not seasonally adjusted"
        manifest.append(rec)
        if rec.status == "download-failed":
            warnings.append({"stage": "bps_download", "source_file": filename, "message": rec.error, "severity": "error"})
            continue
        rows, micro = parse_bps_monthly_file(dest.read_bytes(), filename)
        all_rows.extend(rows)
        micro_total += micro
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in all_rows:
        key = (str(row["cbsa_code"]), str(row["date"]))
        if key not in by_key:
            by_key[key] = row
        else:
            by_key[key]["total_units"] = parse_float(by_key[key]["total_units"]) + parse_float(row["total_units"])
    out = sorted(by_key.values(), key=lambda r: (r["cbsa_code"], r["date"]))
    return out, micro_total


def _split_tab(line: str) -> List[str]:
    return [p.strip() for p in line.rstrip("\n").split("\t")]


def read_bls_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        header = _split_tab(next(fh))
        rows = []
        for line in fh:
            p = _split_tab(line)
            if len(p) < len(header):
                p.extend([""] * (len(header) - len(p)))
            rows.append({h: p[i] for i, h in enumerate(header)})
    return header, rows


def _is_individual_msa_area(area_code: str, area_name: str, metropolitan_codes: Optional[set] = None) -> bool:
    code = re.sub(r"\D", "", str(area_code)).zfill(5)
    name = str(area_name)
    low = name.lower()
    if not re.fullmatch(r"\d{5}", code):
        return False
    if code == "00000":
        return False
    if "statewide" in low or "all metropolitan statistical areas" in low:
        return False
    if "metropolitan division" in low:
        return False
    if metropolitan_codes is not None and code not in metropolitan_codes:
        return False
    return True


def parse_bls_series_metadata(dirs: Dict[str, Path], geo_rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, str]]:
    _, area_rows = read_bls_table(dirs["raw_bls"] / "sm.area")
    _, series_rows = read_bls_table(dirs["raw_bls"] / "sm.series")
    _, industry_rows = read_bls_table(dirs["raw_bls"] / "sm.industry")
    industries = {r.get("industry_code", ""): r.get("industry_name", "") for r in industry_rows}
    metropolitan_codes = {
        str(r.get("cbsa_code", "")).zfill(5) for r in geo_rows
        if "Metropolitan" in str(r.get("metro_micro_type", ""))
    }
    area_meta = {r.get("area_code", ""): r.get("area_name", "") for r in area_rows}
    sa, nsa = {}, {}
    for row in series_rows:
        sid = row.get("series_id", "")
        area_code = str(row.get("area_code", "")).zfill(5)
        area_name = area_meta.get(row.get("area_code", ""), row.get("area_name", ""))
        seasonal = row.get("seasonal", "").upper()
        industry_code = row.get("industry_code", "")
        industry_name = industries.get(industry_code, row.get("industry_name", ""))
        data_type = row.get("data_type_code", "")
        if not _is_individual_msa_area(area_code, area_name, metropolitan_codes):
            continue
        if row.get("supersector_code", "") not in {"00", ""}:
            continue
        if industry_code != "00000000":
            continue
        if "total nonfarm" not in industry_name.lower() and industry_name:
            continue
        if data_type != "01":
            continue
        if data_type == "26":
            continue
        meta = {
            "bls_series_id": sid,
            "bls_state_code": row.get("state_code", ""),
            "bls_area_code": area_code,
            "cbsa_code": area_code,
            "bls_area_title": area_name,
            "units": "All Employees, In Thousands",
            "seasonal_adjustment_flag": seasonal,
            "industry_code": industry_code,
            "industry_name": industry_name or "Total Nonfarm",
            "data_type_code": data_type,
            "source_file": "sm.data.54.TotalNonFarm.All",
            "source_vintage": "current BLS SM bulk download",
        }
        if seasonal == "S" and sid.startswith("SMS"):
            sa[sid] = meta
        elif seasonal == "U" and sid.startswith("SMU"):
            nsa[sid] = meta
    return sa, nsa, area_meta


def parse_bls_employment_data(data_rows: Sequence[Dict[str, str]],
                              official_ids: Dict[str, Dict[str, str]],
                              nsa_ids: Dict[str, Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    official_rows, nsa_rows = [], []
    for raw in data_rows:
        sid = raw.get("series_id", "")
        period = raw.get("period", "")
        if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
            continue
        meta = official_ids.get(sid) or nsa_ids.get(sid)
        if not meta:
            continue
        value = parse_float(raw.get("value"))
        if value is None:
            continue
        foot = raw.get("footnote_codes", "")
        row = dict(meta)
        row.update({
            "date": f"{int(raw.get('year')):04d}-{int(period[1:]):02d}-01",
            "employment_thousands_sa": value if sid in official_ids else "",
            "employment_thousands_nsa": value if sid in nsa_ids else "",
            "employment_level": value,
            "preliminary_flag": int("P" in str(foot).upper()),
            "footnote_codes": foot,
            "seasonal_adjustment_source": "official_bls" if sid in official_ids else "not_seasonally_adjusted",
        })
        if sid in official_ids:
            official_rows.append(row)
        else:
            nsa_rows.append(row)
    return (sorted(official_rows, key=lambda r: (r["cbsa_code"], r["date"])),
            sorted(nsa_rows, key=lambda r: (r["cbsa_code"], r["date"])))


def fetch_bls(dirs: Dict[str, Path], force: bool, manifest: List[DownloadRecord],
              warnings: List[Dict[str, object]],
              geo_rows: Sequence[Dict[str, object]],
              data_root: Optional[Path] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    diagnostics: List[Dict[str, object]] = []
    metadata_files = BLS_BULK_FILES[:-1]
    data_file = BLS_BULK_FILES[-1]
    for filename, url, dataset, min_size in metadata_files:
        rec = download_bls_bulk_file(filename, url, dataset, min_size,
                                     dirs["raw_bls"], force, diagnostics)
        manifest.append(rec)
        if rec.status == "download-failed":
            warnings.append({"stage": "bls_download", "source_file": filename, "message": rec.error, "severity": "error"})
            if data_root is not None:
                write_bls_download_diagnosis(data_root, diagnostics)
            return [], [], diagnostics
    official_ids, nsa_ids, _ = parse_bls_series_metadata(dirs, geo_rows)
    filename, url, dataset, min_size = data_file
    rec = download_bls_bulk_file(filename, url, dataset, min_size,
                                 dirs["raw_bls"], force, diagnostics)
    if rec.status == "download-failed":
        api_rec, api_diag = bls_api_fallback_get(
            sorted(set(official_ids) | set(nsa_ids)),
            dirs["raw_bls"] / filename,
            1990,
            _dt.date.today().year,
        )
        diagnostics.append(api_diag)
        if api_rec is not None:
            api_rec.dataset = dataset
            rec = api_rec
        else:
            warnings.append({"stage": "bls_download", "source_file": filename, "message": rec.error, "severity": "error"})
            manifest.append(rec)
            if data_root is not None:
                write_bls_download_diagnosis(data_root, diagnostics)
            return [], [], diagnostics
    manifest.append(rec)
    _, data_rows = read_bls_table(dirs["raw_bls"] / "sm.data.54.TotalNonFarm.All")
    official_rows, nsa_rows = parse_bls_employment_data(data_rows, official_ids, nsa_ids)
    if data_root is not None:
        write_bls_download_diagnosis(data_root, diagnostics)
    return official_rows, nsa_rows, diagnostics


def parse_xlsx_first_sheet(path: Path) -> List[List[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                txt = "".join(t.text or "" for t in si.findall(".//a:t", ns))
                shared.append(txt)
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(zf.read(sheet_name))
        rows = []
        for row in root.findall(".//a:row", ns):
            vals = []
            last_col = 0
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                col_letters = re.sub(r"\d", "", ref)
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + ord(ch.upper()) - ord("A") + 1
                while last_col + 1 < col_idx:
                    vals.append("")
                    last_col += 1
                v = c.find("a:v", ns)
                raw = "" if v is None else v.text or ""
                if c.attrib.get("t") == "s" and raw != "":
                    raw = shared[int(raw)]
                vals.append(raw)
                last_col = col_idx
            if any(str(x).strip() for x in vals):
                rows.append(vals)
        return rows


def fetch_geography(dirs: Dict[str, Path], force: bool, manifest: List[DownloadRecord],
                    warnings: List[Dict[str, object]]) -> List[Dict[str, object]]:
    dest = dirs["raw_geo"] / "list1_2023.xlsx"
    rec = http_download(CBSA_DELINEATION_URL, dest, force=force)
    rec.agency = "U.S. Census Bureau / OMB"
    rec.dataset = "2023 CBSA metropolitan and micropolitan delineation file"
    rec.official_source_page = CENSUS_DELINEATION_PAGE
    rec.release_vintage = "July 2023"
    rec.seasonal_adjustment_status = "not applicable"
    manifest.append(rec)
    if rec.status == "download-failed":
        warnings.append({"stage": "geography_download", "message": rec.error, "severity": "error"})
        return []
    rows = parse_xlsx_first_sheet(dest)
    header_i = None
    for i, row in enumerate(rows[:40]):
        low = [str(x).strip().lower() for x in row]
        if "cbsa code" in low and "cbsa title" in low:
            header_i = i
            break
    if header_i is None:
        warnings.append({"stage": "geography_parse", "message": "could not find CBSA Code/Title header", "severity": "error"})
        return []
    header = [str(x).strip() for x in rows[header_i]]
    idx = {h.lower(): i for i, h in enumerate(header)}
    out = []
    for raw in rows[header_i + 1:]:
        def get(name):
            i = idx.get(name.lower())
            return raw[i].strip() if i is not None and i < len(raw) else ""
        cbsa = re.sub(r"\D", "", get("CBSA Code")).zfill(5)
        title = get("CBSA Title")
        typ = get("Metropolitan/Micropolitan Statistical Area")
        county = get("County/County Equivalent")
        state = get("State Name")
        st = re.sub(r"\D", "", get("FIPS State Code")).zfill(2)
        co = re.sub(r"\D", "", get("FIPS County Code")).zfill(3)
        if not cbsa.strip("0") or not title:
            continue
        out.append({
            "cbsa_code": cbsa,
            "census_cbsa_title": title,
            "metropolitan_division_code": get("Metropolitan Division Code"),
            "metropolitan_division_title": get("Metropolitan Division Title"),
            "metro_micro_type": typ,
            "county_name": county,
            "state_name": state,
            "county_fips": st + co if st.strip("0") and co.strip("0") else "",
            "geographic_vintage": "2023 Census/OMB delineation",
        })
    return out


def find_x13_executable(data_root: Path) -> Optional[Path]:
    candidates = []
    for base in [data_root / "tools" / "x13", Path("tools") / "x13"]:
        if base.exists():
            candidates.extend(base.rglob("x13as*.exe"))
            candidates.extend(base.rglob("x13as"))
    path_env = os.environ.get("PATH", "")
    for part in path_env.split(os.pathsep):
        for name in ("x13as_ascii.exe", "x13as.exe", "x13as"):
            p = Path(part) / name
            if p.exists():
                candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    return None


def discover_x13_zip() -> Optional[str]:
    try:
        req = urllib.request.Request(X13_PAGE, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=90) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    hrefs = re.findall(r'href="([^"]+\.zip)"', html, flags=re.I)
    for href in hrefs:
        low = href.lower()
        if "x13as" in low and "ascii" in low:
            return urllib.parse.urljoin(X13_PAGE, href)
    for href in hrefs:
        if "x13as" in href.lower():
            return urllib.parse.urljoin(X13_PAGE, href)
    return None


def install_x13_if_possible(data_root: Path, manifest: List[DownloadRecord],
                            warnings: List[Dict[str, object]], force: bool = False) -> Optional[Path]:
    existing = find_x13_executable(data_root)
    if existing:
        return existing
    url = discover_x13_zip()
    if not url:
        warnings.append({"stage": "x13_install", "message": "could not discover official Census X-13 zip", "severity": "warning"})
        return None
    dest = data_root / "tools" / "x13" / Path(urllib.parse.urlparse(url).path).name
    rec = http_download(url, dest, force=force)
    rec.agency = "U.S. Census Bureau"
    rec.dataset = "X-13ARIMA-SEATS executable"
    rec.official_source_page = X13_PAGE
    rec.release_vintage = "current Census release"
    rec.seasonal_adjustment_status = "software"
    manifest.append(rec)
    if rec.status == "download-failed":
        warnings.append({"stage": "x13_install", "message": rec.error, "severity": "warning"})
        return None
    try:
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(data_root / "tools" / "x13")
    except Exception as exc:
        warnings.append({"stage": "x13_install", "message": f"downloaded but extraction failed: {exc}", "severity": "warning"})
        return None
    return find_x13_executable(data_root)


def run_x13_segment(exe: Path, cbsa: str, title: str, segment: Sequence[Tuple[str, float]],
                    work_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    start = segment[0][0]
    y, m = start[:7].split("-")
    stem = f"permits_{cbsa}_{start[:7].replace('-', '')}_{segment[-1][0][:7].replace('-', '')}"
    spc = work_dir / f"{stem}.spc"
    d11 = work_dir / f"{stem}.d11"
    data_lines = "\n".join(str(float(v)) for _, v in segment)
    spec = f"""series {{
  title = "{cbsa} permits"
  start = {int(y)}.{int(m)}
  period = 12
  data = (
{data_lines}
  )
}}
transform {{ function = none }}
regression {{ aictest = (td easter) }}
automdl {{}}
x11 {{ mode = add save = (d11) }}
"""
    atomic_write_text(spc, spec)
    if d11.exists():
        proc_returncode, proc_stdout, proc_stderr = 0, "", ""
    else:
        proc = subprocess.run([str(exe), str(spc.with_suffix(""))], cwd=work_dir,
                              capture_output=True, text=True, timeout=120, check=False)
        proc_returncode, proc_stdout, proc_stderr = proc.returncode, proc.stdout, proc.stderr
    rows = []
    status = "ok" if proc_returncode == 0 and d11.exists() else f"failed:{proc_returncode}"
    if d11.exists():
        values = []
        for line in d11.read_text(errors="replace").splitlines():
            parts = line.split()
            for part in parts:
                val = parse_float(part)
                if val is not None:
                    values.append(val)
        # X-13 d11 files include calendar columns in many formats; keep the last
        # len(segment) numeric values, which correspond to saved adjusted series.
        values = values[-len(segment):]
        if len(values) == len(segment):
            for (date, nsa), sa in zip(segment, values):
                rows.append({
                    "cbsa_code": cbsa,
                    "cbsa_title": title,
                    "date": date,
                    "permits_units_nsa": nsa,
                    "permits_units_sa": sa,
                    "x13_status": "ok",
                    "x13_spec_id": stem,
                    "contiguous_segment_start": segment[0][0],
                    "contiguous_segment_end": segment[-1][0],
                    "source_vintage": "Census BPS monthly file",
                })
        else:
            status = "failed:could_not_parse_d11"
    diag = {
        "series_id": cbsa,
        "x13_spec_id": stem,
        "status": status,
        "returncode": proc_returncode,
        "stderr": proc_stderr[-1000:],
        "stdout": proc_stdout[-1000:],
        "n_observed": len(segment),
        "segment_start": segment[0][0],
        "segment_end": segment[-1][0],
        "spec_path": str(spc),
    }
    if not rows:
        for date, nsa in segment:
            rows.append({
                "cbsa_code": cbsa,
                "cbsa_title": title,
                "date": date,
                "permits_units_nsa": nsa,
                "permits_units_sa": "",
                "x13_status": status,
                "x13_spec_id": stem,
                "contiguous_segment_start": segment[0][0],
                "contiguous_segment_end": segment[-1][0],
                "source_vintage": "Census BPS monthly file",
            })
    return rows, diag


def seasonal_adjust_permits(permits: Sequence[Dict[str, object]], data_root: Path,
                            min_months: int, skip_x13: bool, manifest: List[DownloadRecord],
                            warnings: List[Dict[str, object]], force: bool) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], bool]:
    exe = None if skip_x13 else install_x13_if_possible(data_root, manifest, warnings, force=force)
    if exe is None:
        diagnostics = [{
            "series_id": "",
            "x13_spec_id": "",
            "status": "skipped" if skip_x13 else "x13_unavailable",
            "returncode": "",
            "stderr": "",
            "stdout": "",
            "n_observed": 0,
            "segment_start": "",
            "segment_end": "",
            "spec_path": "",
        }]
        return [], diagnostics, False
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in permits:
        grouped.setdefault(str(row["cbsa_code"]), []).append(row)
    out, diagnostics = [], []
    for cbsa, rows in grouped.items():
        rows = sorted(rows, key=lambda r: str(r["date"]))
        by_month = {}
        for row in rows:
            ym = str(row["date"])[:7]
            if ym in by_month:
                diagnostics.append({"series_id": cbsa, "status": "duplicate_month", "segment_start": ym, "segment_end": ym, "n_observed": 0})
                continue
            by_month[ym] = row
        observed = sorted(by_month)
        for start, end, n in contiguous_segments(observed):
            if n < min_months:
                diagnostics.append({"series_id": cbsa, "status": "segment_too_short", "segment_start": start, "segment_end": end, "n_observed": n})
                continue
            segment = []
            for ym in month_range(start, end):
                row = by_month[ym]
                segment.append((row["date"], float(row["total_units"])))
            seg_rows, diag = run_x13_segment(exe, cbsa, rows[0].get("cbsa_title", ""), segment,
                                             data_root / "tools" / "x13" / "work")
            out.extend(seg_rows)
            diagnostics.append(diag)
    return sorted(out, key=lambda r: (r["cbsa_code"], r["date"])), diagnostics, True


def build_crosswalk(zillow_rows: Sequence[Dict[str, object]], bps_rows: Sequence[Dict[str, object]],
                    bls_rows: Sequence[Dict[str, object]], geo_rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    title_to_geo: Dict[str, List[Dict[str, object]]] = {}
    for row in geo_rows:
        if "Metropolitan" not in str(row.get("metro_micro_type", "")):
            continue
        title_to_geo.setdefault(normalize_title(row["census_cbsa_title"]), []).append(row)
    bps_by_code = {}
    for row in bps_rows:
        bps_by_code.setdefault(str(row["cbsa_code"]).zfill(5), row.get("cbsa_title", ""))
    bls_by_code = {}
    for row in bls_rows:
        bls_by_code.setdefault(str(row["cbsa_code"]).zfill(5), row.get("bls_area_title", row.get("area_title", "")))
    zmeta = {}
    for row in zillow_rows:
        zmeta[str(row["zillow_region_id"])] = row["zillow_region_name"]
    accepted, review = [], []
    for zid, zname in sorted(zmeta.items(), key=lambda kv: kv[1]):
        matches = title_to_geo.get(normalize_title(zname), [])
        base = {
            "zillow_region_id": zid,
            "zillow_region_name": zname,
            "cbsa_code": "",
            "census_cbsa_title": "",
            "bls_area_code": "",
            "bls_area_title": "",
            "match_method": "name_normalized_exact",
            "geographic_vintage": "2023 Census/OMB delineation",
            "match_status": "",
            "review_reason": "",
        }
        codes = sorted({m["cbsa_code"] for m in matches})
        if len(codes) == 1:
            code = codes[0]
            row = dict(base)
            row.update({
                "cbsa_code": code,
                "census_cbsa_title": matches[0]["census_cbsa_title"],
                "bls_area_code": code if code in bls_by_code else "",
                "bls_area_title": bls_by_code.get(code, ""),
                "match_status": "accepted" if code in bps_by_code or code in bls_by_code else "accepted_no_covariate_code_observed",
            })
            accepted.append(row)
        elif len(codes) > 1:
            row = dict(base)
            row["match_status"] = "manual_review"
            row["review_reason"] = f"ambiguous normalized title matched codes {codes}"
            review.append(row)
        else:
            row = dict(base)
            row["match_status"] = "manual_review"
            row["review_reason"] = "no exact normalized official CBSA title match"
            review.append(row)
    return accepted, review


def classify_zillow_geography(zillow_rows: Sequence[Dict[str, object]], bps_rows: Sequence[Dict[str, object]],
                              bls_rows: Sequence[Dict[str, object]], geo_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    zmeta = {}
    for row in zillow_rows:
        zmeta[str(row["zillow_region_id"])] = {
            "zillow_region_id": str(row["zillow_region_id"]),
            "zillow_region_name": str(row.get("zillow_region_name", "")),
            "state_name": str(row.get("state_name", "")),
            "region_type": str(row.get("region_type", "")),
        }
    title_map: Dict[str, List[Dict[str, object]]] = {}
    division_map: Dict[str, List[Dict[str, object]]] = {}
    for row in geo_rows:
        title_map.setdefault(normalize_title(row.get("census_cbsa_title", "")), []).append(row)
        div = str(row.get("metropolitan_division_title", "")).strip()
        if div:
            division_map.setdefault(normalize_title(div), []).append(row)
    bps_title_map = {}
    for row in bps_rows:
        bps_title_map.setdefault(normalize_title(row.get("cbsa_title", "")), str(row.get("cbsa_code", "")).zfill(5))
    bls_title_map = {}
    for row in bls_rows:
        bls_title_map.setdefault(normalize_title(row.get("bls_area_title", row.get("area_title", ""))), str(row.get("cbsa_code", "")).zfill(5))
    out = []
    for zid, meta in sorted(zmeta.items(), key=lambda kv: kv[1]["zillow_region_name"]):
        ztitle = meta["zillow_region_name"]
        norm = normalize_title(ztitle)
        row = dict(meta)
        row.update({
            "normalized_title": norm,
            "cbsa_code": "",
            "official_title": "",
            "classification": "unresolved",
            "classification_reason": "",
            "deterministic_rule": "",
        })
        if meta["region_type"].lower() != "msa":
            row["classification"] = "non_cbsa_zillow_region"
            row["classification_reason"] = "Zillow RegionType is not msa"
        else:
            matches = title_map.get(norm, [])
            codes = sorted({str(m.get("cbsa_code", "")).zfill(5) for m in matches if m.get("cbsa_code")})
            types = {str(m.get("metro_micro_type", "")) for m in matches}
            if len(codes) == 1 and any("Metropolitan" in t for t in types):
                row["classification"] = "current_metropolitan_cbsa"
                row["cbsa_code"] = codes[0]
                row["official_title"] = matches[0].get("census_cbsa_title", "")
                row["classification_reason"] = "exact normalized Census CBSA title match"
                row["deterministic_rule"] = "normalized_title_exact"
            elif len(codes) == 1 and any("Micropolitan" in t for t in types):
                row["classification"] = "current_micropolitan_cbsa"
                row["cbsa_code"] = codes[0]
                row["official_title"] = matches[0].get("census_cbsa_title", "")
                row["classification_reason"] = "exact normalized Census micropolitan title match"
                row["deterministic_rule"] = "normalized_title_exact"
            elif norm in division_map:
                div_matches = division_map[norm]
                row["classification"] = "metropolitan_division"
                row["cbsa_code"] = str(div_matches[0].get("cbsa_code", "")).zfill(5)
                row["official_title"] = div_matches[0].get("metropolitan_division_title", "")
                row["classification_reason"] = "exact normalized Census metropolitan division title match"
                row["deterministic_rule"] = "division_title_exact"
            elif norm in bps_title_map or norm in bls_title_map:
                row["classification"] = "historical_or_retired_cbsa"
                row["cbsa_code"] = bps_title_map.get(norm, bls_title_map.get(norm, ""))
                row["official_title"] = ztitle
                row["classification_reason"] = "observed in BPS/BLS but not current 2023 CBSA title"
                row["deterministic_rule"] = "legacy_source_title_exact"
            elif len(codes) > 1:
                row["classification"] = "unresolved"
                row["classification_reason"] = f"ambiguous current CBSA title match: {codes}"
            else:
                row["classification_reason"] = "no deterministic exact-code or exact-title match"
        out.append(row)
    return out


def build_availability(zillow: Sequence[Dict[str, object]], permits_nsa: Sequence[Dict[str, object]],
                       permits_sa: Sequence[Dict[str, object]], emp_sa: Sequence[Dict[str, object]],
                       emp_local: Sequence[Dict[str, object]], crosswalk: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    z_by_code: Dict[str, set] = {}
    id_to_code = {str(r["zillow_region_id"]): str(r["cbsa_code"]).zfill(5) for r in crosswalk if r.get("cbsa_code")}
    for row in zillow:
        code = id_to_code.get(str(row["zillow_region_id"]))
        if code and row.get("zhvi_all_homes_sa") not in ("", None):
            z_by_code.setdefault(code, set()).add(str(row["date"])[:7])
    def code_month_set(rows, value_col):
        out = {}
        for r in rows:
            val = r.get(value_col)
            if val not in ("", None):
                out.setdefault(str(r["cbsa_code"]).zfill(5), set()).add(str(r["date"])[:7])
        return out
    p_nsa = code_month_set(permits_nsa, "total_units")
    p_sa = code_month_set(permits_sa, "permits_units_sa")
    e_sa = code_month_set(emp_sa, "employment_thousands_sa")
    e_loc = code_month_set(emp_local, "employment_thousands_sa")
    codes = sorted(set(z_by_code) | set(p_nsa) | set(p_sa) | set(e_sa) | set(e_loc))
    months = sorted(set().union(*(z_by_code.get(c, set()) | p_nsa.get(c, set()) | p_sa.get(c, set()) | e_sa.get(c, set()) | e_loc.get(c, set()) for c in codes))) if codes else []
    status = {str(r["cbsa_code"]).zfill(5): r.get("match_status", "") for r in crosswalk}
    rows = []
    for code in codes:
        for ym in months:
            rows.append({
                "cbsa_code": code,
                "date": ym + "-01",
                "zhvi_sa_available": int(ym in z_by_code.get(code, set())),
                "permits_nsa_available": int(ym in p_nsa.get(code, set())),
                "permits_sa_available": int(ym in p_sa.get(code, set())),
                "employment_official_sa_available": int(ym in e_sa.get(code, set())),
                "employment_local_x13_sa_available": int(ym in e_loc.get(code, set())),
                "geography_match_status": status.get(code, ""),
            })
    return rows


def coverage_by_msa(availability: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in availability:
        grouped.setdefault(str(row["cbsa_code"]), []).append(row)
    out = []
    for code, rows in grouped.items():
        rows = sorted(rows, key=lambda r: str(r["date"]))
        rec = {"cbsa_code": code}
        for prefix, col in [
            ("zhvi", "zhvi_sa_available"),
            ("permits_nsa", "permits_nsa_available"),
            ("permits_sa", "permits_sa_available"),
            ("employment_official_sa", "employment_official_sa_available"),
            ("employment_local_x13_sa", "employment_local_x13_sa_available"),
        ]:
            months = [r["date"][:7] for r in rows if int(r.get(col, 0))]
            segs = contiguous_segments(months)
            rec[f"{prefix}_earliest"] = min(months, default="")
            rec[f"{prefix}_latest"] = max(months, default="")
            rec[f"{prefix}_internal_missing_months"] = len(detect_missing_months(months))
            rec[f"{prefix}_longest_contiguous"] = max((s[2] for s in segs), default=0)
        out.append(rec)
    return out


def balanced_panel_frontier(availability: Sequence[Dict[str, object]],
                            thresholds: Sequence[int] = (120, 180, 240)) -> List[Dict[str, object]]:
    codes = sorted({str(r["cbsa_code"]) for r in availability})
    months = sorted({str(r["date"])[:7] for r in availability})
    by = {(str(r["cbsa_code"]), str(r["date"])[:7]): r for r in availability}
    rows = []
    for L in thresholds:
        best = None
        for i in range(0, max(len(months) - L + 1, 0)):
            for j in range(i + L - 1, len(months)):
                window = months[i:j + 1]
                official = []
                fallback = []
                miss_z = miss_p = miss_e = 0
                for code in codes:
                    ok_z = all(int(by.get((code, m), {}).get("zhvi_sa_available", 0)) for m in window)
                    ok_p = all(int(by.get((code, m), {}).get("permits_sa_available", 0)) for m in window)
                    ok_e = all(int(by.get((code, m), {}).get("employment_official_sa_available", 0)) for m in window)
                    ok_el = all(int(by.get((code, m), {}).get("employment_local_x13_sa_available", 0)) for m in window)
                    if ok_z and ok_p and ok_e:
                        official.append(code)
                    if ok_z and ok_p and (ok_e or ok_el):
                        fallback.append(code)
                    miss_z += int(not ok_z)
                    miss_p += int(not ok_p)
                    miss_e += int(not ok_e)
                cand = {
                    "min_window_months": L,
                    "start_date": window[0] + "-01",
                    "end_date": window[-1] + "-01",
                    "T_months": len(window),
                    "N_complete_official_sa": len(official),
                    "N_complete_with_local_employment_fallback": len(fallback),
                    "N_missing_zhvi": miss_z,
                    "N_missing_permits": miss_p,
                    "N_missing_employment": miss_e,
                    "N_unresolved_geography": 0,
                    "official_sa_cbsa_codes": ";".join(official),
                    "fallback_cbsa_codes": ";".join(fallback),
                }
                if best is None or (cand["N_complete_official_sa"], cand["T_months"]) > (best["N_complete_official_sa"], best["T_months"]):
                    best = cand
        if best is not None:
            rows.append(best)
    return rows


def write_overlap_report(data_root: Path, summary: Dict[str, object],
                         accepted_existing: Sequence[str], archived: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Housing MSA Overlap Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Required Questions",
        f"1. Existing files accepted: {', '.join(accepted_existing) if accepted_existing else 'none'}",
        f"2. Existing files archived/rejected: {len(archived)}. " + "; ".join(f"{a['relative_path']} ({a['reason']})" for a in archived[:20]),
        f"3. Required raw data downloaded programmatically: {summary.get('raw_downloads_complete')}",
        f"4. X-13 requires manual installation: {summary.get('x13_manual_install_required')}",
        f"5. Zillow all-homes metros available: {summary.get('n_zillow_metros')}",
        f"6. Matched to permits: {summary.get('n_matched_permits')}",
        f"7. Matched to official SA employment: {summary.get('n_matched_official_sa_employment')}",
        f"8. Matched to all three official-primary series: {summary.get('n_matched_all_three_official')}",
        f"9. Additional metros with local-X-13 employment fallback: {summary.get('n_additional_with_local_employment')}",
        f"10. Common start/end dates: {summary.get('common_start_date')} to {summary.get('common_end_date')}",
        f"11. Feasible balanced panels: see balanced_panel_frontier.csv",
        f"12. Internal missing observations: see series_coverage_by_msa.csv",
        f"13. Geographic matches requiring manual review: {summary.get('n_manual_review_matches')}",
        "14. Interpolation or imputation performed: no.",
        "",
        "No winsorization, standardization, demeaning, interpolation, endpoint filling, forecast, or backcast values are produced by this audit pipeline.",
    ]
    atomic_write_text(data_root / "audit" / "overlap_report.md", "\n".join(lines) + "\n")


def run_housing_audit(data_root: Path, *, audit_existing_only: bool = False,
                      clean_download: bool = False, reuse_validated: bool = True,
                      force_download: bool = False, min_sa_months: int = 84,
                      skip_x13: bool = False, official_employment_only: bool = False,
                      allow_local_employment_x13: bool = False,
                      retry_source: Optional[str] = None) -> Tuple[int, Dict[str, object]]:
    if retry_source not in (None, "bls"):
        raise ValueError("retry_source currently supports only None or 'bls'")
    dirs = ensure_dirs(data_root)
    warnings: List[Dict[str, object]] = []
    manifest: List[DownloadRecord] = []
    write_code_audit(Path(__file__).resolve().parents[1], data_root)
    inventory = inventory_existing_data(data_root)
    write_inventory(data_root, inventory)
    archive_dir, archived = (data_root / "archive_existing_skipped", [])
    if retry_source is None:
        archive_dir, archived = archive_invalid_existing(data_root, inventory)
    accepted_existing: List[str] = []
    if audit_existing_only:
        atomic_write_csv(data_root / "audit" / "parser_warnings.csv", warnings, ["stage", "source_file", "message", "severity"])
        atomic_write_json(data_root / "audit" / "source_manifest.json", [r.as_dict() for r in manifest])
        return 0, {"archived": archived, "accepted_existing": accepted_existing}

    force = bool(force_download)
    zillow_rows, zsum = ([], {})
    permits_nsa, micro_count = ([], 0)
    permits_sa: List[Dict[str, object]] = []
    x13_diag: List[Dict[str, object]] = []
    x13_ok = True
    geo: List[Dict[str, object]] = []
    emp_sa: List[Dict[str, object]] = []
    emp_nsa: List[Dict[str, object]] = []
    bls_diagnostics: List[Dict[str, object]] = []

    if retry_source == "bls":
        zillow_path = dirs["raw_zillow"] / Path(ZILLOW_ALL_HOMES_URL).name
        if zillow_path.exists():
            accepted_existing.append(str(zillow_path.relative_to(data_root)))
        else:
            warnings.append({"stage": "zillow_cache", "source_file": str(zillow_path), "message": "cached Zillow raw file missing", "severity": "error"})
        permits_nsa = read_simple_csv(dirs["processed"] / "permits_metro_nsa_long.csv")
        permits_sa = read_simple_csv(dirs["processed"] / "permits_metro_sa_long.csv")
        x13_diag = read_simple_csv(data_root / "audit" / "x13_diagnostics.csv")
        geo = fetch_geography(dirs, False, manifest, warnings)
        if permits_nsa:
            accepted_existing.append("processed/permits_metro_nsa_long.csv")
        if permits_sa:
            accepted_existing.append("processed/permits_metro_sa_long.csv")
        if x13_diag:
            accepted_existing.append("audit/x13_diagnostics.csv")
    else:
        zillow_path = fetch_zillow(dirs, force, manifest) if clean_download else None
        permits_nsa, micro_count = fetch_bps(dirs, force, manifest, warnings) if clean_download else ([], 0)
        geo = fetch_geography(dirs, force, manifest, warnings) if clean_download else []

    if zillow_path and zillow_path.exists():
        try:
            zillow_rows, zsum = parse_zillow_all_homes(zillow_path, source_vintage="current Zillow Research download")
        except Exception as exc:
            warnings.append({"stage": "zillow_parse", "source_file": str(zillow_path), "message": str(exc), "severity": "error"})

    if clean_download or retry_source == "bls":
        emp_sa, emp_nsa, bls_diagnostics = fetch_bls(dirs, force, manifest, warnings, geo, data_root)

    fields_z = ["zillow_region_id", "zillow_region_name", "region_type", "state_name", "date", "zhvi_all_homes_sa", "source_file", "source_vintage"]
    atomic_write_csv(dirs["processed"] / "zhvi_all_homes_metro_sa_long.csv", zillow_rows, fields_z)
    fields_p = ["cbsa_code", "cbsa_title", "metropolitan_or_micropolitan_flag", "year", "month", "date", "total_units", "source_vintage", "source_file"]
    atomic_write_csv(dirs["processed"] / "permits_metro_nsa_long.csv", permits_nsa, fields_p)
    fields_e_sa = [
        "bls_series_id", "bls_state_code", "bls_area_code", "cbsa_code",
        "bls_area_title", "date", "employment_thousands_sa",
        "preliminary_flag", "footnote_codes", "source_file", "source_vintage",
        "seasonal_adjustment_source",
    ]
    fields_e_nsa = [
        "bls_series_id", "bls_state_code", "bls_area_code", "cbsa_code",
        "bls_area_title", "date", "employment_thousands_nsa",
        "preliminary_flag", "footnote_codes", "source_file", "source_vintage",
        "seasonal_adjustment_source",
    ]
    atomic_write_csv(dirs["processed"] / "employment_metro_official_sa_long.csv", emp_sa, fields_e_sa)
    atomic_write_csv(dirs["processed"] / "employment_metro_nsa_availability_long.csv", emp_nsa, fields_e_nsa)

    fields_psa = ["cbsa_code", "cbsa_title", "date", "permits_units_nsa", "permits_units_sa", "x13_status", "x13_spec_id", "contiguous_segment_start", "contiguous_segment_end", "source_vintage"]
    if retry_source != "bls":
        permits_sa, x13_diag, x13_ok = seasonal_adjust_permits(permits_nsa, data_root, min_sa_months,
                                                               skip_x13, manifest, warnings, force)
        atomic_write_csv(dirs["processed"] / "permits_metro_sa_long.csv", permits_sa, fields_psa)
        atomic_write_csv(data_root / "audit" / "x13_diagnostics.csv", x13_diag,
                         ["series_id", "x13_spec_id", "status", "returncode", "stderr", "stdout", "n_observed", "segment_start", "segment_end", "spec_path"])
    else:
        x13_ok = bool(permits_sa)

    emp_local: List[Dict[str, object]] = []
    atomic_write_csv(dirs["processed"] / "employment_metro_local_x13_sa_long.csv", emp_local, fields_e_sa)

    crosswalk, manual = build_crosswalk(zillow_rows, permits_nsa, emp_sa, geo)
    cross_fields = ["zillow_region_id", "zillow_region_name", "cbsa_code", "census_cbsa_title", "bls_area_code", "bls_area_title", "match_method", "geographic_vintage", "match_status", "review_reason"]
    atomic_write_csv(dirs["processed"] / "housing_msa_crosswalk.csv", crosswalk, cross_fields)
    atomic_write_csv(data_root / "audit" / "msa_matches_manual_review.csv", manual, cross_fields)
    geo_class = classify_zillow_geography(zillow_rows, permits_nsa, emp_sa + emp_nsa, geo)
    geo_class_fields = [
        "zillow_region_id", "zillow_region_name", "state_name", "region_type",
        "normalized_title", "cbsa_code", "official_title", "classification",
        "classification_reason", "deterministic_rule",
    ]
    atomic_write_csv(data_root / "audit" / "zillow_geography_classification.csv", geo_class, geo_class_fields)

    availability = build_availability(zillow_rows, permits_nsa, permits_sa, emp_sa, emp_local, crosswalk)
    av_fields = ["cbsa_code", "date", "zhvi_sa_available", "permits_nsa_available", "permits_sa_available", "employment_official_sa_available", "employment_local_x13_sa_available", "geography_match_status"]
    atomic_write_csv(dirs["processed"] / "housing_msa_monthly_availability.csv", availability, av_fields)
    coverage = coverage_by_msa(availability)
    cov_fields = ["cbsa_code", "zhvi_earliest", "zhvi_latest", "zhvi_internal_missing_months", "zhvi_longest_contiguous", "permits_nsa_earliest", "permits_nsa_latest", "permits_nsa_internal_missing_months", "permits_nsa_longest_contiguous", "permits_sa_earliest", "permits_sa_latest", "permits_sa_internal_missing_months", "permits_sa_longest_contiguous", "employment_official_sa_earliest", "employment_official_sa_latest", "employment_official_sa_internal_missing_months", "employment_official_sa_longest_contiguous", "employment_local_x13_sa_earliest", "employment_local_x13_sa_latest", "employment_local_x13_sa_internal_missing_months", "employment_local_x13_sa_longest_contiguous"]
    atomic_write_csv(data_root / "audit" / "series_coverage_by_msa.csv", coverage, cov_fields)
    frontier = balanced_panel_frontier(availability)
    frontier_fields = ["min_window_months", "start_date", "end_date", "T_months", "N_complete_official_sa", "N_complete_with_local_employment_fallback", "N_missing_zhvi", "N_missing_permits", "N_missing_employment", "N_unresolved_geography", "official_sa_cbsa_codes", "fallback_cbsa_codes"]
    atomic_write_csv(data_root / "audit" / "balanced_panel_frontier.csv", frontier, frontier_fields)

    cw_codes = {str(r["cbsa_code"]).zfill(5) for r in crosswalk}
    p_codes = {str(r["cbsa_code"]).zfill(5) for r in permits_nsa}
    psa_codes = {str(r["cbsa_code"]).zfill(5) for r in permits_sa}
    e_codes = {str(r["cbsa_code"]).zfill(5) for r in emp_sa}
    matched_permits = cw_codes & p_codes
    matched_emp = cw_codes & e_codes
    matched_all = cw_codes & psa_codes & e_codes
    common_months = [r["date"][:7] for r in availability
                     if int(r["zhvi_sa_available"]) and int(r["permits_sa_available"]) and int(r["employment_official_sa_available"])]
    summary = {
        "raw_downloads_complete": bool(zillow_rows and permits_nsa and emp_sa and geo),
        "x13_manual_install_required": not x13_ok and not skip_x13,
        "n_zillow_metros": zsum.get("n_metros", 0),
        "n_matched_permits": len(matched_permits),
        "n_matched_official_sa_employment": len(matched_emp),
        "n_matched_all_three_official": len(matched_all),
        "n_additional_with_local_employment": 0,
        "common_start_date": (min(common_months) + "-01") if common_months else "",
        "common_end_date": (max(common_months) + "-01") if common_months else "",
        "n_manual_review_matches": len(manual),
        "n_micropolitan_permit_rows_excluded": micro_count,
        "n_archived_existing_files": len(archived),
        "n_bls_official_sa_series": len({r.get("bls_series_id", "") for r in emp_sa}),
        "n_bls_nsa_series": len({r.get("bls_series_id", "") for r in emp_nsa}),
        "employment_earliest_date": min((str(r.get("date", "")) for r in emp_sa), default=""),
        "employment_latest_date": max((str(r.get("date", "")) for r in emp_sa), default=""),
        "n_zillow_current_metropolitan_cbsa": sum(1 for r in geo_class if r.get("classification") == "current_metropolitan_cbsa"),
        "n_zillow_micropolitan_or_non_msa": sum(1 for r in geo_class if r.get("classification") in {"current_micropolitan_cbsa", "non_cbsa_zillow_region", "metropolitan_division"}),
        "bls_transport_methods": ";".join(sorted({str(r.get("transport")) for r in bls_diagnostics if r.get("transport")})),
    }
    atomic_write_csv(data_root / "audit" / "overlap_summary.csv", [summary], list(summary.keys()))
    atomic_write_csv(data_root / "audit" / "parser_warnings.csv", warnings,
                     ["stage", "source_file", "message", "severity"])
    atomic_write_json(data_root / "audit" / "source_manifest.json", [r.as_dict() for r in manifest])
    log = "\n".join(json.dumps(r.as_dict(), sort_keys=True) for r in manifest)
    atomic_write_text(data_root / "audit" / "download_log.txt", log + ("\n" if log else ""))
    write_overlap_report(data_root, summary, accepted_existing, archived)
    if summary["x13_manual_install_required"]:
        warnings.append({"stage": "x13", "message": f"Install official Census X-13 from {X13_PAGE} into data/zillow/tools/x13/", "severity": "error"})
        atomic_write_csv(data_root / "audit" / "parser_warnings.csv", warnings,
                         ["stage", "source_file", "message", "severity"])
        return 2, summary
    if not summary["raw_downloads_complete"]:
        return 1, summary
    return 0, summary
