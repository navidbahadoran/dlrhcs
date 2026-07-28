#!/usr/bin/env python3
"""Audit/download the clean all-homes monthly MSA housing data inputs.

This script does not run the empirical estimator.  It preserves the legacy
top/bottom Zillow implementation and writes audit-ready raw, processed, and
coverage files under ``data/zillow``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from dlrhcs.housing_data import run_housing_audit  # noqa: E402
from dlrhcs.paths import find_repo_root, resolve_repo_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit and acquire monthly MSA housing data.")
    ap.add_argument("--repo-root", default=None,
                    help="explicit DLRHCS repository root; default uses DLRHCS_ROOT or script discovery")
    ap.add_argument("--data-root", default="data/zillow",
                    help="repository-relative or absolute data/zillow root")
    ap.add_argument("--audit-existing-only", action="store_true",
                    help="only inventory/audit existing files; do not download or process replacements")
    ap.add_argument("--clean-download", action="store_true",
                    help="download official raw replacements for missing or invalid inputs")
    ap.add_argument("--reuse-validated", action="store_true",
                    help="reuse validated raw files already present under data-root/raw")
    ap.add_argument("--force-download", action="store_true",
                    help="redownload official raw files even if cached")
    ap.add_argument("--min-sa-months", type=int, default=84,
                    help="minimum contiguous observed months required before X-13 adjustment")
    ap.add_argument("--skip-x13", action="store_true",
                    help="skip X-13 seasonal adjustment and mark the stage incomplete")
    ap.add_argument("--official-employment-only", action="store_true",
                    help="produce overlap based on official BLS SA employment only")
    ap.add_argument("--allow-local-employment-x13", action="store_true",
                    help="allow separately labeled local X-13 employment fallback diagnostics")
    ap.add_argument("--retry-source", choices=["bls"], default=None,
                    help="retry one source and recompute dependent overlap/audit outputs; currently supports bls")
    ap.add_argument("--bls-local-dir", default=None,
                    help="directory containing the six manually downloaded official BLS SM bulk files")
    args = ap.parse_args()
    if args.min_sa_months < 1:
        raise SystemExit("--min-sa-months must be positive")
    try:
        repo_root = find_repo_root(start=__file__, explicit=args.repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc))
    data_root = resolve_repo_path(args.data_root, repo_root)
    bls_local_dir = Path(args.bls_local_dir) if args.bls_local_dir else None
    if bls_local_dir is not None:
        bls_local_dir = resolve_repo_path(bls_local_dir, repo_root)
    code, summary = run_housing_audit(
        data_root,
        audit_existing_only=args.audit_existing_only,
        clean_download=args.clean_download,
        reuse_validated=args.reuse_validated,
        force_download=args.force_download,
        min_sa_months=args.min_sa_months,
        skip_x13=args.skip_x13,
        official_employment_only=args.official_employment_only,
        allow_local_employment_x13=args.allow_local_employment_x13,
        retry_source=args.retry_source,
        bls_local_dir=bls_local_dir,
    )
    print("[housing-audit] summary")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    if code:
        print(f"[housing-audit] incomplete; see {data_root / 'audit'}")
    else:
        print(f"[housing-audit] complete; see {data_root / 'audit'}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
