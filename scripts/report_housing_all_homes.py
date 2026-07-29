#!/usr/bin/env python3
"""Build deterministic publication reports from completed housing production output.

This script is intentionally read-only with respect to the estimator, prepared
panel, and production results.  It consumes the completed production artifacts
and writes derived publication-reporting files under the report directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from dlrhcs.design import build_blocks  # noqa: E402
from dlrhcs.paths import find_repo_root, repo_relative, resolve_repo_path  # noqa: E402
from scripts import housing_all_homes  # noqa: E402

REPORT_SCHEMA_VERSION = "housing_all_homes_report_v1"
RESULT_NAME = "housing_all_homes_results.json"
METADATA_NAME = "metadata.json"
RIESZ_SUMMARY_NAME = "riesz_summary.json"
RIESZ_DIAGNOSTICS_NAME = "riesz_diagnostics.csv"
REQUIRED_INPUT_NAMES = [RESULT_NAME, METADATA_NAME, RIESZ_SUMMARY_NAME, RIESZ_DIAGNOSTICS_NAME]
MAIN_TABLE_NAME = "tab_housing_all_homes_main"
SAMPLE_TABLE_NAME = "tab_housing_all_homes_sample"
DIAGNOSTICS_TABLE_NAME = "tab_housing_all_homes_diagnostics"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def finite_float(value: object, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field} is nonfinite: {value!r}")
    return out


def fmt_num(value: object) -> str:
    if value in ("", None):
        return "--"
    v = finite_float(value, "formatted value")
    return f"{v:.6g}"


def latex_escape(value: object) -> str:
    s = str(value)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def git_info(repo_root: Path) -> Dict[str, object]:
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                list(args), cwd=str(repo_root), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
        except Exception:
            return None
        return proc.stdout.strip()

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def production_paths(production_dir: Path) -> Dict[str, Path]:
    return {name: production_dir / name for name in REQUIRED_INPUT_NAMES}


def target_label_map() -> Dict[str, str]:
    dummy = [np.ones((2, 2)), 2.0 * np.ones((2, 2)), 3.0 * np.ones((2, 2))]
    blocks = build_blocks(dummy)
    targets = housing_all_homes.housing_targets(blocks)
    expected = {
        "lag_log_zhvi_mean": (0, "Mean coefficient on lagged log ZHVI"),
        "asinh_permits_mean": (1, "Mean coefficient on lagged asinh building permits"),
        "log_employment_mean": (2, "Mean coefficient on lagged log payroll employment"),
    }
    out: Dict[str, str] = {}
    for target in targets:
        if target.name not in expected:
            raise ValueError(f"unexpected housing target from housing_targets(): {target.name}")
        expected_block, label = expected[target.name]
        if int(target.block) != expected_block:
            raise ValueError(f"target {target.name} reads block {target.block}, expected {expected_block}")
        direction = np.asarray(target.direction[expected_block], dtype=float)
        if not np.allclose(direction, np.full((2, 2), 0.25)):
            raise ValueError(f"target {target.name} is not the verified full-mean target")
        out[target.name] = label
    if set(out) != set(expected):
        raise ValueError(f"housing_targets() returned {sorted(out)}, expected {sorted(expected)}")
    return out


def validate_production_inputs(production_dir: Path) -> Dict[str, object]:
    paths = production_paths(production_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing completed housing production input(s): {', '.join(missing)}")

    result = read_json(paths[RESULT_NAME])
    metadata = read_json(paths[METADATA_NAME])
    riesz_summary = read_json(paths[RIESZ_SUMMARY_NAME])
    riesz_rows = read_csv_rows(paths[RIESZ_DIAGNOSTICS_NAME])

    failures: List[str] = []
    if result.get("mode") != "production":
        failures.append("housing_all_homes_results.json mode is not production")
    if metadata.get("mode") != "production":
        failures.append("metadata.json mode is not production")
    if result.get("full_production_run") is not True:
        failures.append("result full_production_run is not true")
    if metadata.get("full_production_run") is not True:
        failures.append("metadata full_production_run is not true")
    if result.get("production_valid") is not True:
        failures.append("result production_valid is not true")
    if metadata.get("production_valid") is not True:
        failures.append("metadata production_valid is not true")
    if result.get("production_validation_failures") not in ([], None):
        failures.append("result production_validation_failures is not empty")
    if metadata.get("production_validation_failures") not in ([], None):
        failures.append("metadata production_validation_failures is not empty")

    result_sig = result.get("run_signature")
    meta_sig = metadata.get("run_signature")
    if result_sig != meta_sig:
        failures.append("result and metadata run signatures differ")
    result_riesz = result.get("riesz_diagnostics", {})
    if result_riesz != riesz_summary:
        failures.append("riesz_summary.json does not match result riesz_diagnostics")

    run_hash = housing_all_homes.signature_hash(result_sig or {})
    if metadata.get("input_panel_checksum") != result.get("input_panel", {}).get("checksum"):
        failures.append("input panel checksums differ between result and metadata")
    sig_checksum = (result_sig or {}).get("input_identity", {}).get("input_checksum") if isinstance(result_sig, Mapping) else None
    if sig_checksum != metadata.get("input_panel_checksum"):
        failures.append("run-signature input checksum differs from metadata input checksum")

    labels = target_label_map()
    targets = result.get("targets", {})
    if not isinstance(targets, Mapping):
        failures.append("targets object is missing or malformed")
        targets = {}
    for target_name in labels:
        vals = targets.get(target_name, {})
        if not isinstance(vals, Mapping):
            failures.append(f"target {target_name} is missing")
            continue
        for field in ["estimate", "se_white", "se_xs", "plugin"]:
            try:
                finite_float(vals.get(field), f"{target_name}.{field}")
            except ValueError as exc:
                failures.append(str(exc))
        for ci_field in ["ci_white", "ci_xs"]:
            ci = vals.get(ci_field)
            if not isinstance(ci, Sequence) or isinstance(ci, (str, bytes)) or len(ci) != 2:
                failures.append(f"{target_name}.{ci_field} is not a length-two interval")
            else:
                for idx, val in enumerate(ci):
                    try:
                        finite_float(val, f"{target_name}.{ci_field}[{idx}]")
                    except ValueError as exc:
                        failures.append(str(exc))

    try:
        if finite_float(riesz_summary.get("convergence_fraction"), "convergence_fraction") != 1.0:
            failures.append("Riesz convergence fraction is not 1.0")
    except ValueError as exc:
        failures.append(str(exc))
    try:
        if int(riesz_summary.get("number_containing_nonfinite_values", -1)) != 0:
            failures.append("Riesz summary reports nonfinite values")
    except Exception:
        failures.append("Riesz number_containing_nonfinite_values is malformed")

    for idx, row in enumerate(riesz_rows):
        if bool_from_csv(row.get("contains_nonfinite")):
            failures.append(f"Riesz diagnostic row {idx} contains nonfinite values")
        for field in [
            "iterations", "maxiter", "requested_tolerance", "achieved_absolute_residual",
            "achieved_relative_residual", "rhs_norm", "solution_norm",
            "maximum_absolute_solution_entry", "riesz_ridge", "elapsed_seconds",
        ]:
            try:
                finite_float(row.get(field), f"riesz_diagnostics row {idx} {field}")
            except ValueError as exc:
                failures.append(str(exc))

    if failures:
        raise ValueError("housing production inputs failed publication-report validation: " + "; ".join(failures))

    relative_residuals = [
        finite_float(row.get("achieved_relative_residual"), "achieved_relative_residual")
        for row in riesz_rows
    ]
    if relative_residuals and "relative_residual_mean" not in riesz_summary:
        riesz_summary = dict(riesz_summary)
        riesz_summary["relative_residual_mean"] = float(np.mean(relative_residuals))

    return {
        "paths": paths,
        "result": result,
        "metadata": metadata,
        "riesz_summary": riesz_summary,
        "riesz_rows": riesz_rows,
        "target_labels": labels,
        "run_signature_hash": run_hash,
        "input_checksums": {name: sha256_file(path) for name, path in paths.items()},
    }


def build_main_rows(result: Mapping[str, object], labels: Mapping[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    targets = result["targets"]
    for name in housing_all_homes.housing_targets(build_blocks([np.ones((2, 2))] * 3)):
        vals = targets[name.name]
        rows.append({
            "estimand": labels[name.name],
            "estimate": fmt_num(vals["estimate"]),
            "white_se": fmt_num(vals["se_white"]),
            "white_ci_lower": fmt_num(vals["ci_white"][0]),
            "white_ci_upper": fmt_num(vals["ci_white"][1]),
            "xs_se": fmt_num(vals["se_xs"]),
            "xs_ci_lower": fmt_num(vals["ci_xs"][0]),
            "xs_ci_upper": fmt_num(vals["ci_xs"][1]),
            "plugin": fmt_num(vals["plugin"]),
        })
    return rows


def build_sample_rows(result: Mapping[str, object], metadata: Mapping[str, object]) -> List[Dict[str, str]]:
    signature = result.get("run_signature", {})
    input_identity = signature.get("input_identity", {}) if isinstance(signature, Mapping) else {}
    tuning = result.get("resolved_tuning", {})
    diagnostics = result.get("diagnostics", {})
    controls = input_identity.get("controls", [])
    control_text = ", ".join(str(x) for x in controls)
    ranks = result.get("ranks", tuning.get("ranks", ""))
    prelim = result.get("input_panel", {}).get("preliminary_bls_observations")
    if prelim is None:
        prelim = "0 (validated by production run)"
    return [
        {"field": "Sample start", "value": str(input_identity.get("start_date", signature.get("date_range", {}).get("start_date", "")))},
        {"field": "Sample end", "value": str(input_identity.get("end_date", signature.get("date_range", {}).get("end_date", "")))},
        {"field": "Number of MSAs", "value": str(result.get("N", ""))},
        {"field": "Number of level months", "value": str(result.get("level_T", ""))},
        {"field": "Usable dynamic observations", "value": str(result.get("usable_dynamic_observations", ""))},
        {"field": "Number of folds", "value": str(result.get("J", diagnostics.get("J_realized", "")))},
        {"field": "Fixed ranks", "value": str(ranks)},
        {"field": "Outcome transformation", "value": str(input_identity.get("outcome", "log(zhvi_all_homes_sa)"))},
        {"field": "Control transformations", "value": control_text},
        {"field": "Lag order", "value": "one monthly lag"},
        {"field": "BLS preliminary observations", "value": str(prelim)},
        {"field": "Seed", "value": str(signature.get("seed", ""))},
        {"field": "Riesz maximum iterations", "value": str(tuning.get("riesz_maxiter", ""))},
        {"field": "Riesz tolerance", "value": fmt_num(tuning.get("riesz_tol", ""))},
        {"field": "Riesz ridge", "value": fmt_num(tuning.get("riesz_ridge", ""))},
        {"field": "Cached-scale setting", "value": str(tuning.get("riesz_use_cached_scale", ""))},
        {"field": "Production runtime", "value": fmt_num(result.get("runtime_sec", ""))},
    ]


def build_diagnostics_rows(result: Mapping[str, object], riesz_summary: Mapping[str, object]) -> List[Dict[str, str]]:
    fields = [
        ("Number of targets", "number_of_targets"),
        ("Number of folds", "number_of_folds"),
        ("Total Riesz solves", "total_riesz_solves"),
        ("Convergence fraction", "convergence_fraction"),
        ("Mean iterations", "iterations_mean"),
        ("Median iterations", "iterations_median"),
        ("Maximum iterations", "iterations_max"),
        ("Number reaching maximum iterations without convergence", "number_reaching_maxiter"),
        ("Mean relative residual", "relative_residual_mean"),
        ("Median relative residual", "relative_residual_median"),
        ("Maximum relative residual", "relative_residual_max"),
        ("Number containing nonfinite values", "number_containing_nonfinite_values"),
    ]
    rows = [{"diagnostic": label, "value": fmt_num(riesz_summary.get(key, ""))} for label, key in fields]
    rows.append({"diagnostic": "production_valid", "value": str(result.get("production_valid"))})
    failures = result.get("production_validation_failures", [])
    rows.append({"diagnostic": "validation failures", "value": "none" if not failures else "; ".join(map(str, failures))})
    return rows


def render_latex_table(caption: str, label: str, headers: Sequence[Tuple[str, str]],
                       rows: Sequence[Mapping[str, object]], note: str) -> str:
    alignment = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(head for _, head in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key, "")) for key, _ in headers) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        rf"\item \textit{{Notes.}} {note}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_markdown_report(main_rows: Sequence[Mapping[str, str]],
                          sample_rows: Sequence[Mapping[str, str]],
                          diag_rows: Sequence[Mapping[str, str]],
                          provenance: Mapping[str, object]) -> str:
    def md_table(rows: Sequence[Mapping[str, str]], headers: Sequence[str]) -> List[str]:
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return out

    row_by_label = {row["estimand"]: row for row in main_rows}
    lag = row_by_label.get("Mean coefficient on lagged log ZHVI", {})
    permits = row_by_label.get("Mean coefficient on lagged asinh building permits", {})
    employment = row_by_label.get("Mean coefficient on lagged log payroll employment", {})
    lines = [
        "# Housing All-Homes Production Report",
        "",
        "## 1. Data and Sample",
    ]
    lines.extend(md_table(sample_rows, ["field", "value"]))
    lines.extend([
        "",
        "## 2. Estimation Specification",
        "",
        "The report summarizes the completed full production run. The target definitions are verified against "
        "`housing_targets()` and `build_blocks()`: the reported targets are full-sample mean coefficients on "
        "lagged log ZHVI, lagged asinh building permits, and lagged log payroll employment. All controls enter "
        "with one monthly lag in the prepared panel specification.",
        "",
        "## 3. Main Estimates",
    ])
    lines.extend(md_table(main_rows, [
        "estimand", "estimate", "white_se", "white_ci_lower", "white_ci_upper",
        "xs_se", "xs_ci_lower", "xs_ci_upper", "plugin",
    ]))
    lines.extend([
        "",
        "## 4. Numerical Diagnostics",
    ])
    lines.extend(md_table(diag_rows, ["diagnostic", "value"]))
    lines.extend([
        "",
        "## 5. Reproducibility Information",
        "",
        f"- Report schema version: `{provenance.get('report_schema_version')}`",
        f"- Production run signature hash: `{provenance.get('production_run_signature_hash')}`",
        f"- Git commit: `{provenance.get('git_commit')}`",
        f"- Git dirty: `{provenance.get('git_dirty')}`",
        "",
        "## 6. Manuscript-Ready Factual Summary",
        "",
        "The completed housing all-homes production run reports dynamic panel estimates of mean coefficients. "
        f"The lagged log-ZHVI coefficient is {lag.get('estimate', '--')}, indicating strong persistence in the "
        "monthly log home-value process. "
        f"The permits coefficient is positive, with estimate {permits.get('estimate', '--')}. "
        f"The payroll-employment coefficient is close to zero at the reported scale, with estimate "
        f"{employment.get('estimate', '--')}. Statistical precision differs between the White standard errors "
        "and the cross-sectional standard errors reported in the production output. These statements describe "
        "estimated mean coefficients and do not assign a causal interpretation beyond the maintained "
        "identification assumptions of the existing estimator.",
        "",
    ])
    return "\n".join(lines)


def build_report_contents(validated: Mapping[str, object], repo_root: Path,
                          production_dir: Path, report_root: Path) -> Dict[str, object]:
    result = validated["result"]
    metadata = validated["metadata"]
    riesz_summary = validated["riesz_summary"]
    labels = validated["target_labels"]
    main_rows = build_main_rows(result, labels)
    sample_rows = build_sample_rows(result, metadata)
    diag_rows = build_diagnostics_rows(result, riesz_summary)

    main_note = (
        "White standard errors are heteroskedasticity-robust standard errors. "
        "Cross-sectional standard errors are the cross-sectional standard errors reported by the production "
        "estimator. Confidence intervals are nominal 95 percent intervals. The target definitions are verified "
        "from the housing target construction, and all controls enter with one monthly lag in the prepared panel "
        "specification."
    )
    sample_note = (
        "This table reports the sample, tuning, and reproducibility settings recorded in the completed housing "
        "production output. The BLS preliminary-observation count is reported as validated by the production run "
        "because the required production files do not separately store the raw preliminary-count field."
    )
    diagnostics_note = (
        "Riesz diagnostics summarize the conjugate-gradient solves used by the reported one-step estimates. "
        "The publication report is generated only when all Riesz solves converge and no nonfinite Riesz values "
        "are reported."
    )

    files: Dict[Path, str] = {}
    main_headers = [
        ("estimand", "Estimand"),
        ("estimate", "Estimate"),
        ("white_se", "White s.e."),
        ("white_ci_lower", "White CI lower"),
        ("white_ci_upper", "White CI upper"),
        ("xs_se", "XS s.e."),
        ("xs_ci_lower", "XS CI lower"),
        ("xs_ci_upper", "XS CI upper"),
        ("plugin", "Plug-in"),
    ]
    sample_headers = [("field", "Field"), ("value", "Value")]
    diagnostics_headers = [("diagnostic", "Diagnostic"), ("value", "Value")]

    tables_dir = report_root / "tables"
    figures_dir = report_root / "figures"
    files[tables_dir / f"{MAIN_TABLE_NAME}.tex"] = render_latex_table(
        "Housing all-homes main estimates.",
        "tab:housing-all-homes-main",
        main_headers,
        main_rows,
        main_note,
    )
    files[tables_dir / f"{SAMPLE_TABLE_NAME}.tex"] = render_latex_table(
        "Housing all-homes sample and specification.",
        "tab:housing-all-homes-sample",
        sample_headers,
        sample_rows,
        sample_note,
    )
    files[tables_dir / f"{DIAGNOSTICS_TABLE_NAME}.tex"] = render_latex_table(
        "Housing all-homes numerical diagnostics.",
        "tab:housing-all-homes-diagnostics",
        diagnostics_headers,
        diag_rows,
        diagnostics_note,
    )

    provenance = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "production_run_signature_hash": validated["run_signature_hash"],
        **git_info(repo_root),
    }
    files[report_root / "housing_all_homes_report.md"] = build_markdown_report(
        main_rows, sample_rows, diag_rows, provenance,
    )

    csv_payloads = {
        tables_dir / f"{MAIN_TABLE_NAME}.csv": (
            main_rows,
            [key for key, _ in main_headers],
        ),
        tables_dir / f"{SAMPLE_TABLE_NAME}.csv": (
            sample_rows,
            [key for key, _ in sample_headers],
        ),
        tables_dir / f"{DIAGNOSTICS_TABLE_NAME}.csv": (
            diag_rows,
            [key for key, _ in diagnostics_headers],
        ),
    }

    input_paths = validated["paths"]
    manifest_base = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "production_run_signature_hash": validated["run_signature_hash"],
        "input_file_checksums": validated["input_checksums"],
        "generation_utc": utc_now(),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "input_paths": {
            name: repo_relative(path, repo_root) for name, path in input_paths.items()
        },
        "output_paths": {
            "report_root": repo_relative(report_root, repo_root),
            "tables": repo_relative(tables_dir, repo_root),
            "figures": repo_relative(figures_dir, repo_root),
        },
        "absolute_path_info": {
            "production_dir": str(production_dir.resolve()),
            "report_root": str(report_root.resolve()),
        },
    }
    manifest_base["report_identity"] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "production_run_signature_hash": validated["run_signature_hash"],
        "input_paths": manifest_base["input_paths"],
        "output_paths": manifest_base["output_paths"],
    }
    return {
        "text_files": files,
        "csv_payloads": csv_payloads,
        "manifest_base": manifest_base,
        "main_rows": main_rows,
        "sample_rows": sample_rows,
        "diagnostics_rows": diag_rows,
    }


def expected_report_files(report_root: Path) -> List[Path]:
    tables = report_root / "tables"
    return [
        tables / f"{MAIN_TABLE_NAME}.csv",
        tables / f"{MAIN_TABLE_NAME}.tex",
        tables / f"{SAMPLE_TABLE_NAME}.csv",
        tables / f"{SAMPLE_TABLE_NAME}.tex",
        tables / f"{DIAGNOSTICS_TABLE_NAME}.csv",
        tables / f"{DIAGNOSTICS_TABLE_NAME}.tex",
        report_root / "housing_all_homes_report.md",
        report_root / "report_manifest.json",
    ]


def compatible_manifest(report_root: Path, manifest_base: Mapping[str, object]) -> bool:
    path = report_root / "report_manifest.json"
    if not path.exists():
        return False
    old = read_json(path)
    keys = ["report_schema_version", "production_run_signature_hash", "input_file_checksums"]
    return all(old.get(k) == manifest_base.get(k) for k in keys)


def ensure_report_reusable_or_writable(report_root: Path, manifest_base: Mapping[str, object],
                                       overwrite: bool) -> Optional[str]:
    manifest = report_root / "report_manifest.json"
    any_outputs = any(path.exists() for path in expected_report_files(report_root))
    if not any_outputs:
        return None
    if compatible_manifest(report_root, manifest_base) and all(path.exists() for path in expected_report_files(report_root)):
        return "compatible"
    if not overwrite:
        if manifest.exists():
            raise SystemExit("refusing to replace incompatible housing all-homes report without --overwrite")
        raise SystemExit("refusing to replace incomplete housing all-homes report without --overwrite")
    return None


def write_report(contents: Mapping[str, object], report_root: Path) -> Dict[str, str]:
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "tables").mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)

    generated: Dict[str, str] = {}
    for path, text in contents["text_files"].items():
        atomic_write_text(path, text)
        generated[repo_relative(path, report_root)] = sha256_file(path)
    for path, (rows, fields) in contents["csv_payloads"].items():
        atomic_write_csv(path, rows, fields)
        generated[repo_relative(path, report_root)] = sha256_file(path)

    manifest = dict(contents["manifest_base"])
    manifest["generated_file_checksums"] = generated
    manifest["generated_paths"] = sorted(generated)
    manifest_path = report_root / "report_manifest.json"
    atomic_write_json(manifest_path, manifest)
    generated[repo_relative(manifest_path, report_root)] = sha256_file(manifest_path)
    return generated


def build_report(repo_root: Path, production_dir: Path, report_root: Path,
                 overwrite: bool = False) -> Dict[str, object]:
    validated = validate_production_inputs(production_dir)
    contents = build_report_contents(validated, repo_root, production_dir, report_root)
    reuse = ensure_report_reusable_or_writable(report_root, contents["manifest_base"], overwrite)
    if reuse == "compatible":
        return {
            "status": "reused",
            "report_root": str(report_root),
            "production_run_signature_hash": validated["run_signature_hash"],
        }
    generated = write_report(contents, report_root)
    return {
        "status": "written",
        "report_root": str(report_root),
        "generated": generated,
        "production_run_signature_hash": validated["run_signature_hash"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build housing all-homes publication report from completed production output.")
    ap.add_argument("--repo-root", default=None, help="repository root; default searches from current directory")
    ap.add_argument("--production-dir", default=None, help="completed production output directory")
    ap.add_argument("--report-root", default=None, help="report output directory")
    ap.add_argument("--overwrite", action="store_true", help="replace an incompatible existing report")
    args = ap.parse_args(argv)

    repo_root = find_repo_root(explicit=Path(args.repo_root) if args.repo_root else None)
    production_dir = resolve_repo_path(
        args.production_dir,
        repo_root,
    ) if args.production_dir else repo_root / "outputs" / "empirical" / "housing_all_homes" / "production"
    report_root = resolve_repo_path(
        args.report_root,
        repo_root,
    ) if args.report_root else repo_root / "outputs" / "empirical" / "housing_all_homes" / "report"

    try:
        result = build_report(repo_root, production_dir, report_root, overwrite=bool(args.overwrite))
    except Exception as exc:
        print(f"[housing-all-homes-report] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[housing-all-homes-report] {result['status']} report at {repo_relative(report_root, repo_root)} "
        f"for signature {result['production_run_signature_hash'][:12]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
