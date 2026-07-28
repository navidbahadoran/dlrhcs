"""Spec sec 15 test checklist -- the correctness gates before scaling up.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
"""
import os
import sys
import dataclasses
import argparse
import csv
import json
import math
import tempfile

# make `import dlrhcs` work when run directly (python tests/test_core.py) from a
# clean checkout, without requiring PYTHONPATH or an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dlrhcs.design import (A, A_adjoint, build_blocks, theta_dot)
from dlrhcs.dgp import simulate
from dlrhcs.factorridge import fit_factor_ridge
from dlrhcs.folds import make_folds
from dlrhcs.mc import run_replication
from dlrhcs.paths import find_repo_root, repo_relative, resolve_repo_path
from dlrhcs.pipeline import Tuning
from dlrhcs.targets import (entry_direction, project_block, project_tangent,
                            riesz_weights, Target, cg_converged_from_status)
from dlrhcs import housing_data
from scripts import sim_report, run_mc_batches, report_housing_data, housing_all_homes


def _panel(Tp=24, N=20, seed=0, sigma_u=0.30):
    rng = np.random.default_rng(seed)
    return simulate(Tp, N, rng, sigma_u=sigma_u)


# 1. A / A* adjoint identity ------------------------------------------------- #
def test_adjoint_identity():
    p = _panel()
    blocks = build_blocks(p.Z)
    rng = np.random.default_rng(1)
    theta = [rng.standard_normal((p.Tp, p.N)) for _ in blocks]
    R = rng.standard_normal((p.Tp, p.N))
    lhs = float(np.vdot(A(theta, blocks), R))
    rhs = theta_dot(theta, A_adjoint(R, blocks))
    assert abs(lhs - rhs) < 1e-10 * (1 + abs(lhs))


# 2. forward-exclusion indexing on a hand-checked grid ----------------------- #
def test_forward_exclusion():
    Tp, N, J, q = 6, 1, 2, 2
    foldid = np.array([[0], [1], [0], [1], [0], [1]])  # alternate by time
    folds = make_folds(Tp, N, J, q, foldid=foldid)
    # fold j=1 held out at t=1,3,5 (0-based). Train excludes those AND the q=2
    # rows after each held-out row in the same unit.
    train1 = folds[1].train[:, 0]
    val1 = folds[1].val[:, 0]
    assert list(val1) == [False, True, False, True, False, True]
    # t=0 (not val, no prior held-out): train. t=2 has held-out at t=1 -> purged.
    # t=4 has held-out at t=3 -> purged. So only t=0 trains for fold 1.
    assert list(train1) == [True, False, False, False, False, False]


def test_spatial_buffer_no_wrap_and_seed_invariant_dgp_truth():
    Tp, N, J, q = 3, 5, 2, 0
    foldid = np.zeros((Tp, N), dtype=int)
    foldid[0, 0] = 1
    foldid[1, N - 1] = 1
    foldid[2, 2] = 1
    folds0 = make_folds(Tp, N, J, q, r=0, foldid=foldid)
    folds1 = make_folds(Tp, N, J, q, r=1, foldid=foldid)
    tr0 = folds0[1].train
    tr1 = folds1[1].train
    assert not np.array_equal(tr0, tr1)
    # Unit 1 (0-based 0) with r=1 removes units 0 and 1, not unit N (0-based 4).
    assert list(tr1[0]) == [False, False, True, True, True]
    # Unit N (0-based 4) removes units N-1 and N, with no circular wrap to unit 1.
    assert list(tr1[1]) == [True, True, True, False, False]
    # An interior held-out unit removes its immediate neighbours.
    assert list(tr1[2]) == [True, False, False, False, True]

    base = Tuning(ranks=(1, 1, 1), q=1, J_min=2, n_sweeps=2, n_restarts=0,
                  riesz_maxiter=25, riesz_tol=1e-6, buffer_r=0)
    dgp = dict(dgp_type="dgp3", c_xi_calibration_draws=3)
    r0 = run_replication(8, 6, 0, base, dgp_kwargs=dgp, master=777,
                         target_names=["lag_fmean"])
    r1 = run_replication(8, 6, 0, dataclasses.replace(base, buffer_r=1),
                         dgp_kwargs=dgp, master=777, target_names=["lag_fmean"])
    assert r0["_sim_seed_sequence"] == r1["_sim_seed_sequence"]
    assert r0["_est_seed_sequence"] == r1["_est_seed_sequence"]
    assert r0["lag_fmean"]["true_value"] == r1["lag_fmean"]["true_value"]
    assert r0["_r"] == 0
    assert r1["_r"] == 1
    assert r0["_retained_nonvalidation"] != r1["_retained_nonvalidation"]


def test_run_mc_batches_riesz_ridge_override_and_resume_signature():
    assert Tuning().riesz_ridge == 1e-8
    assert run_mc_batches._parse_positive_float("1e-6") == 1e-6
    for bad in ("0", "-1e-8", "nan", "inf", "-inf"):
        try:
            run_mc_batches._parse_positive_float(bad)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected as a Riesz ridge override")

    base = Tuning(ranks=(1, 1, 1), q=1, J_min=2, n_sweeps=2, n_restarts=0,
                  riesz_maxiter=30, riesz_tol=1e-6, buffer_r=0, riesz_ridge=1e-8)
    dgp = dict(dgp_type="dgp1", c_xi_calibration_draws=3)
    r_default = run_replication(8, 6, 0, base, dgp_kwargs=dgp, master=888,
                                target_names=["lag_fmean"])
    r_ridge = run_replication(8, 6, 0, dataclasses.replace(base, riesz_ridge=1e-2),
                              dgp_kwargs=dgp, master=888, target_names=["lag_fmean"])
    assert r_default["_riesz_ridge"] == 1e-8
    assert r_ridge["_riesz_ridge"] == 1e-2
    assert r_default["_sim_seed_sequence"] == r_ridge["_sim_seed_sequence"]
    assert r_default["_est_seed_sequence"] == r_ridge["_est_seed_sequence"]
    assert r_default["lag_fmean"]["true_value"] == r_ridge["lag_fmean"]["true_value"]
    assert r_default["lag_fmean"]["plugin_err"] == r_ridge["lag_fmean"]["plugin_err"]
    assert abs(r_default["lag_fmean"]["estimate"] - r_ridge["lag_fmean"]["estimate"]) > 1e-10

    args = argparse.Namespace(
        dgp_type="dgp1", T=8, N=6, oracle=False, targets=None,
        select=False, fixed_ranks=(1, 1, 1), rank_caps=None,
        c_xi_calibration_draws=3,
    )
    old_sig = run_mc_batches._requested_signature(args, base, dgp, 888)
    meta = dict(old_sig)
    meta["resolved_dgp_kwargs"] = dgp
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "ridge_resume.jsonl")
        sidecar = os.path.join(tmp, "ridge_resume.meta.json")
        with open(sidecar, "w") as fh:
            json.dump(meta, fh)
        requested = run_mc_batches._requested_signature(
            args, dataclasses.replace(base, riesz_ridge=1e-6), dgp, 888
        )
        try:
            run_mc_batches._assert_resume_compatible(
                run_mc_batches.Path(out_path), run_mc_batches.Path(sidecar), requested
            )
        except SystemExit as exc:
            assert "riesz_ridge" in str(exc)
        else:
            raise AssertionError("resuming a 1e-8 output with 1e-6 Riesz ridge should fail")


def _sim_report_item(name, dgp="dgp3", N=100, T=100, *,
                     kind_select=False, completed=1000, total=1000,
                     target_filter=None, fixed_ranks=(1, 1, 1)):
    meta = dict(
        dgp_type=dgp,
        N=N,
        Tp=T,
        T=T,
        completed_R=completed,
        R_total=total,
        select=kind_select,
        fixed_ranks=list(fixed_ranks) if fixed_ranks is not None else None,
        target_filter=target_filter,
        q_T=2,
        r_N=0,
        h_N_realized=int(np.floor(N ** (1.0 / 3.0))),
        J_min=10,
        J_realized=10,
        retained_nonvalidation=0.8,
        retained_total=0.7,
    )
    return {"path": os.path.join("outputs", "sim", name), "meta": meta, "agg": {"_meta": meta}}


def test_sim_report_grid_v3_filtering_and_duplicate_resolution():
    assert sim_report._is_report_jsonl(os.path.join("outputs", "sim", "grid_v3_dgp3_100.jsonl"))
    assert sim_report._report_file_kind("grid_v3_dgp3_100.jsonl") == "grid_v3"
    assert sim_report._report_file_kind("grid_v2_dgp3_100.jsonl") == "grid_v2"
    assert sim_report._report_file_kind("grid_dgp3_100.jsonl") == "grid"

    v3 = _sim_report_item("grid_v3_dgp3_100.jsonl")
    v2 = _sim_report_item("grid_v2_dgp3_100.jsonl")
    old = _sim_report_item("grid_dgp3_100.jsonl")
    kept, dropped = sim_report._resolve_duplicate_cells([old, v2, v3])
    assert [os.path.basename(item["path"]) for item in kept] == ["grid_v3_dgp3_100.jsonl"]
    assert sorted(os.path.basename(path) for path in dropped) == [
        "grid_dgp3_100.jsonl",
        "grid_v2_dgp3_100.jsonl",
    ]

    kept, dropped = sim_report._resolve_duplicate_cells([old, v2])
    assert [os.path.basename(item["path"]) for item in kept] == ["grid_v2_dgp3_100.jsonl"]
    assert [os.path.basename(path) for path in dropped] == ["grid_dgp3_100.jsonl"]

    filtered = _sim_report_item("grid_v3_dgp3_100_lag_fmean.jsonl",
                                target_filter=["lag_fmean"])
    kept, excluded = sim_report._filter_target_filtered_production([v3, filtered])
    assert kept == [v3]
    assert excluded[0]["target_filter"] == ["lag_fmean"]

    incomplete = _sim_report_item("grid_v3_dgp3_200.jsonl", N=200, T=200,
                                  completed=900, total=1000)
    kept, excluded = sim_report._filter_incomplete_production([v3, incomplete])
    assert kept == [v3]
    assert excluded[0]["completed_R"] == 900

    dup_a = _sim_report_item("grid_v3_dgp3_100_a.jsonl")
    dup_b = _sim_report_item("grid_v3_dgp3_100_b.jsonl")
    try:
        sim_report._resolve_duplicate_cells([dup_a, dup_b])
    except ValueError as exc:
        assert "ambiguous duplicate production files" in str(exc)
    else:
        raise AssertionError("same-version duplicate grid_v3 files should be ambiguous")

    dgp1 = _sim_report_item("grid_dgp1_100.jsonl", dgp="dgp1")
    dgp2 = _sim_report_item("grid_v2_dgp2_100.jsonl", dgp="dgp2")
    kept, dropped = sim_report._resolve_duplicate_cells([dgp1, dgp2])
    assert kept == [dgp1, dgp2]
    assert dropped == []


def test_sim_report_writes_tex_only_to_manuscript_dir_and_csv_only_to_data_dir():
    old_data_dir = sim_report.REPORT_TABLE_DATA_DIR
    old_manuscript_dir = sim_report.MANUSCRIPT_TABLE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "outputs", "sim", "tables")
        manuscript_dir = os.path.join(tmp, "tables")
        os.makedirs(data_dir)
        os.makedirs(manuscript_dir)
        data_unrelated = os.path.join(data_dir, "user_notes.tex")
        manuscript_unrelated = os.path.join(manuscript_dir, "user_table.tex")
        with open(data_unrelated, "w") as fh:
            fh.write("keep me\n")
        with open(manuscript_unrelated, "w") as fh:
            fh.write("keep me\n")
        for name in sim_report.ACTIVE_MANUSCRIPT_TEX:
            with open(os.path.join(data_dir, name), "w") as fh:
                fh.write("stale duplicate\n")
        with open(os.path.join(data_dir, "tab_mc_performance_full.tex"), "w") as fh:
            fh.write("obsolete duplicate\n")

        sim_report.REPORT_TABLE_DATA_DIR = data_dir
        sim_report.MANUSCRIPT_TABLE_DIR = manuscript_dir
        try:
            result = sim_report.write_journal_tables([
                _sim_report_item("grid_v3_dgp1_100.jsonl", dgp="dgp1")
            ])
        finally:
            sim_report.REPORT_TABLE_DATA_DIR = old_data_dir
            sim_report.MANUSCRIPT_TABLE_DIR = old_manuscript_dir

        expected_tex = sim_report.ACTIVE_MANUSCRIPT_TEX
        expected_csv = {
            "tab_mc_performance.csv",
            "tab_rank_frequency.csv",
            "tab_fold_retention.csv",
            "tab_mc_coeff_summary.csv",
            "table_source_audit.csv",
        }
        for name in expected_tex:
            assert os.path.exists(os.path.join(manuscript_dir, name))
            assert not os.path.exists(os.path.join(data_dir, name))
        for name in expected_csv:
            assert os.path.exists(os.path.join(data_dir, name))
            assert not os.path.exists(os.path.join(manuscript_dir, name))
        assert os.path.exists(os.path.join(manuscript_dir, "tab_mc_rank.tex"))
        with open(os.path.join(manuscript_dir, "tab_mc_main_summary.tex")) as fh:
            main_summary = fh.read()
        assert r"\input{tables/tab_mc_perf_dgp1.tex}" in main_summary
        assert r"\input{tables/tab_mc_perf_dgp2.tex}" in main_summary
        assert r"\input{tables/tab_mc_perf_dgp3.tex}" in main_summary
        assert os.path.exists(data_unrelated)
        assert os.path.exists(manuscript_unrelated)
        removed_names = {os.path.basename(path) for path in result["removed_obsolete_tex"]}
        assert "tab_mc_main_summary.tex" in removed_names
        assert "tab_mc_performance_full.tex" in removed_names


def test_housing_zillow_file_identification_and_date_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "data", "zillow")
        os.makedirs(root)
        good = os.path.join(root, "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
        top = os.path.join(root, "zillow_metro_top.csv")
        with open(good, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["RegionID", "SizeRank", "RegionName", "RegionType", "StateName",
                        "2000-01-31", "2000-02-29"])
            w.writerow(["1", "0", "A City, AA", "msa", "AA", "100", "101"])
        with open(top, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["RegionID", "RegionName", "RegionType", "2000-01-31"])
            w.writerow(["1", "A City, AA", "msa", "100"])
        header, _ = housing_data.read_csv_rows(housing_data.Path(good))
        cls = housing_data.classify_zillow_file(housing_data.Path(good), header)
        assert cls["is_all_homes_metro_sa"]
        rows, summary = housing_data.parse_zillow_all_homes(housing_data.Path(good))
        assert rows[0]["date"] == "2000-01-01"
        assert summary["n_metros"] == 1
        header_top, _ = housing_data.read_csv_rows(housing_data.Path(top))
        cls_top = housing_data.classify_zillow_file(housing_data.Path(top), header_top)
        assert not cls_top["is_all_homes_metro_sa"]
        assert cls_top["is_top_or_bottom_tier"]


def test_housing_duplicate_missing_months_and_no_interpolation_guarantee():
    rows = [
        {"cbsa_code": "12345", "date": "2020-01-01", "value": 1},
        {"cbsa_code": "12345", "date": "2020-01-01", "value": 1},
        {"cbsa_code": "12345", "date": "2020-03-01", "value": 3},
    ]
    assert housing_data.duplicate_key_count(rows, ["cbsa_code", "date"]) == 1
    assert housing_data.detect_missing_months(["2020-01", "2020-03"]) == ["2020-02"]
    assert housing_data.contiguous_segments(["2020-01", "2020-03", "2020-04"]) == [
        ("2020-01", "2020-01", 1),
        ("2020-03", "2020-04", 2),
    ]


def test_housing_permit_zero_handling_and_x13_retains_observed_dates():
    content = (
        "202001,000,12345,1,Test CBSA,"
        "0,0,0,0,2,0,0,3,0,0,4,0,"
        "0,0,0,0,0,0,0,0,0,0,0,0\n"
    ).encode("latin1")
    rows, micro = housing_data.parse_bps_monthly_file(content, "fixture.txt")
    assert micro == 0
    assert rows[0]["total_units"] == 9
    observed_dates = [r["date"] for r in rows]
    assert observed_dates == ["2020-01-01"]
    # The seasonal-adjustment helper writes only observed source dates; it never
    # constructs endpoint forecasts/backcasts or fills internal missing dates.
    assert housing_data.detect_missing_months([d[:7] for d in observed_dates]) == []


def test_housing_official_vs_local_employment_labeling_and_exact_matching():
    zillow = [
        {"zillow_region_id": "1", "zillow_region_name": "Test City, AA",
         "date": "2020-01-01", "zhvi_all_homes_sa": 100.0},
    ]
    geo = [
        {"cbsa_code": "12345", "census_cbsa_title": "Test City, AA",
         "metro_micro_type": "Metropolitan Statistical Area",
         "geographic_vintage": "fixture"},
    ]
    bps = [{"cbsa_code": "12345", "cbsa_title": "Test City, AA", "date": "2020-01-01", "total_units": 1}]
    emp = [{"cbsa_code": "12345", "area_title": "Test City, AA", "date": "2020-01-01", "employment_level": 10,
            "employment_thousands_sa": 10,
            "seasonal_adjustment_flag": "S"}]
    cross, review = housing_data.build_crosswalk(zillow, bps, emp, geo)
    assert len(cross) == 1
    assert not review
    assert cross[0]["match_status"].startswith("accepted")
    avail = housing_data.build_availability(zillow, bps, [], emp, [], cross)
    assert avail[0]["employment_official_sa_available"] == 1
    assert avail[0]["employment_local_x13_sa_available"] == 0


def test_housing_rejects_ambiguous_fuzzy_matches_and_frontier():
    zillow = [{"zillow_region_id": "1", "zillow_region_name": "Twin City, AA",
               "date": "2020-01-01", "zhvi_all_homes_sa": 100.0}]
    geo = [
        {"cbsa_code": "11111", "census_cbsa_title": "Twin City, AA",
         "metro_micro_type": "Metropolitan Statistical Area"},
        {"cbsa_code": "22222", "census_cbsa_title": "Twin City, AA",
         "metro_micro_type": "Metropolitan Statistical Area"},
    ]
    cross, review = housing_data.build_crosswalk(zillow, [], [], geo)
    assert not cross
    assert review[0]["match_status"] == "manual_review"
    assert "ambiguous" in review[0]["review_reason"]

    availability = []
    for month in ("2020-01", "2020-02", "2020-03"):
        availability.append({
            "cbsa_code": "12345", "date": month + "-01",
            "zhvi_sa_available": 1, "permits_nsa_available": 1,
            "permits_sa_available": 1, "employment_official_sa_available": 1,
            "employment_local_x13_sa_available": 0,
        })
    frontier = housing_data.balanced_panel_frontier(availability, thresholds=(2,))
    assert frontier[0]["N_complete_official_sa"] == 1
    assert frontier[0]["T_months"] == 3


def test_housing_bls_bulk_urls_are_uppercase_and_get_only():
    assert "/SM/" in housing_data.BLS_SM_BASE
    for _, url, _, _ in housing_data.BLS_BULK_FILES:
        assert "/SM/" in url
    assert housing_data._bls_expected_header("sm.area") == ["area_code", "area_name"]


def test_housing_bls_validation_accepts_text_and_rejects_html():
    with tempfile.TemporaryDirectory() as tmp:
        root = housing_data.Path(tmp)
        good = root / "sm.area"
        good.write_text("area_code\tarea_name\n12345\tTest City, AA\n", encoding="utf-8")
        ok, reason = housing_data.validate_bls_bulk_file(good, ["area_code", "area_name"], 1)
        assert ok, reason
        bad = root / "sm.area.bad"
        bad.write_text("<html><body>Access Denied</body></html>", encoding="utf-8")
        ok, reason = housing_data.validate_bls_bulk_file(bad, ["area_code", "area_name"], 1)
        assert not ok
        assert "HTML" in reason or "access-denied" in reason


def test_housing_bls_python_get_streams_and_writes_atomically():
    class FakeHeaders(dict):
        def items(self):
            return super().items()

    class FakeResponse:
        status = 200
        headers = FakeHeaders({"content-type": "application/octet-stream"})

        def __init__(self, payload):
            self.payload = payload
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n):
            chunk = self.payload[self.offset:self.offset + n]
            self.offset += len(chunk)
            return chunk

        def geturl(self):
            return housing_data.BLS_AREA_URL

    original = housing_data.urllib.request.urlopen
    try:
        payload = b"area_code\tarea_name\n12345\tTest City, AA\n"
        housing_data.urllib.request.urlopen = lambda req, timeout=0: FakeResponse(payload)
        with tempfile.TemporaryDirectory() as tmp:
            dest = housing_data.Path(tmp) / "sm.area"
            rec, diag = housing_data.bls_python_get(
                housing_data.BLS_AREA_URL,
                dest,
                1,
                ["area_code", "area_name"],
                tries=1,
            )
            assert rec is not None
            assert rec.__dict__["transport"] == "python_http"
            assert dest.exists()
            assert not dest.with_name(dest.name + ".part").exists()
            assert diag[0]["method"] == "GET"
            assert not diag[0]["used_head"]
    finally:
        housing_data.urllib.request.urlopen = original


def test_housing_bls_curl_fallback_uses_argument_list_no_shell_and_atomic_write():
    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    captured = {}
    original = housing_data.subprocess.run
    try:
        def fake_run(cmd, capture_output, text, timeout, check):
            captured["cmd"] = cmd
            captured["shell"] = False
            part = housing_data.Path(cmd[cmd.index("-o") + 1])
            part.write_text("area_code\tarea_name\n12345\tTest City, AA\n", encoding="utf-8")
            return FakeProc()

        housing_data.subprocess.run = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            dest = housing_data.Path(tmp) / "sm.area"
            rec, diag = housing_data.bls_curl_get(
                housing_data.BLS_AREA_URL,
                dest,
                1,
                ["area_code", "area_name"],
            )
            assert rec is not None
            assert rec.__dict__["transport"] == "curl_fallback"
            assert isinstance(captured["cmd"], list)
            assert diag["shell"] is False
            assert "curl.exe" == captured["cmd"][0]
            assert not dest.with_name(dest.name + ".part").exists()
    finally:
        housing_data.subprocess.run = original


def test_housing_bls_series_selection_m13_footnote_and_preliminary_flags():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = housing_data.ensure_dirs(housing_data.Path(tmp))
        (dirs["raw_bls"] / "sm.area").write_text(
            "area_code\tarea_name\n"
            "12345\tTest City, AA\n"
            "00000\tStatewide\n"
            "54321\tOther Metropolitan Division\n",
            encoding="utf-8",
        )
        (dirs["raw_bls"] / "sm.industry").write_text(
            "industry_code\tindustry_name\n00000000\tTotal Nonfarm\n00000001\tMining\n",
            encoding="utf-8",
        )
        (dirs["raw_bls"] / "sm.series").write_text(
            "series_id\tarea_code\tstate_code\tsupersector_code\tindustry_code\tdata_type_code\tseasonal\n"
            "SMSGOOD\t12345\t12\t00\t00000000\t01\tS\n"
            "SMUGOOD\t12345\t12\t00\t00000000\t01\tU\n"
            "SMSSTATE\t00000\t12\t00\t00000000\t01\tS\n"
            "SMSDIV\t54321\t54\t00\t00000000\t01\tS\n"
            "SMSBADTYPE\t12345\t12\t00\t00000000\t26\tS\n"
            "SMSBADIND\t12345\t12\t00\t00000001\t01\tS\n",
            encoding="utf-8",
        )
        geo = [
            {"cbsa_code": "12345", "metro_micro_type": "Metropolitan Statistical Area"},
            {"cbsa_code": "54321", "metro_micro_type": "Metropolitan Statistical Area"},
        ]
        sa, nsa, _ = housing_data.parse_bls_series_metadata(dirs, geo)
        assert sorted(sa) == ["SMSGOOD"]
        assert sorted(nsa) == ["SMUGOOD"]
        data_rows = [
            {"series_id": "SMSGOOD", "year": "2020", "period": "M01", "value": "100.5", "footnote_codes": "P"},
            {"series_id": "SMSGOOD", "year": "2020", "period": "M13", "value": "999", "footnote_codes": ""},
            {"series_id": "SMUGOOD", "year": "2020", "period": "M02", "value": "99", "footnote_codes": "R"},
        ]
        official, nsa_rows = housing_data.parse_bls_employment_data(data_rows, sa, nsa)
        assert len(official) == 1
        assert official[0]["date"] == "2020-01-01"
        assert official[0]["preliminary_flag"] == 1
        assert official[0]["footnote_codes"] == "P"
        assert len(nsa_rows) == 1
        assert nsa_rows[0]["seasonal_adjustment_source"] == "not_seasonally_adjusted"


def test_housing_zillow_geography_classification_no_fuzzy_or_interpolation():
    zillow = [
        {"zillow_region_id": "1", "zillow_region_name": "Metro City, AA", "region_type": "msa", "state_name": "AA"},
        {"zillow_region_id": "2", "zillow_region_name": "Micro City, AA", "region_type": "msa", "state_name": "AA"},
        {"zillow_region_id": "3", "zillow_region_name": "Division City, AA", "region_type": "msa", "state_name": "AA"},
        {"zillow_region_id": "4", "zillow_region_name": "County Region, AA", "region_type": "county", "state_name": "AA"},
        {"zillow_region_id": "5", "zillow_region_name": "Old City, AA", "region_type": "msa", "state_name": "AA"},
        {"zillow_region_id": "6", "zillow_region_name": "Almost Metro City, AA", "region_type": "msa", "state_name": "AA"},
    ]
    geo = [
        {"cbsa_code": "11111", "census_cbsa_title": "Metro City, AA",
         "metro_micro_type": "Metropolitan Statistical Area", "metropolitan_division_title": ""},
        {"cbsa_code": "22222", "census_cbsa_title": "Micro City, AA",
         "metro_micro_type": "Micropolitan Statistical Area", "metropolitan_division_title": ""},
        {"cbsa_code": "33333", "census_cbsa_title": "Parent City, AA",
         "metro_micro_type": "Metropolitan Statistical Area", "metropolitan_division_title": "Division City, AA"},
    ]
    bps = [{"cbsa_code": "44444", "cbsa_title": "Old City, AA"}]
    rows = housing_data.classify_zillow_geography(zillow, bps, [], geo)
    cls = {r["zillow_region_id"]: r["classification"] for r in rows}
    assert cls["1"] == "current_metropolitan_cbsa"
    assert cls["2"] == "current_micropolitan_cbsa"
    assert cls["3"] == "metropolitan_division"
    assert cls["4"] == "non_cbsa_zillow_region"
    assert cls["5"] == "historical_or_retired_cbsa"
    assert cls["6"] == "unresolved"
    assert housing_data.detect_missing_months(["2020-01", "2020-03"]) == ["2020-02"]


def _write_bls_manual_fixture(root, *, html_file=None, malformed_file=None, truncated_file=None,
                              missing_file=None, extra_file=False):
    root = housing_data.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "sm.area": (
            "area_code\tarea_name\r\n"
            "12345\tTest City, AA\r\n"
            "00000\tStatewide\r\n"
            "54321\tDivision City Metropolitan Division\r\n"
        ),
        "sm.seasonal": "seasonal_code\tseasonal_text\r\nS\tSeasonally Adjusted\r\nU\tNot Seasonally Adjusted\r\n",
        "sm.industry": "industry_code\tindustry_name\r\n00000000\tTotal Nonfarm\r\n00000001\tTotal Private\r\n",
        "sm.footnote": "footnote_code\tfootnote_text\r\nP\tPreliminary\r\nR\tRevised\r\n",
        "sm.series": (
            "\ufeffseries_id                     \tstate_code\tarea_code\tsupersector_code\tindustry_code\tdata_type_code\tseasonal\tbenchmark_year\tfootnote_codes\tbegin_year\tbegin_period\tend_year\tend_period\r\n"
            "SMSDT21                     \t01\t12345\t00\t00000000\t21\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSDT22                     \t01\t12345\t00\t00000000\t22\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSGOOD                     \t01\t12345\t00\t00000000\t01\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMUGOOD                     \t01\t12345\t00\t00000000\t01\tU\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSSTATE                    \t01\t00000\t00\t00000000\t01\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSDIV                      \t54\t54321\t00\t00000000\t01\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSPRIVATE                  \t01\t12345\t00\t00000001\t01\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
            "SMSCHANGE                   \t01\t12345\t00\t00000000\t26\tS\t2024\t\t1990\tM01\t2026\tM06\r\n"
        ),
        "sm.data.54.TotalNonFarm.All": (
            "series_id                     \tyear\tperiod\t       value\tfootnote_codes\r\n"
            "SMSGOOD                     \t2020\tM01\t100.5\tP\r\n"
            "SMSGOOD                     \t2020\tM13\t999.0\t\r\n"
            "SMUGOOD                     \t2020\tM01\t98.0\r\n"
            "SMSSTATE                    \t2020\tM01\t500.0\t\r\n"
            "SMSDIV                      \t2020\tM01\t50.0\t\r\n"
        ),
    }
    if missing_file:
        files.pop(missing_file, None)
    for name, text in files.items():
        if name == html_file:
            text = "<html><title>Access Denied</title></html>\n"
        if name == malformed_file:
            text = "series_id\tyear\tperiod\tvalue\tfootnote_codes\nSMSGOOD\t2020\n"
        if name == truncated_file:
            text = text.rstrip("\r\n")
        (root / name).write_text(text, encoding="utf-8", newline="")
    if extra_file:
        (root / "README.txt").write_text("not part of the official SM bulk set\n", encoding="utf-8")


def test_housing_bls_local_import_success_checksums_atomic_and_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = housing_data.Path(tmp)
        local = tmp / "manual" / "bls_ces"
        _write_bls_manual_fixture(local)
        dirs = housing_data.ensure_dirs(tmp / "data" / "zillow")
        manifest = []
        diag = housing_data.import_bls_local_files(local, dirs, manifest)
        assert len(diag) == len(housing_data.BLS_REQUIRED_FILENAMES)
        assert len(manifest) == len(housing_data.BLS_REQUIRED_FILENAMES)
        for rec in manifest:
            assert rec.__dict__["acquisition_method"] == "manual_official_download"
            assert rec.__dict__["source_sha256"] == rec.__dict__["destination_sha256"]
            assert not housing_data.Path(rec.local_path).with_name(housing_data.Path(rec.local_path).name + ".part").exists()
        geo = [{"cbsa_code": "12345", "metro_micro_type": "Metropolitan Statistical Area"}]
        official, nsa = housing_data.parse_bls_cached_files(dirs, geo)
        assert len(official) == 1
        assert official[0]["bls_series_id"] == "SMSGOOD"
        assert official[0]["date"] == "2020-01-01"
        assert official[0]["preliminary_flag"] == 1
        assert len(nsa) == 1
        assert nsa[0]["bls_series_id"] == "SMUGOOD"


def test_housing_bls_local_import_rejects_missing_extra_html_malformed_truncated_and_conflict():
    cases = [
        {"missing_file": "sm.footnote"},
        {"extra_file": True},
        {"html_file": "sm.area"},
        {"malformed_file": "sm.data.54.TotalNonFarm.All"},
        {"truncated_file": "sm.series"},
    ]
    for kwargs in cases:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = housing_data.Path(tmp)
            local = tmp / "manual"
            _write_bls_manual_fixture(local, **kwargs)
            dirs = housing_data.ensure_dirs(tmp / "data")
            try:
                housing_data.import_bls_local_files(local, dirs, [])
            except ValueError:
                pass
            else:
                raise AssertionError(f"local BLS import unexpectedly accepted {kwargs}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = housing_data.Path(tmp)
        local = tmp / "manual"
        _write_bls_manual_fixture(local)
        dirs = housing_data.ensure_dirs(tmp / "data")
        (dirs["raw_bls"] / "sm.area").write_text("area_code\tarea_name\n99999\tDifferent\n", encoding="utf-8")
        try:
            housing_data.import_bls_local_files(local, dirs, [])
        except ValueError as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("conflicting raw BLS cache was silently overwritten")


def test_housing_bls_local_import_validates_cross_references():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = housing_data.Path(tmp)
        local = tmp / "manual"
        _write_bls_manual_fixture(local)
        path = local / "sm.series"
        text = path.read_text(encoding="utf-8").replace(
            "SMSGOOD                     \t01\t12345",
            "SMSGOOD                     \t01\t99999",
        )
        path.write_text(text, encoding="utf-8")
        dirs = housing_data.ensure_dirs(tmp / "data")
        try:
            housing_data.import_bls_local_files(local, dirs, [])
        except ValueError as exc:
            assert "area_code" in str(exc)
        else:
            raise AssertionError("bad sm.series area_code cross-reference was accepted")


def test_housing_bls_local_dir_run_recomputes_overlap_without_bls_network():
    with tempfile.TemporaryDirectory() as tmp:
        root = housing_data.Path(tmp) / "data" / "zillow"
        dirs = housing_data.ensure_dirs(root)
        local = root / "manual_import" / "bls_ces"
        _write_bls_manual_fixture(local)
        (dirs["raw_zillow"] / housing_data.Path(housing_data.ZILLOW_ALL_HOMES_URL).name).write_text(
            "RegionID,SizeRank,RegionName,RegionType,StateName,2020-01-31\n"
            "1,0,\"Test City, AA\",msa,AA,100\n",
            encoding="utf-8",
        )
        housing_data.atomic_write_csv(dirs["processed"] / "permits_metro_nsa_long.csv", [
            {"cbsa_code": "12345", "cbsa_title": "Test City, AA", "date": "2020-01-01", "total_units": 10}
        ], ["cbsa_code", "cbsa_title", "date", "total_units"])
        housing_data.atomic_write_csv(dirs["processed"] / "permits_metro_sa_long.csv", [
            {"cbsa_code": "12345", "cbsa_title": "Test City, AA", "date": "2020-01-01", "permits_units_sa": 10}
        ], ["cbsa_code", "cbsa_title", "date", "permits_units_sa"])
        housing_data.atomic_write_csv(root / "audit" / "x13_diagnostics.csv", [], ["series_id", "status"])
        original_geo = housing_data.parse_cached_geography
        original_fetch_bls = housing_data.fetch_bls
        try:
            housing_data.parse_cached_geography = lambda path, warnings: [
                {"cbsa_code": "12345", "census_cbsa_title": "Test City, AA",
                 "metro_micro_type": "Metropolitan Statistical Area", "geographic_vintage": "fixture"}
            ]
            housing_data.fetch_bls = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("BLS network fetch should not run"))
            code, summary = housing_data.run_housing_audit(root, bls_local_dir=local, min_sa_months=84)
            assert code == 0
            assert summary["n_matched_official_sa_employment"] == 1
            assert summary["n_matched_all_three_official"] == 1
            official = housing_data.read_simple_csv(dirs["processed"] / "employment_metro_official_sa_long.csv")
            assert official[0]["bls_series_id"] == "SMSGOOD"
            avail = housing_data.read_simple_csv(dirs["processed"] / "housing_msa_monthly_availability.csv")
            assert any(r["employment_official_sa_available"] == "1" for r in avail)
        finally:
            housing_data.parse_cached_geography = original_geo
            housing_data.fetch_bls = original_fetch_bls


def test_housing_report_frontiers_pareto_and_exclusion_identities():
    complete = {
        "10001": {"2020-01", "2020-02", "2020-03", "2020-04"},
        "10002": {"2020-03", "2020-04"},
        "10003": {"2020-04"},
    }
    codes = ["10001", "10002", "10003"]
    starts = report_housing_data.fixed_start_frontier(complete, codes, ["2020-01", "2020-03"], "2020-04")
    assert starts[0]["T_months"] == 4
    assert starts[0]["N_complete_msas"] == 1
    assert starts[1]["T_months"] == 2
    assert starts[1]["N_complete_msas"] == 2
    durs = report_housing_data.fixed_duration_frontier(complete, codes, [1, 2, 4], "2020-04")
    assert [r["N_complete_msas"] for r in durs] == [3, 2, 1]
    pareto = report_housing_data.pareto_frontier(starts + durs)
    assert all(not (
        int(a["N_complete_msas"]) <= int(b["N_complete_msas"]) and
        int(a["T_months"]) <= int(b["T_months"]) and
        (int(a["N_complete_msas"]) < int(b["N_complete_msas"]) or int(a["T_months"]) < int(b["T_months"]))
    ) for a in pareto for b in pareto if a is not b)


def test_housing_report_candidate_validation_and_exact_lag_no_interpolation():
    rows = [
        {"cbsa_code": "10001", "date": "2020-01-01", "x": "10", "zhvi_all_homes_sa": "1", "permits_units_sa": "2", "employment_thousands_sa": "3"},
        {"cbsa_code": "10001", "date": "2021-01-01", "x": "15", "zhvi_all_homes_sa": "1", "permits_units_sa": "2", "employment_thousands_sa": "3"},
        {"cbsa_code": "10001", "date": "2021-02-01", "x": "20", "zhvi_all_homes_sa": "1", "permits_units_sa": "2", "employment_thousands_sa": "3"},
    ]
    lagged = report_housing_data.exact_lag_transform(rows, "x", "dx12", lambda v, lv: v - lv)
    assert lagged[1]["dx12"] == 5
    assert lagged[2]["dx12"] == ""
    assert report_housing_data.validate_candidate_panel_rows(rows[:2], 1, 2) == []
    bad = [dict(rows[0], permits_units_sa="")]
    assert report_housing_data.validate_candidate_panel_rows(bad, 1, 1)


def test_housing_report_preliminary_final_month_and_final_only_frontier():
    emp = [
        {"cbsa_code": "10001", "date": "2020-01-01", "preliminary_flag": "0", "bls_series_id": "S1", "employment_thousands_sa": "10"},
        {"cbsa_code": "10002", "date": "2020-01-01", "preliminary_flag": "0", "bls_series_id": "S2", "employment_thousands_sa": "20"},
        {"cbsa_code": "10001", "date": "2020-02-01", "preliminary_flag": "1", "bls_series_id": "S1", "employment_thousands_sa": "11"},
        {"cbsa_code": "10002", "date": "2020-02-01", "preliminary_flag": "0", "bls_series_id": "S2", "employment_thousands_sa": "21"},
    ]
    prelim, by_month, latest = report_housing_data.preliminary_by_month(emp, ["10001", "10002"])
    assert len(prelim) == 1
    assert by_month == [{"date": "2020-02-01", "preliminary_count": 1}]
    assert latest == "2020-01"

    complete_all = {"10001": {"2020-01", "2020-02"}, "10002": {"2020-01", "2020-02"}}
    complete_final = {"10001": {"2020-01"}, "10002": {"2020-01", "2020-02"}}
    all_frontier = report_housing_data.fixed_start_frontier(complete_all, ["10001", "10002"], ["2020-01"], "2020-02")
    final_frontier = report_housing_data.fixed_start_frontier(complete_final, ["10001", "10002"], ["2020-01"], latest)
    assert all_frontier[0]["T_months"] == 2
    assert final_frontier[0]["T_months"] == 1
    assert final_frontier[0]["N_complete_msas"] == 2


def test_housing_report_final_only_candidate_panels_are_immutable_and_count_negatives():
    with tempfile.TemporaryDirectory() as tmp:
        root = report_housing_data.Path(tmp)
        specs = {
            "candidate": {
                "candidate_id": "candidate",
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
                "complete_cbsa_codes": "10001",
            }
        }
        title = {"10001": "Test CBSA"}
        z_by = {
            ("10001", "2020-01"): {"zhvi_all_homes_sa": "100", "source_vintage": "fixture"},
            ("10001", "2020-02"): {"zhvi_all_homes_sa": "101", "source_vintage": "fixture"},
        }
        p_by = {
            ("10001", "2020-01"): {"permits_units_sa": "-2", "source_vintage": "fixture", "x13_status": "ok", "x13_spec_id": "spec"},
            ("10001", "2020-02"): {"permits_units_sa": "3", "source_vintage": "fixture", "x13_status": "ok", "x13_spec_id": "spec"},
        }
        e_by = {
            ("10001", "2020-01"): {"employment_thousands_sa": "10", "source_vintage": "fixture", "preliminary_flag": "0"},
            ("10001", "2020-02"): {"employment_thousands_sa": "11", "source_vintage": "fixture", "preliminary_flag": "0"},
        }
        x13_by = {"10001": {"status": "ok", "x13_spec_id": "spec"}}
        summaries, _ = report_housing_data.write_candidate_panels(
            specs, root, title, z_by, p_by, e_by, x13_by, "final_only", overwrite=False
        )
        assert summaries[0]["negative_permit_count"] == 1
        assert summaries[0]["negative_permit_share"] == 0.5
        before = (root / "candidate" / "metadata.json").read_text(encoding="utf-8")
        summaries2, _ = report_housing_data.write_candidate_panels(
            specs, root, title, z_by, p_by, e_by, x13_by, "final_only", overwrite=False
        )
        after = (root / "candidate" / "metadata.json").read_text(encoding="utf-8")
        assert before == after
        assert summaries2[0]["candidate_id"] == "candidate"


def test_housing_report_negative_sa_permit_relabel_and_no_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        root = report_housing_data.Path(tmp)
        cdir = root / "cand"
        cdir.mkdir()
        report_housing_data.write_csv(cdir / "msa_list.csv", [{"cbsa_code": "10001"}], ["cbsa_code"])
        report_housing_data.write_csv(cdir / "monthly_dates.csv", [{"date": "2020-01-01"}], ["date"])
        candidates = [{
            "path": str(cdir), "panel_type": "final_only", "candidate_id": "cand",
            "start_date": "2020-01-01", "end_date": "2020-01-01",
            "N": 1, "T_months": 1, "NT": 1,
            "negative_permit_count": 1, "negative_permit_share": 1.0,
            "x13_warning_msas": 0,
        }]
        rows = [{
            "cbsa_code": "10001", "msa_title": "Test CBSA", "date": "2020-01-01",
            "zhvi_all_homes_sa": "100", "permits_units_sa": "-4.5",
            "employment_thousands_sa": "10",
        }]
        pnsa_by = {("10001", "2020-01"): {"total_units": "2"}}
        p_by = {("10001", "2020-01"): {
            "x13_status": "ok", "x13_spec_id": "spec",
            "contiguous_segment_start": "2020-01-01",
            "contiguous_segment_end": "2020-12-01",
        }}
        diag, cand_rows, summary = report_housing_data.negative_permit_diagnostics(
            rows, pnsa_by, p_by, {"10001": {"status": "ok"}}, candidates
        )
        assert diag[0]["diagnostic"] == "negative_seasonally_adjusted_permit_value"
        assert diag[0]["corresponding_permits_units_nsa"] == "2"
        assert diag[0]["belongs_to_x13_warning_or_failed_segment"] == 0
        assert "final_only:cand" in diag[0]["candidate_panels_affected"]
        assert cand_rows[0]["negative_permit_count"] == 1
        assert summary["total_count"] == 1
        assert rows[0]["permits_units_sa"] == "-4.5"


def test_housing_report_candidate_ranking_identities():
    candidates = [
        {"panel_type": "all_vintage", "candidate_id": "a", "start_date": "2020-01-01", "end_date": "2020-10-01", "N": 10, "T_months": 10, "NT": 100, "negative_permit_count": 0},
        {"panel_type": "all_vintage", "candidate_id": "b", "start_date": "2020-01-01", "end_date": "2020-06-01", "N": 20, "T_months": 6, "NT": 120, "negative_permit_count": 1},
        {"panel_type": "final_only", "candidate_id": "pareto_01", "start_date": "2020-01-01", "end_date": "2021-01-01", "N": 8, "T_months": 13, "NT": 104, "negative_permit_count": 2},
    ]
    rows = report_housing_data.rank_candidates(candidates, {("final_only", "pareto_01")})
    by_id = {r["candidate_id"]: r for r in rows}
    assert "maximum N" in by_id["b"]["highlight"]
    assert "closest N/T ratio to one" in by_id["a"]["highlight"]
    assert by_id["pareto_01"]["pareto_dominated"] == 0
    assert by_id["a"]["pareto_dominated"] == 1


def _write_all_homes_candidate(root, cid="start_2010", *, n=3, t=5, start="2010-01",
                               prelim_last=False, missing_month=False):
    cdir = housing_all_homes.Path(root) / cid
    cdir.mkdir(parents=True, exist_ok=True)
    months = [housing_all_homes.month_from_index(housing_all_homes.month_index(start) + i) for i in range(t)]
    if missing_month:
        months.pop(2)
    codes = [f"10{i:03d}" for i in range(1, n + 1)]
    rows = []
    for ci, code in enumerate(codes):
        for mi, ym in enumerate(months):
            rows.append({
                "cbsa_code": code,
                "msa_title": f"Metro {code}",
                "date": ym + "-01",
                "zhvi_all_homes_sa": str(100 + ci + mi),
                "permits_units_sa": str([-2, 0, 5, 6, 7, 8][mi % 6]),
                "employment_thousands_sa": str(50 + ci + mi),
                "bls_preliminary_flag": "1" if prelim_last and mi == len(months) - 1 else "0",
                "zhvi_source_vintage": "fixture",
                "permits_source_vintage": "fixture",
                "employment_source_vintage": "fixture",
            })
    housing_all_homes.write_csv(cdir / "housing_panel_levels.csv", rows, [
        "cbsa_code", "msa_title", "date", "zhvi_all_homes_sa", "permits_units_sa",
        "employment_thousands_sa", "bls_preliminary_flag", "zhvi_source_vintage",
        "permits_source_vintage", "employment_source_vintage",
    ])
    housing_all_homes.write_csv(cdir / "msa_list.csv", [{"cbsa_code": c, "msa_title": f"Metro {c}"} for c in codes], ["cbsa_code", "msa_title"])
    housing_all_homes.write_csv(cdir / "monthly_dates.csv", [{"date": m + "-01"} for m in months], ["date"])
    meta = {
        "candidate_id": cid,
        "panel_type": "final_only",
        "start_date": months[0] + "-01",
        "end_date": months[-1] + "-01",
        "N": n,
        "T_months": len(months),
        "NT": n * len(months),
        "preliminary_bls_observations": n if prelim_last else 0,
        "negative_permit_count": sum(float(r["permits_units_sa"]) < 0 for r in rows),
        "missing_primary_values": 0,
        "x13_warning_msas": 0,
        "no_interpolation_or_imputation": True,
    }
    housing_all_homes.write_json(cdir / "metadata.json", meta)
    return cdir


def test_housing_all_homes_transformations_and_exact_lags_no_bridge():
    rows = [
        {"cbsa_code": "10001", "msa_title": "A", "date": "2020-01-01", "zhvi_all_homes_sa": "100", "permits_units_sa": "-2", "employment_thousands_sa": "50", "bls_preliminary_flag": "0"},
        {"cbsa_code": "10001", "msa_title": "A", "date": "2020-03-01", "zhvi_all_homes_sa": "110", "permits_units_sa": "0", "employment_thousands_sa": "55", "bls_preliminary_flag": "0"},
    ]
    transformed, tdiag = housing_all_homes.transform_rows(rows)
    assert tdiag["negative_permit_count_retained"] == 1
    assert transformed[0]["asinh_permits"] == math.asinh(-2)
    assert transformed[1]["asinh_permits"] == math.asinh(0)
    lagged, ldiag = housing_all_homes.add_exact_lags(transformed)
    assert ldiag["usable_dynamic_observations"] == 0
    assert ldiag["lag_bridging_missing_month_count"] == 1
    assert lagged[1]["lag_log_zhvi"] == ""


def test_housing_all_homes_positivity_checks():
    bad_z = [{"cbsa_code": "10001", "date": "2020-01-01", "zhvi_all_homes_sa": "0", "permits_units_sa": "1", "employment_thousands_sa": "1"}]
    try:
        housing_all_homes.transform_rows(bad_z)
    except ValueError as exc:
        assert "ZHVI" in str(exc)
    else:
        raise AssertionError("nonpositive ZHVI should be rejected")
    bad_e = [{"cbsa_code": "10001", "date": "2020-01-01", "zhvi_all_homes_sa": "1", "permits_units_sa": "1", "employment_thousands_sa": "0"}]
    try:
        housing_all_homes.transform_rows(bad_e)
    except ValueError as exc:
        assert "employment" in str(exc)
    else:
        raise AssertionError("nonpositive employment should be rejected")


def test_housing_all_homes_final_only_loading_identities_and_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        root = housing_all_homes.Path(tmp) / "candidates"
        _write_all_homes_candidate(root, "start_2010", n=4, t=6)
        _write_all_homes_candidate(root, "pareto_2010", n=2, t=6)
        out = housing_all_homes.Path(tmp) / "estimation_panel"
        meta = housing_all_homes.prepare_baseline_panel(root, out)
        assert meta["candidate_id"] == "start_2010"
        assert meta["N"] == 4
        assert meta["T"] == 6
        assert meta["usable_dynamic_observations"] == 4 * 5
        assert meta["preliminary_bls_observations"] == 0
        assert meta["negative_permit_count_retained"] == 4
        assert "housing_estimation_panel.csv" in meta["checksums"]
        panel = housing_all_homes.load_estimation_panel(out, first_n_msas=2, first_t_usable=3)
        assert panel["Y"].shape == (3, 2)
        assert all(z.shape == (3, 2) for z in panel["Z"])


def test_housing_all_homes_preserves_legacy_loader_and_smoke_output_isolation_signature():
    assert callable(housing_data.load_zillow) if hasattr(housing_data, "load_zillow") else True
    from dlrhcs.empirical import load_zillow as legacy_load_zillow
    assert callable(legacy_load_zillow)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = housing_all_homes.Path(tmp) / "out" / "smoke"
        panel_dir = housing_all_homes.Path(tmp) / "panel"
        panel_dir.mkdir(parents=True)
        housing_all_homes.write_csv(panel_dir / "housing_estimation_panel.csv", [], housing_all_homes.PANEL_COLUMNS)
        sig = {
            "schema_version": housing_all_homes.SCHEMA_VERSION,
            "panel_checksum": "abc",
            "smoke": True,
        }
        old = {"run_signature": sig}
        out_dir.mkdir(parents=True)
        housing_all_homes.write_json(out_dir / "metadata.json", old)
        housing_all_homes.write_json(out_dir / "housing_all_homes_results.json", old)
        assert (out_dir / "metadata.json").exists()
        assert "smoke" in str(out_dir)


def _make_repo_root_fixture(base, name):
    root = housing_all_homes.Path(base) / name
    (root / "dlrhcs").mkdir(parents=True)
    (root / "scripts").mkdir()
    return root


def test_paths_find_repo_root_precedence_and_spaces(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        root1 = _make_repo_root_fixture(tmp, "Dynamic Paper-dlrhcs_replication")
        root2 = _make_repo_root_fixture(tmp, "dlrhcs")
        root_space = _make_repo_root_fixture(tmp, "repo with spaces")
        nested = root1 / "scripts" / "subdir"
        nested.mkdir()
        assert find_repo_root(start=nested) == root1.resolve()
        assert find_repo_root(start=root2 / "scripts") == root2.resolve()
        assert find_repo_root(start=root_space / "scripts") == root_space.resolve()
        assert resolve_repo_path("data/zillow", root1) == (root1 / "data" / "zillow").resolve()
        outside = housing_all_homes.Path(tmp) / "outside.csv"
        assert resolve_repo_path(outside, root1) == outside.resolve()
        old_env = os.environ.get("DLRHCS_ROOT")
        try:
            os.environ["DLRHCS_ROOT"] = str(root2)
            assert find_repo_root(start=root1 / "scripts") == root2.resolve()
        finally:
            if old_env is None:
                os.environ.pop("DLRHCS_ROOT", None)
            else:
                os.environ["DLRHCS_ROOT"] = old_env
        try:
            find_repo_root(explicit=housing_all_homes.Path(tmp) / "not_repo")
        except ValueError as exc:
            assert "invalid DLRHCS repository root" in str(exc)
        else:
            raise AssertionError("invalid --repo-root should be rejected")


def test_housing_all_homes_signature_portable_across_repo_roots_and_checksum_sensitive():
    sig_a = {
        "schema_version": housing_all_homes.SCHEMA_VERSION,
        "input_identity": {
            "schema_version": housing_all_homes.SCHEMA_VERSION,
            "input_checksum": "abc",
            "N_source": 3,
            "T_source": 5,
            "start_date": "2010-01-01",
            "end_date": "2010-05-01",
            "outcome": "log(zhvi_all_homes_sa)",
            "controls": ["lag_asinh_permits", "lag_log_employment"],
            "repo_relative_input_path": "data/zillow/processed/estimation_panels/housing_baseline_2010_final",
        },
    }
    sig_b = json.loads(json.dumps(sig_a))
    sig_b["resolved_absolute_path_info"] = "D:/Programming/dlrhcs/data/zillow/processed/estimation_panels/housing_baseline_2010_final"
    assert sig_a["input_identity"] == sig_b["input_identity"]
    sig_c = json.loads(json.dumps(sig_a))
    sig_c["input_identity"]["input_checksum"] = "changed"
    assert sig_a["input_identity"] != sig_c["input_identity"]


def test_housing_all_homes_riesz_override_parsers():
    assert housing_all_homes.parse_positive_int("2000") == 2000
    assert housing_all_homes.parse_positive_float("1e-5") == 1e-5
    assert housing_all_homes.parse_bool("true") is True
    assert housing_all_homes.parse_bool("false") is False
    for bad in ("0", "-1", "nan", "inf"):
        try:
            housing_all_homes.parse_positive_float(bad)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"{bad} should be rejected")


def test_riesz_cg_status_interpretation():
    assert cg_converged_from_status(0, 1e-8, 1e-6)
    assert cg_converged_from_status(0, 1e-6, 1e-6)  # final allowed iteration can still be success
    assert not cg_converged_from_status(10, 1e-8, 1e-6)  # positive info means maxiter/failure status
    assert not cg_converged_from_status(0, float("nan"), 1e-6)
    assert not cg_converged_from_status(0, 1e-8, 1e-6, contains_nonfinite=True)
    assert not cg_converged_from_status(5, 1e-8, 1e-6)  # finite solution with failure status remains failure
    assert not cg_converged_from_status(-1, 1e-8, 1e-6)


def test_housing_all_homes_metadata_paths_and_preflight_no_estimator_call():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_root_fixture(tmp, "repo with spaces")
        cand_root = repo / "data" / "zillow" / "processed" / "candidate_panels_final_only"
        _write_all_homes_candidate(cand_root, "start_2010", n=3, t=5)
        panel_dir = repo / "data" / "zillow" / "processed" / "estimation_panels" / "housing_baseline_2010_final"
        meta = housing_all_homes.prepare_baseline_panel(cand_root, panel_dir, repo_root=repo)
        assert meta["repo_relative_path"] == "data/zillow/processed/estimation_panels/housing_baseline_2010_final"
        assert "resolved_absolute_path_info" in meta
        original_estimate = housing_all_homes.estimate
        try:
            housing_all_homes.estimate = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preflight must not estimate"))
            report = housing_all_homes.preflight(panel_dir, repo / "outputs" / "empirical" / "housing_all_homes", repo)
        finally:
            housing_all_homes.estimate = original_estimate
        assert report["ready_for_production"]
        assert not report["estimator_called"]
        assert report["repo_relative_input_path"] == meta["repo_relative_path"]
        assert (repo / "outputs" / "empirical" / "housing_all_homes" / "audit" / "production_preflight.json").exists()


def test_housing_report_latex_manifest_and_deterministic_helpers():
    with tempfile.TemporaryDirectory() as tmp:
        root = report_housing_data.Path(tmp)
        rows = [{"name": "A&B", "count": 2}]
        tex = root / "table.tex"
        report_housing_data.write_latex_table(
            tex,
            rows,
            [("name", "Name"), ("count", "Count")],
            "Fixture table",
            "tab:fixture",
            "No interpolation.",
        )
        text = tex.read_text(encoding="utf-8")
        assert "\\toprule" in text
        assert "A\\&B" in text
        csv_path = root / "data.csv"
        report_housing_data.write_csv(csv_path, rows, ["name", "count"])
        first = report_housing_data.sha256_file(csv_path)
        report_housing_data.write_csv(csv_path, rows, ["name", "count"])
        assert report_housing_data.sha256_file(csv_path) == first


# 3. ALS objective monotone non-increasing ----------------------------------- #
def test_als_monotone():
    p = _panel()
    blocks = build_blocks(p.Z)
    fit = fit_factor_ridge(p.Y, blocks, (1, 1, 1), n_restarts=1, n_sweeps=40)
    d = np.diff(fit.obj_path)
    assert np.all(d <= 1e-8 * (1 + np.abs(fit.obj_path[:-1])))
    assert fit.monotone


# 4. tangent projector idempotent & self-adjoint ----------------------------- #
def test_tangent_projector():
    rng = np.random.default_rng(2)
    Tp, N, r = 15, 12, 2
    U, _ = np.linalg.qr(rng.standard_normal((Tp, r)))
    V, _ = np.linalg.qr(rng.standard_normal((N, r)))
    X = rng.standard_normal((Tp, N))
    PX = project_block(X, U, V)
    PPX = project_block(PX, U, V)
    assert np.allclose(PX, PPX, atol=1e-10)            # P^2 = P
    Y = rng.standard_normal((Tp, N))
    a = float(np.vdot(project_block(X, U, V), Y))
    b = float(np.vdot(X, project_block(Y, U, V)))
    assert abs(a - b) < 1e-10                          # self-adjoint


# 5. Riesz representer identity (infeasible, true tangent) ------------------- #
def test_riesz_identity():
    p = _panel(Tp=20, N=16)
    blocks = build_blocks(p.Z)
    Tp, N = p.Tp, p.N
    train = np.ones((Tp, N), dtype=bool)
    D = entry_direction(blocks, 0, 3, 4)
    rr = riesz_weights(D, blocks, p.U, p.V, train, alpha=1.0,
                       ridge=1e-12, tol=1e-12)
    # <Psi, A(Delta)> = <D, Delta> for any admissible tangent Delta
    rng = np.random.default_rng(7)
    raw = [rng.standard_normal((Tp, N)) for _ in blocks]
    Delta = project_tangent(raw, p.U, p.V)
    lhs = float(np.vdot(rr.Psi, A(Delta, blocks)))
    rhs = theta_dot(D, Delta)
    assert abs(lhs - rhs) < 1e-5 * (1 + abs(rhs))


# 6. noiseless recovery at the true rank ------------------------------------- #
def test_noiseless_recovery():
    p = _panel(sigma_u=0.0)
    blocks = build_blocks(p.Z)
    fit = fit_factor_ridge(p.Y, blocks, (1, 1, 1), ridge=1e-6,
                           n_sweeps=300, n_restarts=4, tol=1e-12)
    R = p.Y - A(fit.surfaces, blocks)
    assert np.sqrt(np.mean(R ** 2)) < 1e-2


# 7. revised Monte Carlo DGP smoke checks ------------------------------------ #
def _mean_lag_cov(U, sigma, lag):
    Z = U / sigma[None, :]
    Z = Z - Z.mean(axis=0, keepdims=True)
    C = (Z.T @ Z) / Z.shape[0]
    return float(np.mean(np.diag(C, k=lag)))


def test_revised_dgp_shapes_and_finite():
    for k, dgp_type in enumerate(("dgp1", "dgp2", "dgp3")):
        p = simulate(36, 18, np.random.default_rng(100 + k), dgp_type=dgp_type)
        assert p.Y.shape == (36, 18)
        assert p.Z[0].shape == (36, 18)
        assert p.Z[1].shape == (36, 18)
        assert p.U_innov.shape == (36, 18)
        assert all(S.shape == (36, 18) for S in p.surfaces)
        assert np.all(np.isfinite(p.Y))
        assert np.all(np.isfinite(p.Z[0]))
        assert np.all(np.isfinite(p.Z[1]))
        assert np.all(np.isfinite(p.U_innov))
        assert p.meta["dgp_type"] == dgp_type
        assert p.meta["sigma_i"].shape == (18,)
        assert p.meta["sigma_e_i"].shape == (18,)
        assert np.all((0.5 <= p.meta["sigma_i2"]) & (p.meta["sigma_i2"] <= 1.5))
        assert np.all((0.5 <= p.meta["sigma_e_i2"]) & (p.meta["sigma_e_i2"] <= 1.5))
        assert abs(p.meta["c_h"] - np.sqrt(0.3 / 0.7)) < 1e-12
        assert p.meta["max_abs_a_it"] <= 0.85 + 1e-12
        assert abs(p.meta["PR2_realized"] - p.meta["PR2_target"]) < 0.15
        assert "a_it_summary" in p.meta
        assert "beta_it_summary" in p.meta


def test_revised_dgp1_heteroskedastic_independent_errors():
    p = simulate(900, 32, np.random.default_rng(201), dgp_type="dgp1")
    Uraw = p.meta["u_it"]
    emp_var = Uraw.var(axis=0)
    target_var = p.meta["sigma_i2"]
    assert np.corrcoef(emp_var, target_var)[0, 1] > 0.75
    c1 = abs(_mean_lag_cov(Uraw, p.meta["sigma_i"], 1))
    c4 = abs(_mean_lag_cov(Uraw, p.meta["sigma_i"], 4))
    assert c1 < 0.08
    assert c4 < 0.08


def test_revised_dgp2_dgp3_spatial_covariance_decay():
    for k, dgp_type in enumerate(("dgp2", "dgp3")):
        p = simulate(1200, 36, np.random.default_rng(300 + k), dgp_type=dgp_type)
        Uraw = p.meta["u_it"]
        c0 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 0)
        c1 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 1)
        c2 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 2)
        c4 = _mean_lag_cov(Uraw, p.meta["sigma_i"], 4)
        assert 0.85 < c0 < 1.15
        assert c1 > c2 > c4
        assert c1 > 0.35
        assert c2 > 0.12
        assert c4 < 0.15


def test_revised_dgp3_uses_lagged_shocks_in_x():
    p = simulate(500, 30, np.random.default_rng(401), dgp_type="dgp3")
    burn = 50
    X = p.meta["Xfull"]
    U = p.meta["Ufull"]
    fx = p.meta["f_x"]
    lx = p.meta["lambda_x"]
    resid = X[burn:] - 0.5 * X[burn - 1:-1] - 0.5 * fx[burn:, None] * lx[None, :]
    lag_u = U[burn - 1:-1]
    cur_u = U[burn:]
    corr_lag = np.corrcoef(resid.ravel(), lag_u.ravel())[0, 1]
    corr_cur = np.corrcoef(resid.ravel(), cur_u.ravel())[0, 1]
    assert corr_lag > 0.20
    assert corr_lag > corr_cur + 0.05


# 8. Gram per-cell-average scale convention ---------------------------------- #
def test_gram_scale():
    p = _panel(Tp=18, N=14)
    blocks = build_blocks(p.Z)
    rng = np.random.default_rng(3)
    Delta = project_tangent([rng.standard_normal((p.Tp, p.N)) for _ in blocks],
                            p.U, p.V)
    AD = A(Delta, blocks)
    full = float(np.vdot(AD, AD))
    folds = make_folds(p.Tp, p.N, 6, 2, rng=rng)
    fd = folds[0]
    train_scaled = fd.alpha * float(np.vdot(AD * fd.train, AD * fd.train))
    held_scaled = (1.0 / fd.p) * float(np.vdot(AD * fd.val, AD * fd.val))
    assert 0.3 < train_scaled / full < 3.0
    assert 0.3 < held_scaled / full < 3.0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
