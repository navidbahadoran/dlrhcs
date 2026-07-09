"""
Monte Carlo harness (spec sec 12) with deterministic seeds, JSONL checkpointing
(resume-safe), and an optional process-parallel backend.

Three studies:
  * ``run_grid``        -- feasible convergence study over the (Tp,N) grid.
  * ``run_grid`` w/ oracle=True  -- the infeasible oracle benchmark (true tangent
                          spaces in the Riesz solve), the sec-12 checkpoint.
  * ``run_purge_sweep`` -- forward-exclusion-window sensitivity at fixed (Tp,N).

Per-replication seeds are derived as ``SeedSequence([master, rep])`` so any rep
can be (re)run independently and reproducibly, on any number of workers.

Parallelism note: NumPy/OpenBLAS deadlocks under ``fork``; we use joblib's
``loky`` backend (separate interpreters) and recommend ``OMP_NUM_THREADS=1`` so
each worker is single-threaded (set in ``run_all``).
"""
from __future__ import annotations

import json
import os
import dataclasses
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .design import build_blocks
from .dgp import simulate
from .pipeline import Tuning, estimate
from .targets import make_target, group_weights


def _merge_dgp_selector(dgp_kwargs=None, dgp_type=None, dgp_id=None):
    """Return simulator kwargs with an optional explicit DGP selector threaded in."""
    out = dict(dgp_kwargs or {})
    if dgp_type is not None:
        if "dgp_type" in out and out["dgp_type"] != dgp_type:
            raise ValueError(f"conflicting dgp_type={out['dgp_type']!r} and {dgp_type!r}")
        out["dgp_type"] = dgp_type
    if dgp_id is not None:
        if "dgp_id" in out and out["dgp_id"] != dgp_id:
            raise ValueError(f"conflicting dgp_id={out['dgp_id']!r} and {dgp_id!r}")
        out["dgp_id"] = dgp_id
    return out


def _normalize_record_dgp(dgp_kwargs):
    raw = (dgp_kwargs or {}).get("dgp_type", (dgp_kwargs or {}).get("dgp_id"))
    if raw is None:
        return "legacy"
    if isinstance(raw, int):
        return f"dgp{raw}"
    key = str(raw).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in ("1", "dgp1", "hetero", "independenthetero", "independentheteroskedastic"):
        return "dgp1"
    if key in ("2", "dgp2", "spatial", "spatialhetero", "spatialheteroskedastic"):
        return "dgp2"
    if key in ("3", "dgp3", "predetermined", "predeterminedcovariates"):
        return "dgp3"
    return str(raw)


def _simulate_kwargs(dgp_kwargs):
    """Drop MC-only metadata before calling the DGP simulator."""
    out = dict(dgp_kwargs or {})
    out.pop("true_rank", None)
    out.pop("true_ranks", None)
    return out


def _mc_tuning_for_dgp(tuning: Tuning, dgp_kind: str, N: int) -> Tuning:
    """Use theorem-aligned simulation SEs without changing estimator internals.

    The second SE produced by ``pipeline.estimate`` is made the lattice Bartlett
    spatial-kernel SE for Monte Carlo runs.  DGP 1 reports White inference as the
    main object; DGP 2/3 report both White and spatial-kernel inference.
    """
    bw = max(1, int(np.floor(N ** (1.0 / 3.0))))
    return dataclasses.replace(tuning, xs_kernel="bartlett", xs_bandwidth=bw)


def _rank_vector(value, B, *, default=None):
    """Normalize a scalar/list rank specification to a length-B integer vector."""
    if value is None:
        value = default
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return [int(value)] * B
    out = [int(v) for v in value]
    if len(out) != B:
        raise ValueError(f"rank vector has length {len(out)}, expected {B}")
    return out


def _true_rank_vector(panel, dgp_kwargs, B):
    """Infer the DGP rank vector, with explicit config metadata taking priority."""
    dgp_kwargs = dgp_kwargs or {}
    for key in ("true_ranks", "true_rank"):
        if key in dgp_kwargs:
            return _rank_vector(dgp_kwargs[key], B)
    if "true_ranks" in panel.meta:
        return _rank_vector(panel.meta["true_ranks"], B)
    if "true_rank" in panel.meta:
        return _rank_vector(panel.meta["true_rank"], B)
    return _rank_vector(dgp_kwargs.get("r", 1), B)


def _rank_selection_flags(selected, true):
    selected = np.asarray(selected, dtype=int)
    true = np.asarray(true, dtype=int)
    if selected.shape != true.shape:
        raise ValueError(f"selected rank shape {selected.shape} != true rank shape {true.shape}")
    below = bool(np.any(selected < true))
    above = bool(np.any(selected > true))
    return dict(rank_exact_correct=bool(np.array_equal(selected, true)),
                rank_underfit=bool(below and not above),
                rank_overfit=bool(above and not below),
                rank_mixed_misspecification=bool(below and above),
                rank_has_underfit_component=below,
                rank_has_overfit_component=above)


# --------------------------------------------------------------------------- #
#  the eight standard MC targets and their truths
# --------------------------------------------------------------------------- #
def standard_targets(blocks, Tp, N, t0=None, i0=None):
    """Entry, group mean, full mean and between-group contrast for lag & slope."""
    t0 = Tp // 2 if t0 is None else t0
    i0 = N // 2 if i0 is None else i0
    g1 = np.arange(0, N // 2)
    g2 = np.arange(N // 2, N)
    w1, w2 = group_weights(N, g1), group_weights(N, g2)
    wf = np.full(N, 1.0 / N)
    targets = []
    for blk, lab in [(0, "lag"), (1, "slope")]:
        targets += [
            make_target(blocks, f"{lab}_entry", blk, "entry", t=t0, i=i0),
            make_target(blocks, f"{lab}_gmean", blk, "mean", t=t0, weights=w1),
            make_target(blocks, f"{lab}_fmean", blk, "mean", t=t0, weights=wf),
            make_target(blocks, f"{lab}_contrast", blk, "contrast", t=t0,
                        weights=w1, weights2=w2),
        ]
    ctx = dict(t0=t0, i0=i0, w1=w1, w2=w2, wf=wf)
    return targets, ctx


def true_value(panel, tg, ctx):
    S = {0: panel.surfaces[0], 1: panel.surfaces[1]}[tg.block]
    name, t0 = tg.name, ctx["t0"]
    if "entry" in name:
        return float(S[t0, ctx["i0"]])
    if "gmean" in name:
        return float(S[t0] @ ctx["w1"])
    if "fmean" in name:
        return float(S[t0] @ ctx["wf"])
    return float(S[t0] @ (ctx["w1"] - ctx["w2"]))


# --------------------------------------------------------------------------- #
#  one replication
# --------------------------------------------------------------------------- #
def run_replication(Tp, N, rep, tuning: Tuning, *, oracle=False,
                    dgp_kwargs=None, dgp_type=None, dgp_id=None,
                    master=2024) -> Dict:
    dgp_kwargs = _merge_dgp_selector(dgp_kwargs, dgp_type=dgp_type, dgp_id=dgp_id)
    dgp_kind = _normalize_record_dgp(dgp_kwargs)
    tuning = _mc_tuning_for_dgp(tuning, dgp_kind, N)
    sim_rng = np.random.default_rng(np.random.SeedSequence([master, rep]))
    est_rng = np.random.default_rng(np.random.SeedSequence([master + 1, rep]))
    panel = simulate(Tp, N, sim_rng, **_simulate_kwargs(dgp_kwargs))
    blocks = build_blocks(panel.Z)
    true_ranks = _true_rank_vector(panel, dgp_kwargs, len(blocks))
    targets, ctx = standard_targets(blocks, Tp, N)
    res = estimate(panel.Y, panel.Z, targets, tuning, rng=est_rng,
                   oracle=oracle, true_U=panel.U, true_V=panel.V)
    rec = {"rep": int(rep)}
    for tg in targets:
        v = true_value(panel, tg, ctx)
        lo, hi = res.ci[tg.name]
        lox, hix = res.ci_xs[tg.name]
        plug = res.onestep.plugins.get(tg.name, float("nan"))
        est = float(res.estimates[tg.name])
        row = dict(true_value=float(v), estimate=est,
                   err=est - v,
                   plugin_err=float(plug - v),
                   se=res.se[tg.name],
                   se_white=res.se[tg.name],
                   cov=int(lo <= v <= hi))
        if dgp_kind != "dgp1":
            row.update(se_xs=res.se_xs[tg.name],
                       se_spatial=res.se_xs[tg.name],
                       cov_xs=int(lox <= v <= hix))
        rec[tg.name] = row
    selected_ranks = [int(x) for x in res.ranks]
    rank_flags = _rank_selection_flags(selected_ranks, true_ranks)
    rec["_q"], rec["_J"], rec["_ranks"] = res.q, res.J, selected_ranks
    rec["_selected_ranks"] = selected_ranks
    rec["_true_ranks"] = [int(x) for x in true_ranks]
    rec["_rank_exact_correct"] = bool(rank_flags["rank_exact_correct"])
    rec["_rank_underfit"] = bool(rank_flags["rank_underfit"])
    rec["_rank_overfit"] = bool(rank_flags["rank_overfit"])
    rec["_rank_mixed_misspecification"] = bool(rank_flags["rank_mixed_misspecification"])
    rec["_rank_has_underfit_component"] = bool(rank_flags["rank_has_underfit_component"])
    rec["_rank_has_overfit_component"] = bool(rank_flags["rank_has_overfit_component"])
    rec["_rank_selection_enabled"] = bool(tuning.select)
    rec["_rank_candidate_caps"] = ([int(x) for x in tuning.r_bar]
                                   if tuning.r_bar is not None else None)
    rec["_tuning_fixed_ranks"] = ([int(x) for x in tuning.ranks]
                                  if tuning.ranks is not None else None)
    rec["_kappa_c"] = float(tuning.kappa_c)
    rec["_J_realized"] = int(res.diagnostics.get("J_realized", res.J))
    rec["_J_min"] = int(res.diagnostics.get("J_min", 0))
    rec["_c_J"] = float(res.diagnostics.get("c_J", float("nan")))
    rec["_J_rule_term"] = int(res.diagnostics.get("J_rule_term", 0))
    rec["_J_manual_override"] = bool(res.diagnostics.get("J_manual_override", False))
    rec["_B_TN_fold_rule"] = float(res.diagnostics.get("B_TN_fold_rule", float("nan")))
    rec["_L_TN_J"] = float(res.diagnostics.get("L_TN_J", float("nan")))
    rec["_r"] = int(getattr(tuning, "buffer_r", 0))
    rec["_retained"] = float(res.diagnostics.get("retained", float("nan")))
    rec["_retained_total"] = float(res.diagnostics.get("retained_total", float("nan")))
    rec["_retained_nonvalidation"] = float(res.diagnostics.get("retained_nonvalidation", float("nan")))
    rec["_retained_nonvalidation_min"] = float(res.diagnostics.get("retained_nonvalidation_min", float("nan")))
    rec["_retained_nonvalidation_max"] = float(res.diagnostics.get("retained_nonvalidation_max", float("nan")))
    rec["_validation_fold_size_mean"] = float(res.diagnostics.get("validation_fold_size_mean", float("nan")))
    rec["_validation_fold_share_mean"] = float(res.diagnostics.get("validation_fold_share_mean", float("nan")))
    rec["_monotone"] = bool(res.diagnostics["monotone"])
    rec["_dgp_type"] = panel.meta.get("dgp_type", "legacy")
    rec["_Tp"], rec["_N"] = int(Tp), int(N)
    rec["_se_xs_type"] = "spatial_kernel"
    rec["_spatial_bandwidth"] = int(tuning.xs_bandwidth)
    return rec


# --------------------------------------------------------------------------- #
#  checkpointed grid runner
# --------------------------------------------------------------------------- #
def _done_reps(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["rep"])
            except Exception:
                pass
    return done


def run_grid(Tp, N, R, tuning: Tuning, out_path, *, oracle=False,
             dgp_kwargs=None, dgp_type=None, dgp_id=None,
             master=2024, n_jobs=1, resume=True):
    """Run R replications at (Tp,N), appending JSONL records to ``out_path``.

    Resume-safe: already-recorded reps are skipped.  ``n_jobs>1`` uses joblib's
    loky backend if available.  Use ``dgp_type``/``dgp_id`` or
    ``dgp_kwargs={'dgp_type': ...}`` to select revised DGP 1--3.
    """
    dgp_kwargs = _merge_dgp_selector(dgp_kwargs, dgp_type=dgp_type, dgp_id=dgp_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done_reps(out_path) if resume else set()
    todo = [r for r in range(R) if r not in done]
    if not todo:
        return out_path

    def work(rep):
        return run_replication(Tp, N, rep, tuning, oracle=oracle,
                               dgp_kwargs=dgp_kwargs, master=master)

    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(work)(r) for r in todo)
    else:
        recs = [work(r) for r in todo]

    with open(out_path, "a") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
#  forward-exclusion-window sweep
# --------------------------------------------------------------------------- #
def run_purge_sweep(Tp, N, R, q_grid, base_tuning: Tuning, out_dir, *,
                    master=2024, n_jobs=1, dgp_kwargs=None,
                    dgp_type=None, dgp_id=None):
    dgp_kwargs = _merge_dgp_selector(dgp_kwargs, dgp_type=dgp_type, dgp_id=dgp_id)
    paths = {}
    for q in q_grid:
        tun = Tuning(**{**asdict(base_tuning), "q": int(q)})
        p = os.path.join(out_dir, f"purge_q{q}_{Tp}.jsonl")
        run_grid(Tp, N, R, tun, p, dgp_kwargs=dgp_kwargs,
                 master=master + 100 * int(q), n_jobs=n_jobs)
        paths[int(q)] = p
    return paths


# --------------------------------------------------------------------------- #
#  aggregation
# --------------------------------------------------------------------------- #
def aggregate(path) -> Dict[str, dict]:
    """Full per-target Monte Carlo battery: bias / abs bias / RMSE / MC sd; mean White
    and cross-sectional s.e. and 95% coverage and interval length; plug-in bias and
    RMSE (debiasing gain); and the studentized statistic's mean, sd, and 5/50/95
    quantiles (CLT evidence).  A ``_meta`` entry carries the retained share,
    monotonicity success rate, and the (q, r, J) exclusion settings."""
    recs = [json.loads(l) for l in open(path)]
    names = [k for k in recs[0] if not k.startswith("_") and k != "rep"]
    z95 = 1.959963984540054
    dgp_kind = recs[0].get("_dgp_type", "unknown")
    out = {}
    for nm in names:
        true = np.array([r[nm].get("true_value", np.nan) for r in recs], float)
        err = np.array([r[nm]["err"] for r in recs], float)
        est = np.array([r[nm].get("estimate", np.nan) for r in recs], float)
        if np.isnan(est).all():
            est = true + err
        se = np.array([r[nm]["se"] for r in recs], float)
        plug = np.array([r[nm].get("plugin_err", np.nan) for r in recs], float)
        zt = err / np.where(se > 0, se, np.nan)
        reject = np.where(np.isfinite(zt), np.abs(zt) > z95, np.nan)
        cover = np.where(se > 0, np.abs(err) <= z95 * se, np.nan)
        size = float(np.nanmean(reject))
        coverage = float(np.nanmean(cover))
        R = len(recs)

        def _binom_mcse(p):
            return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(R, 1)))

        def _nanmean_or_nan(x):
            x = np.asarray(x, float)
            return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

        row = dict(
            R=R,
            true_value=_nanmean_or_nan(true),
            mean_true_value=_nanmean_or_nan(true),
            mean_estimate=_nanmean_or_nan(est),
            bias=float(err.mean()), abs_bias=float(np.abs(err).mean()),
            rmse=float(np.sqrt((err ** 2).mean())), mc_sd=float(err.std()),
            mean_se=float(se.mean()),
            mean_se_white=float(se.mean()),
            size_5pct=float(size),
            size_5pct_white=float(size),
            coverage_95=float(coverage),
            coverage_95_white=float(coverage),
            size_mcse=float(_binom_mcse(size)),
            size_mcse_white=float(_binom_mcse(size)),
            coverage_mcse=float(_binom_mcse(coverage)),
            coverage_mcse_white=float(_binom_mcse(coverage)),
            cov=float(coverage),
            ci_len=float(2 * z95 * se.mean()),
            plugin_bias=float(np.nanmean(plug)),
            plugin_rmse=float(np.sqrt(np.nanmean(plug ** 2))),
            z_mean=float(np.nanmean(zt)), z_sd=float(np.nanstd(zt)),
            z_q05=float(np.nanpercentile(zt, 5)), z_q50=float(np.nanpercentile(zt, 50)),
            z_q95=float(np.nanpercentile(zt, 95)))
        if dgp_kind != "dgp1":
            se_spatial = np.array([r[nm].get("se_spatial", r[nm].get("se_xs", np.nan))
                                   for r in recs], float)
            zsp = err / np.where(se_spatial > 0, se_spatial, np.nan)
            reject_spatial = np.where(np.isfinite(zsp), np.abs(zsp) > z95, np.nan)
            cover_spatial = np.where(se_spatial > 0, np.abs(err) <= z95 * se_spatial, np.nan)
            size_spatial = float(np.nanmean(reject_spatial))
            coverage_spatial = float(np.nanmean(cover_spatial))
            row.update(
                mean_se_xs=float(se_spatial.mean()),
                mean_se_spatial=float(se_spatial.mean()),
                mean_se_spatial_kernel=float(se_spatial.mean()),
                size_5pct_xs=float(size_spatial),
                size_5pct_spatial=float(size_spatial),
                size_5pct_spatial_kernel=float(size_spatial),
                coverage_95_xs=float(coverage_spatial),
                coverage_95_spatial=float(coverage_spatial),
                coverage_95_spatial_kernel=float(coverage_spatial),
                size_mcse_xs=float(_binom_mcse(size_spatial)),
                size_mcse_spatial=float(_binom_mcse(size_spatial)),
                size_mcse_spatial_kernel=float(_binom_mcse(size_spatial)),
                coverage_mcse_xs=float(_binom_mcse(coverage_spatial)),
                coverage_mcse_spatial=float(_binom_mcse(coverage_spatial)),
                coverage_mcse_spatial_kernel=float(_binom_mcse(coverage_spatial)),
                cov_xs=float(coverage_spatial),
                ci_len_xs=float(2 * z95 * se_spatial.mean()),
                ci_len_spatial=float(2 * z95 * se_spatial.mean()),
                ci_len_spatial_kernel=float(2 * z95 * se_spatial.mean()),
                z_xs_mean=float(np.nanmean(zsp)), z_xs_sd=float(np.nanstd(zsp)),
                z_spatial_mean=float(np.nanmean(zsp)), z_spatial_sd=float(np.nanstd(zsp)))
        out[nm] = row
    def _meta_mean(key):
        vals = np.array([r.get(key, np.nan) for r in recs], float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    ret = np.array([r.get("_retained", np.nan) for r in recs], float)
    mono = np.array([float(r.get("_monotone", True)) for r in recs], float)
    se_types = ["white"] if dgp_kind == "dgp1" else ["white", "spatial_kernel"]
    out["_meta"] = dict(retained=float(np.nanmean(ret)),
                        retained_total=_meta_mean("_retained_total"),
                        retained_nonvalidation=_meta_mean("_retained_nonvalidation"),
                        retained_nonvalidation_min=_meta_mean("_retained_nonvalidation_min"),
                        retained_nonvalidation_max=_meta_mean("_retained_nonvalidation_max"),
                        validation_fold_size_mean=_meta_mean("_validation_fold_size_mean"),
                        validation_fold_share_mean=_meta_mean("_validation_fold_share_mean"),
                        retained_alias="retained_nonvalidation",
                        monotone_rate=float(mono.mean()),
                        q=int(recs[0].get("_q", 0)), r=int(recs[0].get("_r", 0)),
                        J=int(recs[0].get("_J", 0)),
                        J_realized=int(recs[0].get("_J_realized", recs[0].get("_J", 0))),
                        J_min=int(recs[0].get("_J_min", 0)),
                        c_J=float(recs[0].get("_c_J", float("nan"))),
                        J_rule_term=int(recs[0].get("_J_rule_term", 0)),
                        J_manual_override=bool(recs[0].get("_J_manual_override", False)),
                        B_TN_fold_rule=float(recs[0].get("_B_TN_fold_rule", float("nan"))),
                        L_TN_J=float(recs[0].get("_L_TN_J", float("nan"))),
                        dgp_type=dgp_kind,
                        Tp=int(recs[0].get("_Tp", 0)), N=int(recs[0].get("_N", 0)),
                        se_types=se_types,
                        spatial_kernel="bartlett_lattice",
                        spatial_bandwidth=int(recs[0].get("_spatial_bandwidth", 0)))
    try:
        out["_rank_frequency"] = aggregate_rank_frequency(path, recs=recs)
    except ValueError as exc:
        out["_rank_frequency"] = dict(available=False, reason=str(exc), R=len(recs))
    return out


def aggregate_rank_frequency(path, recs=None) -> Dict[str, object]:
    """Aggregate selected-rank frequencies from MC JSONL records.

    Returned probabilities are mutually exclusive: underfit/overfit exclude the
    mixed case, while ``p_mixed_misspecification`` captures rank vectors with at
    least one component below and another above the truth.
    """
    if recs is None:
        recs = [json.loads(l) for l in open(path)]
    R = len(recs)
    if R == 0:
        raise ValueError("cannot aggregate rank frequencies from an empty file")

    def _rank_key(vec):
        return ",".join(str(int(x)) for x in vec)

    selected = [r.get("_selected_ranks", r.get("_ranks")) for r in recs]
    true = [r.get("_true_ranks") for r in recs]
    if any(v is None for v in selected) or any(v is None for v in true):
        raise ValueError("rank-frequency aggregation requires _selected_ranks/_true_ranks metadata")
    enabled = [bool(r.get("_rank_selection_enabled", False)) for r in recs]
    if not all(enabled):
        if any(enabled):
            raise ValueError("rank-frequency aggregation requires all records to use rank selection")
        raise ValueError("rank selection was not enabled; fixed-rank records are not rank-frequency results")

    counts = {}
    for vec in selected:
        key = _rank_key(vec)
        counts[key] = counts.get(key, 0) + 1
    modal_key, modal_count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    modal_rank = [int(x) for x in modal_key.split(",")]

    def _mean_bool(key):
        return float(np.mean([bool(r.get(key, False)) for r in recs]))

    out = dict(
        available=True,
        R=int(R),
        dgp_type=recs[0].get("_dgp_type", "unknown"),
        Tp=int(recs[0].get("_Tp", 0)),
        N=int(recs[0].get("_N", 0)),
        q=int(recs[0].get("_q", 0)),
        r=int(recs[0].get("_r", 0)),
        J=int(recs[0].get("_J", 0)),
        J_realized=int(recs[0].get("_J_realized", recs[0].get("_J", 0))),
        J_manual_override=bool(recs[0].get("_J_manual_override", False)),
        J_min=int(recs[0].get("_J_min", 0)),
        c_J=float(recs[0].get("_c_J", float("nan"))),
        rank_selection_enabled=bool(recs[0].get("_rank_selection_enabled", False)),
        rank_candidate_caps=recs[0].get("_rank_candidate_caps"),
        tuning_fixed_ranks=recs[0].get("_tuning_fixed_ranks"),
        kappa_c=float(recs[0].get("_kappa_c", float("nan"))),
        selected_rank_counts=counts,
        modal_selected_rank=modal_rank,
        modal_selected_rank_count=int(modal_count),
        true_rank=[int(x) for x in true[0]],
        p_correct_rank=_mean_bool("_rank_exact_correct"),
        p_underfit=_mean_bool("_rank_underfit"),
        p_overfit=_mean_bool("_rank_overfit"),
        p_mixed_misspecification=_mean_bool("_rank_mixed_misspecification"),
        p_has_underfit_component=_mean_bool("_rank_has_underfit_component"),
        p_has_overfit_component=_mean_bool("_rank_has_overfit_component"),
    )
    return out


def studentized_sample(path, target, kind="white"):
    """Per-replication studentized statistic (err / s.e.) for one target -- the raw
    input for the QQ / histogram CLT figure.  ``kind`` is 'white' or 'xs'."""
    recs = [json.loads(l) for l in open(path)]
    key = "se" if kind == "white" else "se_xs"
    out = [r[target]["err"] / r[target][key]
           for r in recs if r[target].get(key, 0) and r[target][key] > 0]
    return np.array(out, float)


def print_table(agg, title=""):
    if title:
        print(title)
    head = f"{'target':16s} {'bias':>8s} {'rmse':>8s} {'mean_se':>8s} {'mc_sd':>8s} {'cov95':>6s}"
    print(head)
    for nm, r in agg.items():
        if nm.startswith("_"):
            continue
        print(f"{nm:16s} {r['bias']:8.4f} {r['rmse']:8.4f} {r['mean_se']:8.4f} "
              f"{r['mc_sd']:8.4f} {r['cov']:6.3f}")
