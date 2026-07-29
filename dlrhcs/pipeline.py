"""
Full feasible pipeline (spec sec 11).

    estimate(Y, Z_list, P, K, targets, tuning) -> estimates, ses, intervals, diag

Steps: build designs; (optionally) run the roadmap for q/J/box/kappa; build the
scattered purged folds; (optionally) select ranks by the cross-fitted criterion;
refit Theta_hat^0_{-j} on each purged fold and form residuals; solve the feasible
Riesz weights per target; one-step debias; studentize (White + xs); intervals.

Set ``oracle=True`` and pass ``true_U/true_V`` to run the infeasible oracle
benchmark (true tangent spaces in the Riesz solve) -- the spec sec 12 checkpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .design import A, build_blocks
from .factorridge import fit_factor_ridge
from .folds import make_folds
from .onestep import FoldFit, OneStepResult, one_step, white_se, xs_se
from .ranks import rank_penalty, roadmap, select_ranks, _pmap
from .targets import Target


@dataclass
class Tuning:
    ranks: Optional[tuple] = None       # fixed ranks; else select / roadmap
    q: Optional[int] = None
    J: Optional[int] = None             # fixed fold count override
    J_override: Optional[int] = None    # explicit fixed-J alias for debugging/configs
    J_min: int = 10                     # finite-sample floor for rule-chosen J
    c_J: float = 1.0                    # constant in ceil(c_J * B_TN * L_TN^J)
    ridge: float = 0.02
    n_sweeps: int = 80
    n_restarts: int = 4
    tol: float = 1e-8
    scheme: str = "scatter"
    select: bool = False                # run rank selection over roadmap box
    use_roadmap: bool = False           # derive q/J/box/kappa from data
    kappa_c: float = 1.0
    alpha_level: float = 0.05
    riesz_ridge: float = 1e-8
    riesz_tol: float = 1e-10
    riesz_maxiter: int = 2000
    use_riesz_cache: bool = True          # reuse fold-level Riesz setup across targets
    riesz_use_cached_scale: bool = False  # opt-in: common fold scale instead of RHS scale
    xs_bandwidth: Optional[int] = None   # spatial-kernel xs s.e. bandwidth (None=auto)
    xs_kernel: str = "bartlett"          # "bartlett" (spatial, sim) | "cluster" (empirical)
    n_jobs: int = 1                      # cores for rank selection (single-panel use)
    buffer_r: int = 0                    # spatial fold-buffer radius r_TN (0 = time-only)
    r_bar: Optional[tuple] = None        # fixed rank caps for the candidate box


@dataclass
class EstimateResult:
    estimates: Dict[str, float]
    se: Dict[str, float]
    se_xs: Dict[str, float]
    ci: Dict[str, tuple]
    ci_xs: Dict[str, tuple]
    ranks: tuple
    q: int
    J: int
    onestep: OneStepResult
    diagnostics: Dict = field(default_factory=dict)


def _resolve_fold_count(Tp: int, N: int, q: int, r: int, tuning: Tuning):
    """Choose J by explicit override or the finite-fold-floor rule."""
    if tuning.J is not None and tuning.J_override is not None and int(tuning.J) != int(tuning.J_override):
        raise ValueError(f"conflicting J={tuning.J!r} and J_override={tuning.J_override!r}")
    explicit_J = tuning.J_override if tuning.J_override is not None else tuning.J
    TpN = int(Tp * N)
    B_TN = int((int(q) + 1) * (2 * int(r) + 1))
    n_eff = float(Tp * N) / float(Tp + N)
    L_TN_J = float(max(1.0, np.log(np.log(max(n_eff, np.exp(np.exp(1.0)))))))
    J_rule_term = int(np.ceil(float(tuning.c_J) * B_TN * L_TN_J))
    manual = explicit_J is not None
    J = int(explicit_J) if manual else max(int(tuning.J_min), J_rule_term, 2)
    if J < 2 or J > TpN:
        source = "manual override" if manual else "finite-fold-floor rule"
        raise ValueError(f"{source} produced invalid J={J}; expected 2 <= J <= T*N={TpN}")
    diag = dict(J_realized=J,
                J_min=int(tuning.J_min),
                c_J=float(tuning.c_J),
                J_rule_term=J_rule_term,
                J_manual_override=bool(manual),
                B_TN_fold_rule=B_TN,
                L_TN_J=L_TN_J)
    return J, diag


def _fit_diagnostics(fit):
    return dict(
        warm_start_objective=float(fit.warm_start_objective),
        restart_objectives=[float(x) for x in fit.restart_objs],
        random_restart_objectives=[float(x) for x in fit.random_restart_objs],
        best_objective=float(fit.best_objective),
        n_sweeps=int(fit.n_sweeps),
        stopped_before_sweep_cap=bool(fit.stopped_before_sweep_cap),
        converged=bool(fit.converged),
        max_iteration_hit=bool(fit.max_iteration_hit),
        monotone=bool(fit.monotone),
        final_relative_objective_decrease=float(fit.final_relative_objective_decrease),
        stationarity_residual=float(fit.stationarity_residual),
        relative_restart_improvement=float(fit.relative_restart_improvement),
        restart_improvement_gt_1e_6=bool(fit.restart_improvement_gt_1e_6),
        restart_improvement_gt_1e_4=bool(fit.restart_improvement_gt_1e_4),
        restart_diagnostics=getattr(fit, "restart_diagnostics", []),
        selected_restart_label=getattr(fit, "selected_restart_label", ""),
        selected_restart_index=int(getattr(fit, "selected_restart_index", 0)),
        total_sweeps_from_initialization=int(getattr(fit, "total_sweeps_from_initialization", fit.n_sweeps)),
        final_level_sweeps=int(getattr(fit, "final_level_sweeps", fit.n_sweeps)),
        convergence_sweep=getattr(fit, "convergence_sweep", None),
        stopping_reason=getattr(fit, "stopping_reason", ""),
    )


def _summarize_fit_diagnostics(fits):
    if not fits:
        return dict(
            first_stage_fit_diagnostics=[],
            first_stage_warm_start_objective_mean=float("nan"),
            first_stage_best_objective_mean=float("nan"),
            first_stage_n_sweeps_mean=float("nan"),
            first_stage_final_relative_objective_decrease_mean=float("nan"),
            first_stage_final_decrease_lt_1e_6_rate=float("nan"),
            first_stage_final_decrease_lt_1e_5_rate=float("nan"),
            first_stage_stationarity_residual_mean=float("nan"),
            first_stage_relative_restart_improvement_mean=float("nan"),
            first_stage_restart_improvement_gt_1e_6_rate=float("nan"),
            first_stage_restart_improvement_gt_1e_4_rate=float("nan"),
            first_stage_monotone_rate=float("nan"),
            first_stage_stopped_before_sweep_cap_rate=float("nan"),
            first_stage_sweep_cap_hit_rate=float("nan"),
            first_stage_convergence_failure_rate=float("nan"),
            first_stage_max_iteration_hit_rate=float("nan"),
        )
    rows = [_fit_diagnostics(fit) for fit in fits]

    def mean(key):
        vals = np.array([row[key] for row in rows], dtype=float)
        return float(np.mean(vals))

    return dict(
        first_stage_fit_diagnostics=rows,
        first_stage_warm_start_objective_mean=mean("warm_start_objective"),
        first_stage_best_objective_mean=mean("best_objective"),
        first_stage_n_sweeps_mean=mean("n_sweeps"),
        first_stage_final_relative_objective_decrease_mean=mean("final_relative_objective_decrease"),
        first_stage_final_decrease_lt_1e_6_rate=float(np.mean([
            row["final_relative_objective_decrease"] < 1e-6 for row in rows])),
        first_stage_final_decrease_lt_1e_5_rate=float(np.mean([
            row["final_relative_objective_decrease"] < 1e-5 for row in rows])),
        first_stage_stationarity_residual_mean=mean("stationarity_residual"),
        first_stage_relative_restart_improvement_mean=mean("relative_restart_improvement"),
        first_stage_restart_improvement_gt_1e_6_rate=mean("restart_improvement_gt_1e_6"),
        first_stage_restart_improvement_gt_1e_4_rate=mean("restart_improvement_gt_1e_4"),
        first_stage_monotone_rate=mean("monotone"),
        first_stage_stopped_before_sweep_cap_rate=mean("stopped_before_sweep_cap"),
        first_stage_sweep_cap_hit_rate=mean("max_iteration_hit"),
        first_stage_convergence_failure_rate=float(1.0 - mean("converged")),
        first_stage_max_iteration_hit_rate=mean("max_iteration_hit"),
    )


def estimate(Y, Z_list, targets: Sequence[Target], tuning: Tuning,
             P=1, rng=None, foldid=None,
             oracle=False, true_U=None, true_V=None,
             profile_timing: bool = False,
             first_stage_seed: Optional[int] = None,
             first_stage_model_id: str = "pipeline") -> EstimateResult:
    timing = {} if profile_timing else None
    if rng is None:
        rng = np.random.default_rng(0)
    Y = np.asarray(Y, dtype=float)
    Tp, N = Y.shape
    blocks = build_blocks(Z_list)
    B = len(blocks)

    fit_kwargs = dict(ridge=tuning.ridge, n_sweeps=tuning.n_sweeps,
                      tol=tuning.tol, n_restarts=tuning.n_restarts, rng=rng)

    # ---- q, J, ranks, kappa --------------------------------------------------
    ranks, q, kappa, candidates = tuning.ranks, tuning.q, None, None
    rank_table = None
    rm = None
    rank_t0 = time.perf_counter() if profile_timing else None
    if tuning.use_roadmap or tuning.select:
        rm = roadmap(Y, Z_list, P=P, r_bar=tuning.r_bar,
                     kappa_c=tuning.kappa_c, fit_kwargs=fit_kwargs,
                     r_buffer=tuning.buffer_r)
        q = q if q is not None else rm.q
        kappa, candidates = rm.kappa, rm.candidates
    if q is None:
        q = 3
    J, J_diag = _resolve_fold_count(Tp, N, q, tuning.buffer_r, tuning)
    if rm is not None:
        kappa = rank_penalty(rm.sigma2_hat, Tp, N, J, tuning.kappa_c)

    folds = make_folds(Tp, N, J, q, r=tuning.buffer_r, P=P, rng=rng,
                       scheme=tuning.scheme, foldid=foldid)

    if ranks is None:
        if tuning.select and candidates:
            ranks, rank_table = select_ranks(Y, blocks, candidates, folds, kappa,
                                             fit_kwargs, n_jobs=tuning.n_jobs)
        else:
            ranks = tuple([1] * B)
    if profile_timing:
        timing["rank_selection_sec"] = (
            float(time.perf_counter() - rank_t0)
            if (tuning.use_roadmap or tuning.select) else 0.0
        )

    # ---- per-fold purged fits ------------------------------------------------
    # The J fold first-stage fits are independent.  Two paths, by design:
    #   * n_jobs == 1 (the default, used inside the rep-parallel Monte Carlo):
    #     SERIAL with the shared rng -- byte-identical to prior runs, and avoids
    #     nesting parallelism under the already-parallel replication loop.
    #   * n_jobs != 1 (single-panel use, e.g. the empirical): PARALLEL over folds
    #     (joblib loky), each fold seeded from (base, fold) so the result is
    #     reproducible and independent of the core count.
    def _make_foldfit(fd, fit):
        resid = Y - A(fit.surfaces, blocks)
        U, V = (true_U, true_V) if oracle else (fit.U, fit.V)
        return FoldFit(surfaces=fit.surfaces, U=U, V=V, residual=resid,
                       train=fd.train, val=fd.val, p=fd.p, alpha=fd.alpha)
    first_stage_t0 = time.perf_counter() if profile_timing else None
    if tuning.n_jobs and tuning.n_jobs != 1:
        base = fit_kwargs.get("rng", None)
        seed0 = int(base.integers(2 ** 31)) if base is not None else 0
        fk = {k: v for k, v in fit_kwargs.items() if k != "rng"}

        def _fit_fold(fi):
            rng_f = np.random.default_rng(np.random.SeedSequence([seed0, 7919, fi]))
            stable_kwargs = {}
            if first_stage_seed is not None:
                stable_kwargs = dict(global_seed=int(first_stage_seed),
                                     fold_id=int(fi),
                                     model_id=str(first_stage_model_id))
            return fit_factor_ridge(Y, blocks, ranks, mask=folds[fi].train,
                                    rng=rng_f, **fk, **stable_kwargs)
        fits = _pmap(_fit_fold, range(len(folds)), tuning.n_jobs)
        foldfits = [_make_foldfit(folds[fi], fits[fi]) for fi in range(len(folds))]
        mono_ok = all(f.monotone for f in fits)
    else:
        foldfits, fits, mono_ok = [], [], True
        for fi, fd in enumerate(folds):
            stable_kwargs = {}
            if first_stage_seed is not None:
                stable_kwargs = dict(global_seed=int(first_stage_seed),
                                     fold_id=int(fi),
                                     model_id=str(first_stage_model_id))
            fit = fit_factor_ridge(Y, blocks, ranks, mask=fd.train, **fit_kwargs,
                                   **stable_kwargs)
            fits.append(fit)
            mono_ok = mono_ok and fit.monotone
            foldfits.append(_make_foldfit(fd, fit))
    if profile_timing:
        timing["first_stage_sec"] = float(time.perf_counter() - first_stage_t0)

    # ---- one-step + variances ------------------------------------------------
    onestep_t0 = time.perf_counter() if profile_timing else None
    res = one_step(blocks, foldfits, targets,
                   riesz_kwargs=dict(ridge=tuning.riesz_ridge,
                                     tol=tuning.riesz_tol,
                                     maxiter=tuning.riesz_maxiter,
                                     use_cached_scale=tuning.riesz_use_cached_scale),
                   profile_timing=profile_timing,
                   use_riesz_cache=tuning.use_riesz_cache)
    if profile_timing:
        timing["onestep_sec"] = float(time.perf_counter() - onestep_t0)
        timing["riesz_sec"] = float(res.timing.get("riesz_sec", float("nan")))
    z = norm.ppf(1 - tuning.alpha_level / 2)
    se, se_xs, ci, ci_xs = {}, {}, {}, {}
    se_t0 = time.perf_counter() if profile_timing else None
    for tg in targets:
        s = white_se(res, tg.name)
        sx = xs_se(res, tg.name, bandwidth=tuning.xs_bandwidth, kernel=tuning.xs_kernel)
        e = res.estimates[tg.name]
        se[tg.name], se_xs[tg.name] = s, sx
        ci[tg.name] = (e - z * s, e + z * s)
        ci_xs[tg.name] = (e - z * sx, e + z * sx)
    if profile_timing:
        timing["se_sec"] = float(time.perf_counter() - se_t0)

    TpN = float(Tp * N)
    train_counts = np.array([fd.n_pur for fd in folds], dtype=float)
    val_counts = np.array([fd.val.sum() for fd in folds], dtype=float)
    retained_total_by_fold = train_counts / TpN
    nonvalidation_counts = np.maximum(TpN - val_counts, 1.0)
    retained_nonvalidation_by_fold = train_counts / nonvalidation_counts
    retained_total = float(np.mean(retained_total_by_fold))
    retained_nonvalidation = float(np.mean(retained_nonvalidation_by_fold))
    diag = dict(monotone=mono_ok, ranks=ranks, q=q, J=J,
                retained=retained_nonvalidation,
                retained_total=retained_total,
                retained_nonvalidation=retained_nonvalidation,
                retained_total_by_fold=retained_total_by_fold.tolist(),
                retained_nonvalidation_by_fold=retained_nonvalidation_by_fold.tolist(),
                retained_nonvalidation_min=float(np.min(retained_nonvalidation_by_fold)),
                retained_nonvalidation_max=float(np.max(retained_nonvalidation_by_fold)),
                validation_fold_size_mean=float(np.mean(val_counts)),
                validation_fold_share_mean=float(np.mean(val_counts / TpN)))
    diag.update(_summarize_fit_diagnostics(fits))
    diag.update(J_diag)
    if profile_timing:
        diag["timing"] = timing
    if rank_table is not None:
        diag["rank_table"] = [(list(r), float(L), float(d), float(crit))
                              for (r, L, d, crit) in rank_table]
        diag["rank_selection_fit_diagnostics_available"] = False
        diag["rank_selection_fit_diagnostics_todo"] = (
            "Rank-selection candidate fits are not instrumented in the MC JSONL; "
            "final selected per-fold fits are summarized in first_stage_* diagnostics."
        )
    return EstimateResult(estimates=res.estimates, se=se, se_xs=se_xs,
                          ci=ci, ci_xs=ci_xs, ranks=tuple(ranks), q=q, J=J,
                          onestep=res, diagnostics=diag)
