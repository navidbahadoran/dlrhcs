#!/usr/bin/env python3
"""Resume-safe batch runner for production Monte Carlo replications.

Examples
--------
Fixed-rank performance run:

    python scripts/run_mc_batches.py --dgp-type dgp1 --T 100 --N 100 \
      --R-total 1000 --batch-size 25 --out-path outputs/sim/grid_dgp1_100.jsonl \
      --select false --fixed-ranks 1,1,1 --n-jobs 4

Rank-selection run:

    python scripts/run_mc_batches.py --dgp-type dgp3 --T 100 --N 100 \
      --R-total 100 --batch-size 10 --out-path outputs/sim/grid_rank_dgp3_100.jsonl \
      --select true --true-ranks 1,1,1 --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlrhcs.mc import run_replication  # noqa: E402
from dlrhcs.pipeline import Tuning  # noqa: E402


def _parse_bool(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"1", "true", "t", "yes", "y"}:
        return True
    if key in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_ranks(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None or str(value).strip() == "":
        return None
    try:
        ranks = tuple(int(x.strip()) for x in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks:
        raise argparse.ArgumentTypeError("rank vector cannot be empty")
    if any(r < 0 for r in ranks):
        raise argparse.ArgumentTypeError("ranks must be nonnegative")
    return ranks


def _load_config(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def _tuning_from_config(cfg: Dict, *, select: bool,
                        fixed_ranks: Optional[Tuple[int, ...]],
                        rank_caps: Optional[Tuple[int, ...]]) -> Tuning:
    raw = dict(cfg.get("tuning", {}))
    if raw.get("ranks") is not None:
        raw["ranks"] = tuple(raw["ranks"])
    if raw.get("r_bar") is not None:
        raw["r_bar"] = tuple(raw["r_bar"])

    raw.setdefault("J_min", 10)
    raw.setdefault("c_J", 1.0)
    raw.setdefault("kappa_c", 0.015)

    if select:
        if fixed_ranks is not None:
            raise ValueError("--fixed-ranks fixes estimator ranks and cannot be used with --select true; use --true-ranks for rank-frequency truth")
        caps = rank_caps or raw.get("r_bar") or (1, 1, 1)
        raw["ranks"] = None
        raw["select"] = True
        raw["r_bar"] = tuple(caps)
    else:
        raw["ranks"] = tuple(fixed_ranks or raw.get("ranks") or (1, 1, 1))
        raw["select"] = False

    return Tuning(**raw)


def _done_reps(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add(int(rec["rep"]))
            except Exception:
                print(f"[warn] ignoring unreadable JSONL line {line_no} in {path}")
    return done


def _count_duplicate_reps(path: Path) -> int:
    counts: Dict[int, int] = {}
    if not path.exists():
        return 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rep = int(json.loads(line)["rep"])
            except Exception:
                continue
            counts[rep] = counts.get(rep, 0) + 1
    return int(sum(v - 1 for v in counts.values() if v > 1))


def _chunks(items: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _run_one_replication(T: int, N: int, rep: int, tuning: Tuning,
                         dgp_kwargs: Dict, master: int) -> Dict:
    return run_replication(T, N, rep, tuning, dgp_kwargs=dgp_kwargs, master=master)


def _format_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _emit(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _mode_label(args) -> str:
    return "rank-selection" if args.select else "fixed-rank"


def _progress_message(args, *, completed_total: int, current_rep,
                      elapsed: float, avg_seconds: float, remaining: int,
                      rec=None, prefix: str = "progress") -> str:
    eta = _format_eta(avg_seconds * remaining) if completed_total else "unknown"
    jval = rec.get("_J_realized", rec.get("_J")) if rec is not None else None
    retained = rec.get("_retained_nonvalidation") if rec is not None else None
    jtxt = f" J={jval}" if jval is not None else ""
    rtxt = f" retained_nonvalidation={retained:.4f}" if retained is not None else ""
    return (
        f"[mc-batch] {prefix} dgp={args.dgp_type} T={args.T} N={args.N} "
        f"mode={_mode_label(args)} completed={completed_total}/{args.R_total} "
        f"current_rep={current_rep} elapsed={_format_eta(elapsed)} "
        f"avg_sec_per_rep={avg_seconds:.1f} eta={eta}{jtxt}{rtxt}"
    )


def _sidecar_path(out_path: Path) -> Path:
    suffix = "".join(out_path.suffixes)
    if suffix:
        stem = str(out_path)[:-len(suffix)]
        return Path(stem + ".meta.json")
    return Path(str(out_path) + ".meta.json")


def _write_sidecar(path: Path, *, args, tuning: Tuning, completed: int,
                   duplicate_reps: int, started_at: str) -> None:
    meta = {
        "dgp_type": args.dgp_type,
        "T": int(args.T),
        "N": int(args.N),
        "R_total": int(args.R_total),
        "completed_R": int(completed),
        "J_min": int(tuning.J_min),
        "kappa_c": float(tuning.kappa_c),
        "c_xi_calibration_draws": int(args.c_xi_calibration_draws),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "config": args.config,
        "out_path": str(args.out_path),
        "batch_size": int(args.batch_size),
        "n_jobs": int(args.n_jobs),
        "progress_every": int(args.progress_every),
        "quiet": bool(args.quiet),
        "start_rep": int(args.start_rep),
        "select": bool(args.select),
        "fixed_ranks": list(args.fixed_ranks) if args.fixed_ranks is not None else None,
        "true_ranks": list(args.true_ranks) if args.true_ranks is not None else None,
        "rank_caps": list(args.rank_caps) if args.rank_caps is not None else None,
        "duplicate_reps": int(duplicate_reps),
    }
    with path.open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume-safe Monte Carlo batch runner")
    ap.add_argument("--dgp-type", required=True, choices=["dgp1", "dgp2", "dgp3"])
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--N", required=True, type=int)
    ap.add_argument("--R-total", required=True, type=int)
    ap.add_argument("--batch-size", required=True, type=int)
    ap.add_argument("--start-rep", type=int, default=0)
    ap.add_argument("--out-path", required=True, type=Path)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--select", required=True, type=_parse_bool)
    ap.add_argument("--fixed-ranks", type=_parse_ranks, default=None)
    ap.add_argument("--true-ranks", type=_parse_ranks, default=None,
                    help="true DGP ranks used for rank-frequency diagnostics; does not fix estimator ranks")
    ap.add_argument("--rank-caps", type=_parse_ranks, default=None,
                    help="candidate rank caps for --select true; defaults to config r_bar or 1,1,1")
    ap.add_argument("--c-xi-calibration-draws", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel worker processes for replications within each batch")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="print progress after this many completed replications")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress output; JSONL and metadata are still written")
    args = ap.parse_args()

    if args.R_total < 1:
        raise SystemExit("--R-total must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.start_rep < 0 or args.start_rep >= args.R_total:
        raise SystemExit("--start-rep must satisfy 0 <= start_rep < R_total")
    if args.c_xi_calibration_draws < 1:
        raise SystemExit("--c-xi-calibration-draws must be positive")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must be nonzero; use 1 for serial or -1 for all available cores")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    cfg = _load_config(args.config)
    try:
        tuning = _tuning_from_config(cfg, select=args.select,
                                     fixed_ranks=args.fixed_ranks,
                                     rank_caps=args.rank_caps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dgp_kwargs = dict(cfg.get("dgp", {}))
    dgp_kwargs.update({
        "dgp_type": args.dgp_type,
        "c_xi_calibration_draws": int(args.c_xi_calibration_draws),
    })
    if args.true_ranks is not None:
        dgp_kwargs["true_ranks"] = tuple(args.true_ranks)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(args.out_path)
    master = int(cfg.get("master_seed", 2024))
    desired = list(range(int(args.start_rep), int(args.R_total)))
    done = _done_reps(args.out_path)
    todo = [rep for rep in desired if rep not in done]
    started_at = datetime.now(timezone.utc).isoformat()
    duplicate_reps = _count_duplicate_reps(args.out_path)
    already_completed = len(done.intersection(desired))
    run_complete = already_completed >= len(desired)

    _emit(f"[mc-batch] dgp={args.dgp_type} T={args.T} N={args.N} "
          f"R_total={args.R_total} mode={_mode_label(args)}", quiet=args.quiet)
    _emit(f"[mc-batch] out={args.out_path}", quiet=args.quiet)
    _emit(f"[mc-batch] already_completed={already_completed} remaining={len(todo)} "
          f"duplicates={duplicate_reps} complete={run_complete}", quiet=args.quiet)

    completed_now = 0
    t0 = time.time()
    with args.out_path.open("a", buffering=1) as fh:
        for batch_no, batch in enumerate(_chunks(todo, int(args.batch_size)), start=1):
            completed_at_batch_start = len(done.intersection(desired))
            workers = int(args.n_jobs)
            _emit(f"[mc-batch] batch {batch_no}: reps {batch[0]}..{batch[-1]} "
                  f"workers={workers} completed={completed_at_batch_start}/{args.R_total}",
                  quiet=args.quiet)
            batch_start = time.time()
            last_rec = None
            if int(args.n_jobs) == 1:
                recs = []
                for rep in batch:
                    _emit(f"[mc-batch] starting dgp={args.dgp_type} T={args.T} N={args.N} "
                          f"mode={_mode_label(args)} rep={rep}",
                          quiet=args.quiet)
                    recs.append(_run_one_replication(args.T, args.N, rep, tuning,
                                                     dgp_kwargs, master))
            else:
                from joblib import Parallel, delayed
                recs = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
                    delayed(_run_one_replication)(args.T, args.N, rep, tuning,
                                                  dgp_kwargs, master)
                    for rep in batch
                )

            for rec in sorted(recs, key=lambda row: int(row["rep"])):
                rep = int(rec["rep"])
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                done.add(rep)
                completed_now += 1
                last_rec = rec
                elapsed = time.time() - t0
                avg = elapsed / max(completed_now, 1)
                remaining_total = max(int(args.R_total) - len(done.intersection(desired)), 0)
                completed_total = len(done.intersection(desired))
                if completed_now % int(args.progress_every) == 0:
                    _emit(_progress_message(args, completed_total=completed_total,
                                            current_rep=rep, elapsed=elapsed,
                                            avg_seconds=avg,
                                            remaining=remaining_total, rec=rec),
                          quiet=args.quiet)
            duplicate_reps = _count_duplicate_reps(args.out_path)
            completed_total = len(_done_reps(args.out_path).intersection(desired))
            _write_sidecar(sidecar, args=args, tuning=tuning,
                           completed=completed_total,
                           duplicate_reps=duplicate_reps,
                           started_at=started_at)
            elapsed = time.time() - t0
            avg = elapsed / max(completed_now, 1)
            remaining_total = max(int(args.R_total) - completed_total, 0)
            batch_elapsed = time.time() - batch_start
            batch_avg = batch_elapsed / max(len(batch), 1)
            batch_eta = _format_eta(avg * remaining_total) if completed_now else "unknown"
            _emit(f"[mc-batch] batch {batch_no} summary elapsed={_format_eta(batch_elapsed)} "
                  f"avg_sec_per_rep={batch_avg:.1f} eta={batch_eta}",
                  quiet=args.quiet)
            _emit(_progress_message(args, completed_total=completed_total,
                                    current_rep=batch[-1], elapsed=elapsed,
                                    avg_seconds=avg,
                                    remaining=remaining_total, rec=last_rec,
                                    prefix="batch-complete"),
                  quiet=args.quiet)
            _emit(f"[mc-batch] sidecar updated: {sidecar}", quiet=args.quiet)

    duplicate_reps = _count_duplicate_reps(args.out_path)
    completed_total = len(_done_reps(args.out_path).intersection(desired))
    _write_sidecar(sidecar, args=args, tuning=tuning,
                   completed=completed_total,
                   duplicate_reps=duplicate_reps,
                   started_at=started_at)
    elapsed = time.time() - t0
    avg = elapsed / max(completed_now, 1) if completed_now else 0.0
    _emit(f"[mc-batch] done completed_R={completed_total}/{args.R_total} "
          f"elapsed={_format_eta(elapsed)} avg_sec_per_rep={avg:.1f} "
          f"duplicates={duplicate_reps} jsonl={args.out_path} metadata={sidecar}",
          quiet=args.quiet)


if __name__ == "__main__":
    main()
