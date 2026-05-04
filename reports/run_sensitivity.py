#!/usr/bin/env python3
"""Run global sensitivity analysis stages for the climate-observing ABM.

Stages:
- ``smoke``     1 ABM call with tiny ``repeats``/``years``; sanity check the plumbing.
- ``benchmark`` n_samples random points with full ``repeats``; measure wall-clock.
- ``morris``    Morris elementary effects screening.
- ``sobol``     Sobol indices on (a subset of) parameters.

Each stage writes to ``<output-root>/<timestamp>-<stage>-<baseline>/``.

Examples
--------
Local smoke test::

    uv run python reports/run_sensitivity.py smoke

Cluster Morris (one baseline)::

    uv run python reports/run_sensitivity.py morris \
        --memory-baseline personal --n-workers 32 --repeats 30

Cluster Sobol::

    uv run python reports/run_sensitivity.py sobol \
        --memory-baseline collective --N 1024 --n-workers 32 --repeats 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shifting_baseline.sensitivity import (  # noqa: E402
    PARAM_NAMES,
    benchmark_runtime,
    run_morris,
    run_one,
    run_sobol,
)


def _make_run_dir(output_root: Path, stage: str, baseline: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{timestamp}-{stage}-{baseline}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_run_dir(args: argparse.Namespace, stage: str) -> Path:
    """Use ``--output-dir`` verbatim if given (resume mode), else mint a new
    timestamped directory under ``--output-root``.
    """
    if getattr(args, "output_dir", None):
        run_dir = Path(args.output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return _make_run_dir(args.output_root, stage, args.memory_baseline)


def cmd_smoke(args: argparse.Namespace) -> int:
    """One ABM call with tiny knobs; verifies subprocess + parsing path."""
    run_dir = _resolve_run_dir(args, "smoke")
    print(f"[smoke] writing to {run_dir}")
    start = time.perf_counter()
    # Use defaults near the manuscript point; values are valid SA samples.
    sample = [40.0, 5.0, 0.4, 1.0, 0.5]  # max_age, new_agents, loss_rate, sigma, phi
    result = run_one(
        sample,
        memory_baseline=args.memory_baseline,
        run_dir=run_dir / "single_run",
        param_names=PARAM_NAMES,
        repeats=args.repeats,
        years=args.years,
        num_process=args.num_process,
        climate_process=args.climate_process,
        cleanup=False,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - start
    payload = {
        "params": result.params,
        "metrics": result.metrics,
        "run_dir": str(result.run_dir),
        "wall_clock_seconds": result.elapsed_seconds,
        "stage_total_seconds": elapsed,
    }
    out = run_dir / "smoke_result.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    print(f"[smoke] OK in {elapsed:.1f}s — see {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args, "benchmark")
    print(f"[benchmark] writing to {run_dir}")
    df = benchmark_runtime(
        run_dir,
        memory_baseline=args.memory_baseline,
        repeats=args.repeats,
        years=args.years,
        num_process=args.num_process,
        climate_process=args.climate_process,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(df.to_string(index=False))
    print(
        f"\n[benchmark] median = {df['elapsed_seconds'].median():.1f}s, "
        f"mean = {df['elapsed_seconds'].mean():.1f}s, "
        f"max = {df['elapsed_seconds'].max():.1f}s"
    )
    return 0


def cmd_morris(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args, "morris")
    print(
        f"[morris] r={args.r_trajectories} levels={args.num_levels} "
        f"workers={args.n_workers} timeout={args.timeout}s → {run_dir}"
    )
    run_morris(
        run_dir,
        memory_baseline=args.memory_baseline,
        r_trajectories=args.r_trajectories,
        num_levels=args.num_levels,
        repeats=args.repeats,
        years=args.years,
        num_process=args.num_process,
        climate_process=args.climate_process,
        n_workers=args.n_workers,
        keep_run_dirs=args.keep_run_dirs,
        seed=args.seed,
        timeout=args.timeout,
    )
    print(f"[morris] done — see {run_dir}")
    return 0


def cmd_sobol(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args, "sobol")
    print(
        f"[sobol] N={args.N} workers={args.n_workers} timeout={args.timeout}s → {run_dir}"
    )
    run_sobol(
        run_dir,
        memory_baseline=args.memory_baseline,
        N=args.N,
        repeats=args.repeats,
        years=args.years,
        num_process=args.num_process,
        climate_process=args.climate_process,
        n_workers=args.n_workers,
        keep_run_dirs=args.keep_run_dirs,
        seed=args.seed,
        calc_second_order=args.second_order,
        timeout=args.timeout,
    )
    print(f"[sobol] done — see {run_dir}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "reports/results/sensitivity",
    )
    parser.add_argument(
        "--memory-baseline",
        choices=["personal", "collective", "model"],
        default="personal",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument(
        "--num-process",
        type=int,
        default=1,
        help="ABM-internal replicate processes per subprocess.",
    )
    parser.add_argument(
        "--climate-process", choices=["iid", "ar1", "trend_plus_noise"], default="ar1"
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Concurrent ABM subprocess workers (cluster: SLURM_CPUS_PER_TASK).",
    )
    parser.add_argument(
        "--keep-run-dirs",
        action="store_true",
        help="Retain per-sample Hydra dirs (default: cleanup after parsing).",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Per-subprocess timeout in seconds."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Resume into this exact existing directory (must already contain "
            "the stage's samples.csv). Skips creating a new timestamped dir; "
            "completed sample_idx values in raw_outputs.csv are reused."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_smoke = sub.add_parser("smoke", help="One ABM call to verify plumbing.")
    _add_common(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke, repeats=2, years=20)

    p_bench = sub.add_parser("benchmark", help="Measure wall-clock per ABM batch.")
    _add_common(p_bench)
    p_bench.add_argument("--n-samples", type=int, default=5)
    p_bench.set_defaults(func=cmd_benchmark)

    p_morris = sub.add_parser("morris", help="Morris elementary effects.")
    _add_common(p_morris)
    p_morris.add_argument("--r-trajectories", type=int, default=20)
    p_morris.add_argument("--num-levels", type=int, default=4)
    p_morris.set_defaults(func=cmd_morris)

    p_sobol = sub.add_parser("sobol", help="Sobol indices via Saltelli sampling.")
    _add_common(p_sobol)
    p_sobol.add_argument(
        "--N",
        type=int,
        default=1024,
        help="Saltelli base sample size; total = N*(2k+2) "
        "or N*(2k+2) for first-order only.",
    )
    p_sobol.add_argument(
        "--second-order",
        action="store_true",
        help="Compute second-order interactions (more samples).",
    )
    p_sobol.set_defaults(func=cmd_sobol)

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
