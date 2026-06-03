#!/usr/bin/env python3
"""Run the ABM under four climate forcing scenarios and collect results.

This is a targeted robustness check — NOT a full sensitivity analysis.
It answers the reviewer's question: does the ~30-year emergent window
survive when climate has persistence, trend, or both?

Scenarios (empirically grounded parameters):
  1. iid            — null model, no temporal structure
  2. ar1 (φ=0.50)  — persistence at tree-ring reconstruction upper bound
  3. trend+noise    — linear trend β = 0.01 σ/yr (1σ per century)
  4. ar1+trend      — persistence + trend combined (most realistic)

Usage:
  # Local (small repeats for testing):
  uv run python scripts/climate_scenarios.py --repeats 10 --n-workers 2

  # Cluster (full):
  uv run python scripts/climate_scenarios.py --repeats 100 --n-workers 8

  # Only one baseline:
  uv run python scripts/climate_scenarios.py --baselines personal

  # Resume into existing directory:
  uv run python scripts/climate_scenarios.py --output-dir reports/results/climate_scenarios/20260603-...
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Four scenarios with empirically grounded parameters.
# φ=0.50 is from the tree-ring reconstruction (upper bound);
# instrumental data φ ≈ −0.10 ≈ 0, so iid is the empirical baseline.
# Trend β=0.01 σ/yr = 1σ drift per century (moderate-to-strong).
SCENARIOS: dict[str, dict] = {
    "iid": {
        "climate_process": "iid",
        "climate_phi": 0.0,
        "climate_trend": 0.0,
    },
    "ar1": {
        "climate_process": "ar1",
        "climate_phi": 0.50,
        "climate_trend": 0.0,
    },
    "trend": {
        "climate_process": "trend_plus_noise",
        "climate_phi": 0.0,
        "climate_trend": 0.01,
    },
    "ar1_trend": {
        "climate_process": "ar1_trend",
        "climate_phi": 0.50,
        "climate_trend": 0.01,
    },
}


def run_scenario(
    scenario_name: str,
    scenario_params: dict,
    *,
    baseline: str,
    output_dir: Path,
    repeats: int,
    years: int,
    num_process: int,
    timeout: float | None = None,
) -> Path:
    """Run one ABM scenario and return the run directory."""
    run_dir = output_dir / f"{scenario_name}_{baseline}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if correlations.csv already exists (resume support)
    corr_path = run_dir / "correlations.csv"
    if corr_path.exists():
        print(f"  [skip] {scenario_name}/{baseline} — already done")
        return run_dir

    overrides = [
        f"model.repeats={repeats}",
        f"model.num_process={num_process}",
        f"model.years={years}",
        f"model.memory_baseline={baseline}",
        f"model.climate_process={scenario_params['climate_process']}",
        f"model.climate_phi={scenario_params['climate_phi']}",
        f"model.climate_trend={scenario_params['climate_trend']}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]

    cmd = ["uv", "run", "python", "shifting_baseline/abm.py", *overrides]
    env = {**__import__("os").environ, "NO_NOTIFY": "1"}

    print(f"  [{scenario_name}/{baseline}] running...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            timeout=timeout,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.0f}s)")
    except subprocess.CalledProcessError as e:
        print(f"FAILED\n  stderr: {e.stderr[:500]}")
        raise
    return run_dir


def collect_results(output_dir: Path) -> dict:
    """Read all correlations.csv and extract peak statistics."""
    import numpy as np
    import pandas as pd

    results = []
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        corr_path = subdir / "correlations.csv"
        if not corr_path.exists():
            continue
        parts = subdir.name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        scenario, baseline = parts

        corr = pd.read_csv(corr_path, index_col=0)
        numeric = corr.apply(pd.to_numeric, errors="coerce")

        peak_windows = numeric.idxmax(axis=0).astype(float)
        peak_strengths = numeric.max(axis=0).astype(float)

        results.append(
            {
                "scenario": scenario,
                "baseline": baseline,
                "peak_window_mean": peak_windows.mean(),
                "peak_window_std": peak_windows.std(ddof=1),
                "peak_window_median": peak_windows.median(),
                "peak_strength_mean": peak_strengths.mean(),
                "peak_strength_std": peak_strengths.std(ddof=1),
                "n_replicates": int(peak_strengths.notna().sum()),
            }
        )

    df = pd.DataFrame(results)
    summary_path = output_dir / "scenario_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    return {"summary": df, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Run ABM under four climate forcing scenarios"
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--num-process", type=int, default=1)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Concurrent scenario runs (not used yet; scenarios run sequentially)",
    )
    parser.add_argument("--baselines", nargs="+", default=["personal", "collective"])
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Resume into existing directory"
    )
    parser.add_argument(
        "--output-root", type=str, default="reports/results/climate_scenarios"
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_root) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config = {
        "scenarios": SCENARIOS,
        "baselines": args.baselines,
        "repeats": args.repeats,
        "years": args.years,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Output: {output_dir}")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Baselines: {args.baselines}")
    print(f"Repeats: {args.repeats}, Years: {args.years}\n")

    for baseline in args.baselines:
        print(f"=== Baseline: {baseline} ===")
        for name, params in SCENARIOS.items():
            run_scenario(
                name,
                params,
                baseline=baseline,
                output_dir=output_dir,
                repeats=args.repeats,
                years=args.years,
                num_process=args.num_process,
                timeout=args.timeout,
            )
        print()

    results = collect_results(output_dir)
    print("\n" + results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
