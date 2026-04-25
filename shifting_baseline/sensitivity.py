"""Global sensitivity analysis for the climate-observing ABM.

Wraps the existing `shifting_baseline.abm` Hydra entrypoint as a black-box
response function so SALib (Morris / Sobol) can sample its parameters.

The unit of evaluation is one ABM *batch* (`Experiment.batch_run` over
`repeats` ABM replicates) at a fixed parameter vector. From the batch's
`correlations.csv` we extract two scalar response metrics per replicate
(peak-correlation window size, peak-correlation magnitude) and average
across replicates. SALib analyses these aggregated scalars.

See `reports/run_sensitivity.py` for the CLI orchestrator and
`scripts/sensitivity.slurm` for cluster execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

REPO_ROOT = Path(__file__).resolve().parents[1]

PARAM_NAMES: tuple[str, ...] = (
    "max_age",
    "new_agents",
    "loss_rate",
    "climate_sigma",
    "climate_phi",
)

# Note on runtime: ABM wall-clock scales roughly with `max_age * new_agents`
# (steady-state agent count → archive size → O(archive^2) in collective-baseline
# memory updates). At the upper corner (max_age=80, new_agents=15, loss_rate≈0)
# a single batch (`repeats=30, years=100`) can take ~hours. Use the per-call
# `--timeout` flag in the CLI to cap stragglers.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "max_age": (15.0, 80.0),
    "new_agents": (1.0, 15.0),
    "loss_rate": (0.0, 0.8),
    "climate_sigma": (0.5, 2.0),
    "climate_phi": (0.0, 0.9),
}

# Params that must be cast to int when written into the Hydra override.
PARAM_INTEGER: frozenset[str] = frozenset({"max_age", "new_agents"})

RESPONSE_METRICS: tuple[str, ...] = ("peak_window_mean", "peak_strength_mean")


def define_problem(names: Sequence[str] = PARAM_NAMES) -> dict:
    """Return a SALib problem dict for the requested parameter subset."""
    selected = list(names)
    return {
        "num_vars": len(selected),
        "names": selected,
        "bounds": [list(PARAM_BOUNDS[n]) for n in selected],
    }


def coerce_value(name: str, value: float) -> float | int:
    """Cast a sampled float to int for integer-valued params."""
    if name in PARAM_INTEGER:
        return int(round(float(value)))
    return float(value)


def build_overrides(
    params: dict[str, float],
    *,
    memory_baseline: str,
    repeats: int,
    years: int,
    num_process: int,
    climate_process: str = "ar1",
    extra: dict[str, str] | None = None,
) -> list[str]:
    """Compose Hydra overrides for a single ABM invocation."""
    overrides: list[str] = [
        f"model.repeats={repeats}",
        f"model.num_process={num_process}",
        f"model.years={years}",
        f"model.memory_baseline={memory_baseline}",
        f"model.climate_process={climate_process}",
    ]
    for name, value in params.items():
        overrides.append(f"model.{name}={coerce_value(name, value)}")
    if extra:
        overrides.extend(f"{k}={v}" for k, v in extra.items())
    return overrides


def compute_metrics(corr_df: pd.DataFrame) -> dict[str, float]:
    """Extract per-batch response scalars from a ``correlations.csv`` frame.

    Columns are replicate runs (`model_<id>`); index is window size.
    """
    if corr_df.empty:
        raise ValueError("correlations.csv is empty; ABM run produced no output")
    numeric = corr_df.apply(pd.to_numeric, errors="coerce")
    peak_windows = numeric.idxmax(axis=0).astype(float)
    peak_strengths = numeric.max(axis=0).astype(float)
    n_replicates = int(peak_strengths.notna().sum())
    return {
        "peak_window_mean": float(peak_windows.mean()),
        "peak_window_std": (
            float(peak_windows.std(ddof=1)) if n_replicates > 1 else 0.0
        ),
        "peak_strength_mean": float(peak_strengths.mean()),
        "peak_strength_std": (
            float(peak_strengths.std(ddof=1)) if n_replicates > 1 else 0.0
        ),
        "n_replicates": n_replicates,
    }


@dataclass
class RunOneResult:
    params: dict[str, float | int]
    metrics: dict[str, float]
    elapsed_seconds: float
    run_dir: Path


def run_one(
    param_values: Sequence[float],
    *,
    memory_baseline: str,
    run_dir: Path,
    param_names: Sequence[str] = PARAM_NAMES,
    repeats: int = 30,
    years: int = 100,
    num_process: int = 1,
    climate_process: str = "ar1",
    cleanup: bool = False,
    timeout: float | None = None,
) -> RunOneResult:
    """Run one ABM batch at a parameter vector and return scalar metrics.

    Spawns ``uv run python shifting_baseline/abm.py …`` as a subprocess so the
    Hydra entrypoint stays the source of truth. Output goes to ``run_dir``.
    """
    if len(param_values) != len(param_names):
        raise ValueError(
            f"param_values length {len(param_values)} != names length {len(param_names)}"
        )
    params = {name: float(value) for name, value in zip(param_names, param_values)}
    casted = {name: coerce_value(name, value) for name, value in params.items()}

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = build_overrides(
        params,
        memory_baseline=memory_baseline,
        repeats=repeats,
        years=years,
        num_process=num_process,
        climate_process=climate_process,
        extra={"hydra.run.dir": run_dir.as_posix()},
    )
    command = [
        "uv",
        "run",
        "python",
        "shifting_baseline/abm.py",
        *overrides,
    ]

    env = os.environ.copy()
    env.setdefault("EMAIL_DISABLE", "1")
    start = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=timeout, env=env)
    elapsed = time.perf_counter() - start

    corr_path = run_dir / "correlations.csv"
    if not corr_path.exists():
        raise FileNotFoundError(
            f"ABM finished but correlations.csv not found at {corr_path}"
        )
    metrics = compute_metrics(pd.read_csv(corr_path, index_col=0))

    if cleanup:
        shutil.rmtree(run_dir, ignore_errors=True)

    return RunOneResult(
        params=casted, metrics=metrics, elapsed_seconds=elapsed, run_dir=run_dir
    )


def _evaluate_samples(
    samples: np.ndarray,
    *,
    output_root: Path,
    memory_baseline: str,
    param_names: Sequence[str],
    repeats: int,
    years: int,
    num_process: int,
    climate_process: str,
    n_workers: int,
    keep_run_dirs: bool,
    timeout: float | None = None,
) -> pd.DataFrame:
    """Run ABM at each row of ``samples`` (concurrent threads). Returns long-form DataFrame."""

    def _one(idx: int) -> dict:
        run_dir = output_root / f"sample_{idx:06d}"
        try:
            result = run_one(
                samples[idx],
                memory_baseline=memory_baseline,
                run_dir=run_dir,
                param_names=param_names,
                repeats=repeats,
                years=years,
                num_process=num_process,
                climate_process=climate_process,
                cleanup=not keep_run_dirs,
                timeout=timeout,
            )
            row = {
                "sample_idx": idx,
                **result.params,
                **result.metrics,
                "elapsed_seconds": result.elapsed_seconds,
                "status": "ok",
            }
        except subprocess.TimeoutExpired:
            row = {
                "sample_idx": idx,
                "status": f"error: timeout after {timeout}s",
                **{name: float(samples[idx][i]) for i, name in enumerate(param_names)},
            }
        except Exception as exc:  # pylint: disable=broad-except
            row = {
                "sample_idx": idx,
                "status": f"error: {exc}",
                **{name: float(samples[idx][i]) for i, name in enumerate(param_names)},
            }
        return row

    rows: list[dict] = []
    if n_workers <= 1:
        for idx in range(len(samples)):
            rows.append(_one(idx))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_one, idx): idx for idx in range(len(samples))}
            for fut in as_completed(futures):
                rows.append(fut.result())
    rows.sort(key=lambda r: r["sample_idx"])
    return pd.DataFrame(rows)


def benchmark_runtime(
    output_root: Path,
    *,
    memory_baseline: str = "personal",
    repeats: int = 30,
    years: int = 100,
    num_process: int = 1,
    climate_process: str = "ar1",
    n_samples: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Run ``n_samples`` Latin-hypercube-ish points to gauge wall-clock cost."""
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [
            rng.uniform(low=PARAM_BOUNDS[n][0], high=PARAM_BOUNDS[n][1], size=n_samples)
            for n in PARAM_NAMES
        ]
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    df = _evaluate_samples(
        samples,
        output_root=output_root,
        memory_baseline=memory_baseline,
        param_names=PARAM_NAMES,
        repeats=repeats,
        years=years,
        num_process=num_process,
        climate_process=climate_process,
        n_workers=1,
        keep_run_dirs=False,
    )
    df.to_csv(output_root / "benchmark.csv", index=False)
    return df


def run_morris(
    output_root: Path,
    *,
    memory_baseline: str,
    r_trajectories: int = 20,
    num_levels: int = 4,
    repeats: int = 30,
    years: int = 100,
    num_process: int = 1,
    climate_process: str = "ar1",
    n_workers: int = 1,
    keep_run_dirs: bool = False,
    seed: int = 20260425,
    param_names: Sequence[str] = PARAM_NAMES,
    timeout: float | None = None,
) -> dict:
    """Generate Morris trajectories, evaluate, and analyze.

    Writes: sample matrix, raw outputs, and analysis indices.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    problem = define_problem(param_names)
    samples = morris_sample.sample(
        problem, N=r_trajectories, num_levels=num_levels, seed=seed
    )
    np.savetxt(
        output_root / "samples.csv",
        samples,
        header=",".join(problem["names"]),
        delimiter=",",
        comments="",
    )
    (output_root / "problem.json").write_text(json.dumps(problem, indent=2))

    raw = _evaluate_samples(
        samples,
        output_root=output_root / "runs",
        memory_baseline=memory_baseline,
        param_names=problem["names"],
        repeats=repeats,
        years=years,
        num_process=num_process,
        climate_process=climate_process,
        n_workers=n_workers,
        keep_run_dirs=keep_run_dirs,
        timeout=timeout,
    )
    raw.to_csv(output_root / "raw_outputs.csv", index=False)

    indices: dict[str, pd.DataFrame] = {}
    successful = raw[raw["status"] == "ok"]
    if len(successful) != len(samples):
        (output_root / "ERRORS.txt").write_text(
            f"{len(samples) - len(successful)} of {len(samples)} samples failed\n"
        )
    for metric in RESPONSE_METRICS:
        Y = raw[metric].to_numpy(dtype=float)
        if np.isnan(Y).any():
            # SALib will refuse NaNs; fill with median for analysis stability
            Y = np.where(np.isnan(Y), np.nanmedian(Y), Y)
        Si = morris_analyze.analyze(
            problem, samples, Y, num_levels=num_levels, seed=seed
        )
        df = pd.DataFrame(
            {
                "name": problem["names"],
                "mu": Si["mu"],
                "mu_star": Si["mu_star"],
                "sigma": Si["sigma"],
                "mu_star_conf": Si["mu_star_conf"],
            }
        )
        df.to_csv(output_root / f"morris_{metric}.csv", index=False)
        indices[metric] = df
    return {"problem": problem, "samples": samples, "raw": raw, "indices": indices}


def run_sobol(
    output_root: Path,
    *,
    memory_baseline: str,
    N: int = 1024,
    repeats: int = 30,
    years: int = 100,
    num_process: int = 1,
    climate_process: str = "ar1",
    n_workers: int = 1,
    keep_run_dirs: bool = False,
    seed: int = 20260425,
    calc_second_order: bool = False,
    param_names: Sequence[str] = PARAM_NAMES,
    timeout: float | None = None,
) -> dict:
    """Generate Saltelli samples, evaluate ABM, and analyze with Sobol."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    problem = define_problem(param_names)
    samples = sobol_sample.sample(
        problem, N, calc_second_order=calc_second_order, seed=seed
    )
    np.savetxt(
        output_root / "samples.csv",
        samples,
        header=",".join(problem["names"]),
        delimiter=",",
        comments="",
    )
    (output_root / "problem.json").write_text(
        json.dumps(
            {**problem, "N": N, "calc_second_order": calc_second_order, "seed": seed},
            indent=2,
        )
    )

    raw = _evaluate_samples(
        samples,
        output_root=output_root / "runs",
        memory_baseline=memory_baseline,
        param_names=problem["names"],
        repeats=repeats,
        years=years,
        num_process=num_process,
        climate_process=climate_process,
        n_workers=n_workers,
        keep_run_dirs=keep_run_dirs,
        timeout=timeout,
    )
    raw.to_csv(output_root / "raw_outputs.csv", index=False)

    indices: dict[str, pd.DataFrame] = {}
    successful = raw[raw["status"] == "ok"]
    if len(successful) != len(samples):
        (output_root / "ERRORS.txt").write_text(
            f"{len(samples) - len(successful)} of {len(samples)} samples failed\n"
        )
    for metric in RESPONSE_METRICS:
        Y = raw[metric].to_numpy(dtype=float)
        if np.isnan(Y).any():
            Y = np.where(np.isnan(Y), np.nanmedian(Y), Y)
        Si = sobol_analyze.analyze(
            problem, Y, calc_second_order=calc_second_order, seed=seed
        )
        df = pd.DataFrame(
            {
                "name": problem["names"],
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            }
        )
        df.to_csv(output_root / f"sobol_{metric}.csv", index=False)
        indices[metric] = df
    return {"problem": problem, "samples": samples, "raw": raw, "indices": indices}


def iter_param_dicts(
    samples: np.ndarray, names: Iterable[str] = PARAM_NAMES
) -> list[dict[str, float | int]]:
    """Convert a SALib sample matrix into a list of typed param dicts."""
    return [{n: coerce_value(n, v) for n, v in zip(names, row)} for row in samples]
