#!/usr/bin/env python3
"""Run simple ABM climate-scenario robustness checks.

This script focuses on two reviewer-facing checks:
1. Different climate forcing generators (`iid`, `ar1`, `trend_plus_noise`).
2. Different time resolutions via `step_per_year`.

It runs the existing ABM entrypoint with Hydra overrides and summarizes:
- peak window location (`window_peak_location`)
- peak strength (`window_peak_strength`)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """Configuration for one climate/time-resolution scenario."""

    name: str
    climate_process: str
    step_per_year: int
    climate_sigma: float = 1.0
    climate_phi: float = 0.5
    climate_trend: float = 0.0
    subannual_aggregation: str = "mean"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="iid_annual", climate_process="iid", step_per_year=1),
    Scenario(name="iid_subannual4", climate_process="iid", step_per_year=4),
    Scenario(
        name="ar1_annual", climate_process="ar1", step_per_year=1, climate_phi=0.6
    ),
    Scenario(
        name="ar1_subannual4",
        climate_process="ar1",
        step_per_year=4,
        climate_phi=0.6,
    ),
    Scenario(
        name="trend_annual",
        climate_process="trend_plus_noise",
        step_per_year=1,
        climate_trend=0.03,
    ),
    Scenario(
        name="trend_subannual4",
        climate_process="trend_plus_noise",
        step_per_year=4,
        climate_trend=0.03,
    ),
)


def run_scenario(
    repo_root: Path, output_dir: Path, scenario: Scenario, repeats: int, years: int
) -> None:
    """Execute one scenario by calling the ABM entrypoint.

    Args:
        repo_root: Repository root.
        output_dir: Scenario-specific output directory.
        scenario: Scenario settings.
        repeats: Number of repeated runs.
        years: Simulated analysis years.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``climate_sigma`` is interpreted as yearly sigma by the model; the
    # model rescales it to tick-scale internally when step_per_year > 1.
    command = [
        "uv",
        "run",
        "python",
        "shifting_baseline/abm.py",
        f"model.repeats={repeats}",
        "model.num_process=1",
        f"model.years={years}",
        f"model.climate_process={scenario.climate_process}",
        f"model.step_per_year={scenario.step_per_year}",
        f"model.subannual_aggregation={scenario.subannual_aggregation}",
        f"model.climate_sigma={scenario.climate_sigma}",
        f"model.climate_phi={scenario.climate_phi}",
        f"model.climate_trend={scenario.climate_trend}",
        f"hydra.run.dir={output_dir.as_posix()}",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def summarize_scenario(
    scenario: Scenario, corr_path: Path
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Summarize per-run peak windows and strengths for one scenario.

    Args:
        scenario: Scenario metadata.
        corr_path: Path to `correlations.csv`.

    Returns:
        Tuple of (run-level peaks DataFrame, scenario-level aggregate metrics).
    """
    corr_df = pd.read_csv(corr_path, index_col=0)
    corr_df.index = corr_df.index.astype(int)

    run_rows: list[dict[str, float | int | str]] = []
    for col in corr_df.columns:
        series = corr_df[col].astype(float)
        peak_window = int(series.idxmax())
        peak_strength = float(series.max())
        run_rows.append(
            {
                "scenario": scenario.name,
                "run_id": col,
                "window_peak_location": peak_window,
                "window_peak_strength": peak_strength,
            }
        )

    run_peaks = pd.DataFrame(run_rows)
    summary: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "climate_process": scenario.climate_process,
        "step_per_year": scenario.step_per_year,
        "climate_phi": scenario.climate_phi,
        "climate_trend": scenario.climate_trend,
        "n_runs": int(len(run_peaks)),
        "window_peak_location_mean": float(run_peaks["window_peak_location"].mean()),
        "window_peak_location_std": float(run_peaks["window_peak_location"].std(ddof=1))
        if len(run_peaks) > 1
        else 0.0,
        "window_peak_strength_mean": float(run_peaks["window_peak_strength"].mean()),
        "window_peak_strength_std": float(run_peaks["window_peak_strength"].std(ddof=1))
        if len(run_peaks) > 1
        else 0.0,
    }
    return run_peaks, summary


def write_markdown_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write a concise markdown report for reviewer-facing communication."""
    display_cols = [
        "scenario",
        "climate_process",
        "step_per_year",
        "window_peak_location_mean",
        "window_peak_location_std",
        "window_peak_strength_mean",
        "window_peak_strength_std",
        "n_runs",
    ]
    rounded = summary_df[display_cols].round(3)
    header = "| " + " | ".join(rounded.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rounded.itertuples(index=False, name=None)
    ]
    table = "\n".join([header, separator, *rows])
    content = "\n".join(
        [
            "# Climate Scenario Sanity Checks",
            "",
            "This report compares ABM peak-window behavior under different climate forcings",
            "and time resolutions. Window metrics are reported in yearly units.",
            "",
            "## Scenario summary",
            "",
            table,
            "",
            "## Interpretation notes",
            "",
            "- `window_peak_location_mean` is the average peak window location across repeated runs.",
            "- `window_peak_strength_mean` is the average maximum correlation strength across runs.",
            "- `step_per_year > 1` uses subannual forcing and yearly aggregation before correlation.",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ABM climate scenario checks.")
    parser.add_argument(
        "--repeats", type=int, default=4, help="Repeated runs per scenario."
    )
    parser.add_argument("--years", type=int, default=80, help="Analysis years per run.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="reports/results/climate_scenarios",
        help="Directory for scenario outputs and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    all_run_peaks: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for scenario in SCENARIOS:
        scenario_dir = output_root / scenario.name
        run_scenario(
            repo_root, scenario_dir, scenario, repeats=args.repeats, years=args.years
        )
        run_peaks, summary = summarize_scenario(
            scenario, scenario_dir / "correlations.csv"
        )
        all_run_peaks.append(run_peaks)
        summary_rows.append(summary)

    run_peaks_df = pd.concat(all_run_peaks, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["climate_process", "step_per_year"]
    )

    run_peaks_df.to_csv(output_root / "climate_scenario_run_peaks.csv", index=False)
    summary_df.to_csv(output_root / "climate_scenario_summary.csv", index=False)
    write_markdown_summary(summary_df, output_root / "climate_scenario_summary.md")


if __name__ == "__main__":
    main()
