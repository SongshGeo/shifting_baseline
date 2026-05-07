"""Quick visualization of Sobol SA results.

Auto-finds the two most-recent ``reports/results/sensitivity/*-sobol-*``
directories and writes:
  - ``<dir>/sobol_indices.png``       — S1/ST bar chart per metric
  - ``sobol_compare.png``             — both baselines side-by-side
  - ``sobol_distribution.png``        — peak_window / peak_strength histogram
                                         per baseline (uses ok rows only)

Usage:
    uv run python reports/plot_sobol.py
    uv run python reports/plot_sobol.py reports/results/sensitivity/<dir1> [<dir2>]

The script tolerates partial / failed runs: missing CSVs are annotated rather
than crashing, and the figure title shows the failure rate so you can judge
whether to trust the indices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper")

METRICS = ("peak_window_mean", "peak_strength_mean")
SENSITIVITY_ROOT = Path("reports/results/sensitivity")


def find_latest_sobol_dirs(n: int = 2) -> list[Path]:
    if not SENSITIVITY_ROOT.exists():
        return []
    dirs = sorted(
        SENSITIVITY_ROOT.glob("2026*-sobol-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return dirs[:n]


def baseline_of(d: Path) -> str:
    name = d.name
    if "collective" in name:
        return "collective"
    if "personal" in name:
        return "personal"
    if "model" in name:
        return "model"
    return name


def failure_summary(d: Path) -> tuple[int, int, float]:
    """Return (total, ok, failure_pct) reading raw_outputs.csv."""
    f = d / "raw_outputs.csv"
    if not f.exists():
        return 0, 0, float("nan")
    df = pd.read_csv(f)
    total = len(df)
    ok = (df["status"] == "ok").sum() if "status" in df else total
    pct = 100 * (total - ok) / total if total else float("nan")
    return total, ok, pct


def plot_indices_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    x = np.arange(len(df))
    w = 0.38
    ax.bar(
        x - w / 2,
        df["S1"],
        w,
        yerr=df.get("S1_conf"),
        label="$S_1$ (first-order)",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    ax.bar(
        x + w / 2,
        df["ST"],
        w,
        yerr=df.get("ST_conf"),
        label="$S_T$ (total)",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Sobol index")
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


def plot_one_dir(d: Path) -> Path | None:
    total, ok, pct = failure_summary(d)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m in zip(axes, METRICS):
        f = d / f"sobol_{m}.csv"
        if not f.exists():
            ax.text(0.5, 0.5, f"missing\n{f.name}", ha="center", va="center")
            ax.set_axis_off()
            continue
        df = pd.read_csv(f)
        plot_indices_panel(ax, df, f"{baseline_of(d)} — {m}")
    fig.suptitle(
        f"{d.name}    |    {ok}/{total} ok ({100 - pct:.1f}%, " f"fail {pct:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    out = d / "sobol_indices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def plot_compare(dirs: list[Path]) -> Path | None:
    if len(dirs) < 2:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, d in enumerate(dirs[:2]):
        b = baseline_of(d)
        for row, m in enumerate(METRICS):
            f = d / f"sobol_{m}.csv"
            ax = axes[row, col]
            if not f.exists():
                ax.text(0.5, 0.5, f"missing\n{f.name}", ha="center", va="center")
                ax.set_axis_off()
                continue
            df = pd.read_csv(f)
            plot_indices_panel(ax, df, f"{b} — {m}")
    fig.suptitle("Sobol sensitivity indices: personal vs collective", fontsize=12)
    fig.tight_layout()
    out = SENSITIVITY_ROOT / "sobol_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def plot_distribution(dirs: list[Path]) -> Path | None:
    if len(dirs) < 1:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    palette = {"personal": "#4C72B0", "collective": "#DD8452", "model": "#55A868"}
    for ax, m in zip(axes, METRICS):
        for d in dirs:
            b = baseline_of(d)
            raw = pd.read_csv(d / "raw_outputs.csv")
            ok_rows = raw[raw["status"] == "ok"] if "status" in raw else raw
            vals = ok_rows[m].dropna()
            ax.hist(
                vals,
                bins=40,
                alpha=0.55,
                label=f"{b} (n={len(vals)})",
                color=palette.get(b, "#888"),
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xlabel(m)
        ax.set_ylabel("count")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Response distribution across Sobol samples (ok rows only)", fontsize=11
    )
    fig.tight_layout()
    out = SENSITIVITY_ROOT / "sobol_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def plot_failure_corner(dirs: list[Path]) -> Path | None:
    """Diagnostic: which corners of parameter space drove the failures?"""
    have_failures = []
    for d in dirs:
        f = d / "raw_outputs.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "status" in df and (df["status"] != "ok").any():
            have_failures.append((d, df))
    if not have_failures:
        return None

    fig, axes = plt.subplots(
        len(have_failures),
        2,
        figsize=(11, 4.2 * len(have_failures)),
        squeeze=False,
    )
    for row, (d, df) in enumerate(have_failures):
        ok = df[df["status"] == "ok"]
        bad = df[df["status"] != "ok"]
        for col, (xp, yp) in enumerate(
            [("max_age", "new_agents"), ("max_age", "loss_rate")]
        ):
            ax = axes[row, col]
            ax.scatter(
                ok[xp], ok[yp], s=8, alpha=0.25, c="#888", label=f"ok (n={len(ok)})"
            )
            ax.scatter(
                bad[xp],
                bad[yp],
                s=18,
                alpha=0.85,
                c="#C44E52",
                edgecolor="black",
                linewidth=0.4,
                label=f"failed (n={len(bad)})",
            )
            ax.set_xlabel(xp)
            ax.set_ylabel(yp)
            ax.set_title(f"{baseline_of(d)} — {xp} vs {yp}", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.3)
    fig.suptitle("Failed-sample distribution in parameter space", fontsize=11)
    fig.tight_layout()
    out = SENSITIVITY_ROOT / "sobol_failure_corner.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def main(argv: list[str]) -> int:
    if argv:
        dirs = [Path(p) for p in argv]
    else:
        dirs = find_latest_sobol_dirs(2)
    if not dirs:
        print("No sobol output directories found.", file=sys.stderr)
        return 1
    print(f"plotting {len(dirs)} dir(s):")
    for d in dirs:
        print(f"  - {d}")
    for d in dirs:
        plot_one_dir(d)
    plot_compare(dirs)
    plot_distribution(dirs)
    plot_failure_corner(dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
