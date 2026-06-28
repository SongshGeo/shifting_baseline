"""Visualization of Sobol SA results.

Auto-finds the two most-recent ``reports/results/sensitivity/*-sobol-*``
directories and writes:

Exploratory (diagnostics; 150 DPI PNG):
  - ``<dir>/sobol_indices.png``       — S1/ST bar chart per metric
  - ``sobol_compare.png``             — both baselines side-by-side
  - ``sobol_distribution.png``        — peak_window / peak_strength histogram
                                         per baseline (uses ok rows only)
  - ``sobol_failure_corner.png``      — failed samples in param space

Publication quality (for SI; 300 DPI PNG + vector PDF, panel labels a-d):
  - ``sobol_compare_si.pdf``
  - ``sobol_compare_si.png``

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


def find_latest_sobol_dirs(n: int = 3) -> list[Path]:
    """Most-recent sobol dir *per baseline*, newest first.

    Re-runs of one baseline collapse to its latest dir, so a fresh
    collective_lifetime run sits alongside the older personal/collective
    runs instead of evicting one of them on raw mtime.
    """
    if not SENSITIVITY_ROOT.exists():
        return []
    dirs = sorted(
        SENSITIVITY_ROOT.glob("2026*-sobol-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    latest: dict[str, Path] = {}
    for d in dirs:
        latest.setdefault(baseline_of(d), d)
    return list(latest.values())[:n]


def baseline_of(d: Path) -> str:
    name = d.name
    # Order matters: "collective_lifetime" also contains "collective".
    if "collective_lifetime" in name:
        return "collective_lifetime"
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
    ordered = _ordered_dirs(dirs)
    ncol = len(ordered)
    fig, axes = plt.subplots(2, ncol, figsize=(6.5 * ncol, 9), squeeze=False)
    for col, d in enumerate(ordered):
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
    fig.suptitle(
        "Sobol sensitivity indices: " + " vs ".join(baseline_of(d) for d in ordered),
        fontsize=12,
    )
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
    palette = {
        "personal": "#4C72B0",
        "collective": "#DD8452",
        "collective_lifetime": "#55A868",
        "model": "#8172B3",
    }
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


PARAM_LABELS = {
    "max_age": r"$\mathit{max\_age}$",
    "new_agents": r"$\mathit{new\_agents}$",
    "loss_rate": r"$\mathit{loss\_rate}$",
    "climate_sigma": r"$\mathit{climate\_\sigma}$",
    "climate_phi": r"$\mathit{climate\_\phi}$",
}
METRIC_LABELS = {
    "peak_window_mean": "Peak-window location (years)",
    "peak_strength_mean": "Peak-correlation strength",
}


def _ordered_dirs(dirs: list[Path]) -> list[Path]:
    """Canonical baseline order: personal, collective_lifetime, collective, model."""
    order = {"personal": 0, "collective_lifetime": 1, "collective": 2, "model": 3}
    return sorted(dirs, key=lambda d: order.get(baseline_of(d), 99))


def plot_compare_si(dirs: list[Path]) -> tuple[Path, Path] | None:
    """Publication-quality Sobol comparison for SI use.

    Layout: rows = response metric, cols = memory baseline
    (personal | collective_lifetime | collective). Saves both PDF (vector)
    and 300-DPI PNG.
    """
    if len(dirs) < 2:
        return None
    ordered = _ordered_dirs(dirs)
    ncol = len(ordered)
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.bbox": "tight",
        }
    ):
        fig, axes = plt.subplots(
            2, ncol, figsize=(3.6 * ncol, 5.6), sharey=False, squeeze=False
        )
        panel_idx = 0
        for row, m in enumerate(METRICS):
            for col, d in enumerate(ordered):
                ax = axes[row, col]
                b = baseline_of(d)
                f = d / f"sobol_{m}.csv"
                if not f.exists():
                    ax.text(0.5, 0.5, f"missing\n{f.name}", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                df = pd.read_csv(f)
                x = np.arange(len(df))
                w = 0.38
                ax.bar(
                    x - w / 2,
                    df["S1"],
                    w,
                    yerr=df.get("S1_conf"),
                    label="$S_1$",
                    color="#4C72B0",
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=2.2,
                    error_kw={"elinewidth": 0.7},
                )
                ax.bar(
                    x + w / 2,
                    df["ST"],
                    w,
                    yerr=df.get("ST_conf"),
                    label="$S_T$",
                    color="#DD8452",
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=2.2,
                    error_kw={"elinewidth": 0.7},
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [PARAM_LABELS.get(n, n) for n in df["name"]],
                    rotation=30,
                    ha="right",
                )
                ax.axhline(0, color="black", lw=0.5)
                ax.set_ylim(-0.15, 1.1)
                if col == 0:
                    ax.set_ylabel("Sobol index")
                ax.set_title(f"{b} — {METRIC_LABELS.get(m, m)}", pad=5)
                if row == 0 and col == ncol - 1:
                    ax.legend(loc="upper right", frameon=False, handlelength=1.4)
                ax.text(
                    -0.12,
                    1.04,
                    f"({chr(ord('a') + panel_idx)})",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="bottom",
                )
                panel_idx += 1
        fig.tight_layout(h_pad=1.2, w_pad=1.6)
        out_pdf = SENSITIVITY_ROOT / "sobol_compare_si.pdf"
        out_png = SENSITIVITY_ROOT / "sobol_compare_si.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    print(f"  saved {out_pdf}")
    print(f"  saved {out_png}")
    return out_pdf, out_png


def main(argv: list[str]) -> int:
    if argv:
        dirs = [Path(p) for p in argv]
    else:
        dirs = find_latest_sobol_dirs()
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
    plot_compare_si(dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
