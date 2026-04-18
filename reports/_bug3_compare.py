"""Before/after comparison for Bug 3 (compare.py min_periods default fix).

The bug is at ``shifting_baseline/compare.py:58`` — the default
``min_periods`` was written as ``min(np.log2(n), 2)`` but the comment
clearly intends ``max(int(np.log2(n)), 2)``. For realistic ``n >= 4``
the pre-fix default collapsed to ``2``, regardless of ``n``.

The only real-data call path that hits this default is
``sweep_max_corr_year`` at compare.py:283, invoked from
``reports/history.ipynb`` cell 10. That cell drives Figure 4's
``r_benchmark_list`` (per-slice unfiltered ``Avg. Tau``) and
``max_corr`` (per-slice best filtered correlation). Both are computed
by calling ``compare_corr`` with ``filter_func=calc_std_deviation`` but
**without** an explicit ``min_periods`` in ``**compare_kwargs`` — so
the default matters.

This script reproduces the same cell in two configurations on the
same data:

1. ``before`` — ``compare_corr`` monkeypatched to restore the
   pre-fix ``min(np.log2(n), 2)`` default (i.e. min_periods=2).
2. ``after`` — the current, fixed default.

Outputs (under ``reports/results/_bug3_compare/``):

- ``{before,after}/sweep_max_corr_year.csv`` — r_benchmark_list,
  max_corr stack, p_value_list.
- ``summary.md`` — side-by-side numerical diff.

Not a test; intended as a one-shot reproducer for the reviewer-response
package. Mirrors ``reports/_bug2a_compare.py``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize

from shifting_baseline import compare
from shifting_baseline.compare import sweep_max_corr_year, sweep_slices
from shifting_baseline.constants import END
from shifting_baseline.data import HistoricalRecords
from shifting_baseline.filters import calc_std_deviation

logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).resolve().parent / "results" / "_bug3_compare"
BEFORE_DIR = OUT_ROOT / "before"
AFTER_DIR = OUT_ROOT / "after"
BEFORE_DIR.mkdir(parents=True, exist_ok=True)
AFTER_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _with_buggy_default():
    """Temporarily restore the pre-fix buggy ``compare_corr`` default.

    Monkeypatches ``shifting_baseline.compare.compare_corr`` with a
    near-identical replica whose only difference is the default
    ``min_periods`` expression: ``min(np.log2(n), 2)`` (broken) instead
    of the fixed ``max(int(np.log2(n)), 2)``. Restored on exit.
    """
    from functools import wraps

    from shifting_baseline.utils.calc import calc_corr

    original = compare.compare_corr

    @wraps(original)
    def buggy_compare_corr(
        data1,
        data2,
        filter_func=None,
        filter_side="both",
        corr_method="pearson",
        window_error="raise",
        n_diff_w=2,
        **rolling_kwargs,
    ):
        r_benchmark, p_value, n = calc_corr(data1, data2, how=corr_method)
        if filter_func is None:
            return r_benchmark, p_value, n
        default_kwargs = {
            "window": n // 10,
            "center": False,
            "min_periods": min(np.log2(n), 2),  # the bug
            "closed": "both",
        }
        default_kwargs.update(rolling_kwargs)
        if default_kwargs["window"] <= default_kwargs["min_periods"] + n_diff_w:
            if window_error == "raise":
                raise ValueError(
                    f"窗口{default_kwargs['window']}太小，"
                    f"n_periods={default_kwargs['min_periods']}，"
                    f"n_diff_w={n_diff_w}，请增大窗口范围"
                )
            elif window_error == "nan":
                return np.nan, np.nan, n
            else:
                raise ValueError(f"无效的窗口错误处理方式: {window_error}")
        if filter_side == "both":
            d1 = data1.rolling(**default_kwargs).apply(filter_func)
            d2 = data2.rolling(**default_kwargs).apply(filter_func)
        elif filter_side == "left":
            d1 = data1.rolling(**default_kwargs).apply(filter_func)
            d2 = data2
        elif filter_side == "right":
            d1 = data1
            d2 = data2.rolling(**default_kwargs).apply(filter_func)
        else:
            raise ValueError(f"无效的过滤侧: {filter_side}")
        r, p, n_out = calc_corr(d1, d2, how=corr_method)
        return r, p, n_out

    compare.compare_corr = buggy_compare_corr
    try:
        yield
    finally:
        compare.compare_corr = original


def _run_sweep(
    data1: pd.Series,
    data2: pd.Series,
    slices: list[slice],
    cfg,
) -> dict:
    max_corr_year, max_corr, r_benchmark_list, p_value_list = sweep_max_corr_year(
        data1=data1,
        data2=data2,
        slices=slices,
        corr_method=cfg.corr_method,
        windows=np.arange(2, 100),
        min_periods=np.repeat(cfg.min_period, 98),
        filter_func=calc_std_deviation,
        ratio=cfg.ratio,
    )
    return {
        "max_corr_year": max_corr_year,
        "max_corr": max_corr,
        "r_benchmark_list": r_benchmark_list,
        "p_value_list": p_value_list,
    }


def _save(tag: str, out_dir: Path, res: dict) -> None:
    rows = []
    for idx, (r_bench, p, mc) in enumerate(
        zip(res["r_benchmark_list"], res["p_value_list"], res["max_corr"])
    ):
        rows.append(
            {
                "slice_idx": idx,
                "r_benchmark": float(r_bench),
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "max_corr_mean": float(np.mean(mc)),
                "max_corr_n": len(mc),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_max_corr_year.csv", index=False)
    logger.info("[%s] saved %s", tag, out_dir / "sweep_max_corr_year.csv")


def _summarise(before: dict, after: dict) -> str:
    lines: list[str] = []
    lines.append("# Bug 3 before/after summary\n\n")
    lines.append(
        "Scope: `sweep_max_corr_year` called with `filter_func=calc_std_deviation` "
        "and no explicit `min_periods`, replicating `reports/history.ipynb` cell 10 "
        "— the sole real-data call path that hits compare.py:58's default.\n\n"
    )

    b_r = np.asarray(before["r_benchmark_list"], dtype=float)
    a_r = np.asarray(after["r_benchmark_list"], dtype=float)
    lines.append("## `r_benchmark_list` (unfiltered benchmark per slice)\n")
    lines.append(f"- n slices: {len(b_r)}\n")
    lines.append(
        f"- mean (before): **{np.nanmean(b_r):.4f}**; mean (after): **{np.nanmean(a_r):.4f}**\n"
    )
    lines.append(
        f"- max |before - after|: **{float(np.nanmax(np.abs(b_r - a_r))):.6f}**\n"
    )
    lines.append(
        f"- # slices where |diff| > 1e-6: **{int((np.abs(b_r - a_r) > 1e-6).sum())}**\n"
    )
    nan_flip = int((np.isnan(b_r) != np.isnan(a_r)).sum())
    lines.append(f"- # slices flipping NaN ↔ non-NaN: **{nan_flip}**\n")

    # max_corr (per-slice best filtered correlation — NOT affected by the bug
    # because compare_corr_2d receives explicit min_periods; included as sanity).
    b_m = np.array([np.mean(m) for m in before["max_corr"]])
    a_m = np.array([np.mean(m) for m in after["max_corr"]])
    lines.append("\n## `max_corr` mean per slice (should be identical — sanity)\n")
    lines.append(
        f"- max |before - after|: **{float(np.nanmax(np.abs(b_m - a_m))):.6f}**\n"
    )

    b_p = np.asarray(before["p_value_list"], dtype=float)
    a_p = np.asarray(after["p_value_list"], dtype=float)
    lines.append("\n## `p_value_list` (p of unfiltered benchmark)\n")
    lines.append(
        f"- max |before - after|: **{float(np.nanmax(np.abs(b_p - a_p))):.6f}**\n"
    )

    # Figure-4 labels that history.ipynb cell 10 prints in the text
    b_best = np.mean([np.mean(y) for y in before["max_corr_year"]])  # "Best year" label
    a_best = np.mean([np.mean(y) for y in after["max_corr_year"]])
    lines.append("\n## Figure-4 labels (from history.ipynb text annotation)\n")
    lines.append(
        "- `Avg. improvement of Tau` (mean of max_corr - r_benchmark): "
        f"before **{float(np.nanmean(np.array([np.mean(m) for m in before['max_corr']]) - b_r)):.4f}**, "
        f"after **{float(np.nanmean(np.array([np.mean(m) for m in after['max_corr']]) - a_r)):.4f}**\n"
    )
    lines.append(
        f"- `Best year` (mean of max_corr_year): before **{b_best:.2f}**, after **{a_best:.2f}**\n"
    )

    return "".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config")

    # Reproduce history.ipynb cell 3 + cell 10 data setup
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
        to_std=cfg.to_std,
    )
    folder_path = Path(cfg.ds.processed) / cfg.ds.setname
    combined = pd.read_csv(folder_path / "integrated.csv", index_col=0)
    tree_ring = combined["mean"]
    tree_ring.name = "tree_ring"

    history.setup()
    data1, data2 = history.merge_with(tree_ring, split=True)

    slices, _mid_year, _slice_labels = sweep_slices(
        start_year=1470,
        window_size=200,
        step_size=20,
        end_year=END,
    )

    np.random.seed(20260418)

    logger.info("=== running BEFORE (buggy min_periods=2 default) ===")
    with _with_buggy_default():
        before = _run_sweep(data1, data2, slices, cfg)
    _save("before", BEFORE_DIR, before)

    logger.info("=== running AFTER (fixed max(int(log2(n)), 2) default) ===")
    after = _run_sweep(data1, data2, slices, cfg)
    _save("after", AFTER_DIR, after)

    summary = _summarise(before, after)
    (OUT_ROOT / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("summary written to %s", OUT_ROOT / "summary.md")
    print(summary)


if __name__ == "__main__":
    main()
