"""Before/after comparison for Bug 2a (calibration.py reindex fix).

Runs the same mismatch analysis as reports/mismatch.ipynb, twice:

1. ``before`` — restores the original buggy ``for matrix in [...]: matrix = matrix.reindex(...)``
   loop via monkeypatching ``_run_significance_test``, so the reindex is a no-op.
2. ``after`` — uses the current, fixed implementation that assigns explicitly.

Outputs (under ``reports/results/_bug2a_compare/``):

- ``{before,after}/p_value_matrix.csv``
- ``{before,after}/diff_matrix.csv``
- ``{before,after}/cm_df.csv``
- ``{before,after}/figure3.png``
- ``summary.md`` — side-by-side numerical diff.

Not run as a test; intended as a one-shot reproducer for the reviewer-response
package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize
from scipy.stats import norm
from tqdm.auto import tqdm

from shifting_baseline import calibration
from shifting_baseline.calibration import MismatchReport
from shifting_baseline.constants import LEVELS, LEVELS_PROB
from shifting_baseline.data import load_data, load_validation_data
from shifting_baseline.filters import classify

logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).resolve().parent / "results" / "_bug2a_compare"
BEFORE_DIR = OUT_ROOT / "before"
AFTER_DIR = OUT_ROOT / "after"
BEFORE_DIR.mkdir(parents=True, exist_ok=True)
AFTER_DIR.mkdir(parents=True, exist_ok=True)


def _buggy_significance_test(
    self: MismatchReport, mc_runs: int = 1000, shift: int = 1
) -> None:
    """Replica of the pre-fix ``_run_significance_test``.

    Identical to the current implementation except the reindex step is
    written in the broken for-loop form — the loop variable is rebound but
    the original matrices are never mutated.
    """
    all_diff_matrices = []
    n_samples = self.n_raw_samples

    for _ in tqdm(range(mc_runs), desc="MC模拟(before)"):
        random_data = np.random.normal(0, 1, n_samples)
        random_pred = np.random.choice(LEVELS, size=n_samples, p=LEVELS_PROB)

        random_df = pd.DataFrame(
            {
                "value": random_data,
                "pred": random_pred,
                "true": classify(random_data),
            }
        )

        random_df["exact"] = random_df["pred"] == random_df["true"]
        random_df["last"] = self._generate_last_column(random_df, shift=shift)
        random_df["diff"] = random_df["value"] - random_df["last"]

        diff_matrix = self._create_misclassification_matrix(random_df)
        if not diff_matrix.empty:
            all_diff_matrices.append(diff_matrix)

    if not all_diff_matrices:
        return

    combined_matrices = pd.concat(all_diff_matrices)
    mc_mean_matrix = combined_matrices.groupby(level=0).mean()
    mc_std_matrix = combined_matrices.groupby(level=0).std()

    # Intentionally broken: rebinding the loop variable does nothing.
    for matrix in [mc_mean_matrix, mc_std_matrix]:
        matrix = matrix.reindex(index=LEVELS, columns=LEVELS)  # noqa: F841

    z_scores = (self.diff_matrix - mc_mean_matrix) / mc_std_matrix
    z_scores = z_scores.astype(float)

    def safe_norm_cdf(z: float) -> float:
        if pd.isna(z):
            return np.nan
        return 2 * (1 - norm.cdf(abs(z)))

    self.p_value_matrix = z_scores.apply(lambda z: z.map(safe_norm_cdf))


def _run_one(
    tag: str,
    out_dir: Path,
    pred: pd.Series,
    true: pd.Series,
    value_series: pd.Series,
    sig_test_override: Callable | None,
    mc_runs: int,
    seed: int,
) -> MismatchReport:
    np.random.seed(seed)
    report = MismatchReport(pred=pred, true=true, value_series=value_series)
    if sig_test_override is not None:
        # Rebind only on this instance; setattr avoids mypy method-assign on the bound method slot.
        import types

        setattr(
            report,
            "_run_significance_test",
            types.MethodType(sig_test_override, report),
        )
    report.analyze_error_patterns(mc_runs=mc_runs)

    report.cm_df.to_csv(out_dir / "cm_df.csv")
    if report.diff_matrix is not None:
        report.diff_matrix.to_csv(out_dir / "diff_matrix.csv")
    if report.p_value_matrix is not None:
        report.p_value_matrix.to_csv(out_dir / "p_value_matrix.csv")

    fig = report.generate_report_figure(figsize=(5.2, 3))
    fig.savefig(out_dir / "figure3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("[%s] saved to %s", tag, out_dir)
    return report


def _summarise(before: MismatchReport, after: MismatchReport) -> str:
    assert before.diff_matrix is not None and after.diff_matrix is not None
    assert before.p_value_matrix is not None and after.p_value_matrix is not None

    lines: list[str] = []
    lines.append("# Bug 2a before/after summary\n")

    lines.append("## confusion matrix (cm_df)\n")
    cm_diff = (before.cm_df - after.cm_df).abs().sum().sum()
    lines.append(
        f"- absolute cell-sum diff: **{cm_diff}** (expect 0: bug does not touch cm_df)\n"
    )

    lines.append("\n## diff_matrix (non-MC part of analyze_error_patterns)\n")
    diff_abs = (
        (before.diff_matrix.fillna(0) - after.diff_matrix.fillna(0)).abs().sum().sum()
    )
    lines.append(
        f"- absolute cell-sum diff: **{diff_abs:.6f}** (expect 0: bug is only in the MC step)\n"
    )

    lines.append("\n## p_value_matrix\n")
    lines.append(
        f"- before shape: {before.p_value_matrix.shape}, axes: index={list(before.p_value_matrix.index)}, columns={list(before.p_value_matrix.columns)}\n"
    )
    lines.append(
        f"- after  shape: {after.p_value_matrix.shape}, axes: index={list(after.p_value_matrix.index)}, columns={list(after.p_value_matrix.columns)}\n"
    )
    common_index = before.p_value_matrix.index.intersection(after.p_value_matrix.index)
    common_cols = before.p_value_matrix.columns.intersection(
        after.p_value_matrix.columns
    )
    before_sub = before.p_value_matrix.loc[common_index, common_cols]
    after_sub = after.p_value_matrix.loc[common_index, common_cols]
    # numeric diff where both non-NaN
    both_valid = before_sub.notna() & after_sub.notna()
    if both_valid.any().any():
        max_abs = (
            (before_sub.where(both_valid) - after_sub.where(both_valid))
            .abs()
            .max()
            .max()
        )
        lines.append(
            f"- max |before - after| on cells non-NaN in BOTH (aligned on common axes): **{float(max_abs):.6f}**\n"
        )
    else:
        lines.append(
            "- no cells non-NaN in both; arrays disagree on NaN pattern alone.\n"
        )

    # Report which cells changed NaN-ness
    before_full = before.p_value_matrix.reindex(index=LEVELS, columns=LEVELS)
    after_full = after.p_value_matrix.reindex(index=LEVELS, columns=LEVELS)
    nan_flipped = before_full.isna() != after_full.isna()
    n_flipped = int(nan_flipped.values.sum())
    lines.append(
        f"- cells whose NaN/non-NaN status flipped between before and after: **{n_flipped}**\n"
    )
    if n_flipped > 0:
        coords = [
            (i, c)
            for i in nan_flipped.index
            for c in nan_flipped.columns
            if bool(nan_flipped.loc[i, c])
        ]
        lines.append(f"  - coords (pred, true): {coords}\n")

    return "".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config")

    combined, _uncertainties, history = load_data(cfg)

    # ---- Validation fixture (matches mismatch.ipynb) ----
    best_val = cfg.using_val_data
    ds = cfg.ds.validation[best_val]
    _, validation_data = load_validation_data(
        data_path=ds.z_nc,
        resolution=cfg.resolution,
        csv_save_to=ds.csv,
        nc_save_to=ds.z_nc,
    )
    tree_ring = combined["mean"]

    # ---- Main historical-vs-natural fixture ----
    history.setup()
    pred_main, true_main = history.aggregate(
        how=cfg.agg_method,
        to_int=False,
        inplace=True,
    ).merge_with(
        combined["mean"],
        time_range="2-3",
        split=True,
    )

    pred_cls = classify(pred_main, handle_na="skip")
    true_cls = classify(true_main, handle_na="skip")

    mc_runs = 1000
    seed = 20260418

    logger.info("=== running BEFORE (buggy for-loop reindex) ===")
    before = _run_one(
        "before",
        BEFORE_DIR,
        pred=pred_cls,
        true=true_cls,
        value_series=true_main,
        sig_test_override=_buggy_significance_test,
        mc_runs=mc_runs,
        seed=seed,
    )

    logger.info("=== running AFTER (fixed explicit assignment) ===")
    after = _run_one(
        "after",
        AFTER_DIR,
        pred=pred_cls,
        true=true_cls,
        value_series=true_main,
        sig_test_override=None,
        mc_runs=mc_runs,
        seed=seed,
    )

    summary = _summarise(before, after)
    (OUT_ROOT / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("summary written to %s", OUT_ROOT / "summary.md")
    print(summary)

    # Silence unused-variable warnings from the validation report
    _ = (validation_data, tree_ring)


if __name__ == "__main__":
    main()
