#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Prior-sensitivity analysis of the Bayesian N-WDI integration (SI Table S11).

Re-fits :func:`shifting_baseline.mc.combine_reconstructions` under alternative
priors, varying one factor at a time around the manuscript's specification
(theta_t ~ N(0, 1), nu ~ Gamma(2, 0.1)), and reports for each variant:

- the posterior of nu (mean + 94% HDI) and convergence (R-hat, ESS);
- the latent posterior uncertainty (median posterior SD / 94% HDI width);
- agreement of the integrated N-WDI with the baseline fit (Pearson r, max |diff|,
  five-level category agreement and weighted kappa);
- the manuscript's headline numbers, regenerated through
  :func:`shifting_baseline.results.build_results` with the variant N-WDI passed in
  place of the cached one.

Outputs ``${ds.processed}/bayes_prior_sensitivity.csv`` and the "Table S11" sheet
of ``<ds.figs>/CollMemo_Tables.xlsx``.

Run:
  uv run python scripts/bayes_prior_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.constants import START
from shifting_baseline.data import load_nat_data
from shifting_baseline.filters import classify
from shifting_baseline.mc import (
    combine_reconstructions,
    diagnose_trace,
    summarize_latent,
)
from shifting_baseline.results import build_results
from shifting_baseline.utils.io import write_table_sheet
from shifting_baseline.utils.log import get_logger, setup_logger_from_hydra

matplotlib.use("Agg")  # build_results 内部会画热图

_TABLE_SHEET = "Table S11"

# 单因素变化：(SI 表中写的先验改动, combine_reconstructions 参数)；第一项为基线
PRIOR_VARIANTS: list[tuple[str, dict]] = [
    ("Baseline", {}),
    ("θ ~ N(0, 0.5)", {"theta_sigma": 0.5}),
    ("θ ~ N(0, 2)", {"theta_sigma": 2.0}),
    ("θ ~ N(0, 10)", {"theta_sigma": 10.0}),
    ("ν ~ Gamma(2, 1)", {"nu_prior": (2.0, 1.0)}),
    ("ν ~ Exp(1/30)", {"nu_prior": (1.0, 1 / 30)}),
    ("ν = 4 (fixed)", {"nu_fixed": 4.0}),
    ("Gaussian likelihood", {"nu_fixed": np.inf}),
]

# 写入表格的正文数字（results.json 中的键）
_HEADLINE = {
    ("results1", "corr"): "Validation r",
    ("results2", "kendall_tau"): "Kendall tau (H vs N)",
    ("results2", "tau_p_value"): "tau p",
    ("results2", "kappa"): "Weighted kappa (H vs N)",
    ("results2", "accuracy"): "Accuracy (H vs N)",
    ("results3", "w_optimal"): "Optimal window (whole)",
    ("results3", "optimal_segment_w"): "Optimal window (segments)",
    ("results3", "max_increase"): "Max increase (%)",
}


def _agreement(variant: pd.Series, baseline: pd.Series) -> dict[str, float]:
    """与基线 N-WDI 的一致性：连续值与五级分类。"""
    stats = MismatchReport(
        pred=classify(variant), true=classify(baseline)
    ).get_statistics_summary(weights="linear")
    return {
        "Pearson r vs baseline": float(variant.corr(baseline)),
        "Max |diff| vs baseline": float((variant - baseline).abs().max()),
        "Same category (%)": float(stats["accuracy"] * 100),
        "Weighted kappa vs baseline": float(stats["kappa"]),
    }


def format_table(raw: pd.DataFrame) -> pd.DataFrame:
    """把完整结果整理成 SI Table S11 的精简排版（完整数值保留在 csv）。"""

    def _nu(row) -> str:
        if pd.isna(row["nu_mean"]):
            return "–"
        return (
            f"{row['nu_mean']:.0f} [{row['nu_hdi_low']:.0f}, {row['nu_hdi_high']:.0f}]"
        )

    return pd.DataFrame(
        {
            "Prior change": raw["Variant"],
            "Posterior ν": raw.apply(_nu, axis=1),
            "r": raw["Pearson r vs baseline"].map("{:.3f}".format),
            "Same class (%)": raw["Same category (%)"].map("{:.0f}".format),
            "τ (H vs N)": raw["Kendall tau (H vs N)"].map("{:.3f}".format),
            "κ (H vs N)": raw["Weighted kappa (H vs N)"].map("{:.2f}".format),
            "Optimal w (yr)": raw["Optimal window (segments)"],
        }
    )


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        raise ValueError("cfg 不能为空")
    OmegaConf.resolve(cfg)
    setup_logger_from_hydra(cfg)
    log = get_logger(__name__)

    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        start_year=START,
        standardize=True,
    )
    seed = cfg.get("random_seed", 42)

    rows: list[dict[str, object]] = []
    baseline: pd.Series | None = None
    for label, kwargs in PRIOR_VARIANTS:
        log.info("先验敏感性: %s %s", label, kwargs)
        combined, trace = combine_reconstructions(
            datasets,
            uncertainties,
            standardize=True,
            random_seed=seed,
            **kwargs,
        )
        if baseline is None:
            baseline = combined["mean"]
        # 用该变体的 N-WDI 代替缓存，重算正文数字
        results = build_results(cfg, combined)
        row: dict[str, object] = {
            "Variant": label,
            **diagnose_trace(trace),
            **summarize_latent(combined),
            **_agreement(combined["mean"], baseline),
            **{name: results[b][k] for (b, k), name in _HEADLINE.items()},
        }
        rows.append(row)
        log.info("结果: %s", row)

    table = pd.DataFrame(rows).round(3)
    out_csv = Path(cfg.ds.processed) / "bayes_prior_sensitivity.csv"
    table.to_csv(out_csv, index=False)
    log.info("先验敏感性表:\n%s", table.T.to_string())
    write_table_sheet(cfg.ds.figs, format_table(table), _TABLE_SHEET)
    log.info("完成 -> %s", out_csv)


if __name__ == "__main__":
    _main()
