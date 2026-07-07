#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Build the integrated natural-proxy WDI (N-WDI) and its supporting tables.

This script owns **all data loading and Bayesian integration** for the
natural-proxy branch of the study, so that ``reports/natural.ipynb`` only needs
to *load and visualise* the artefacts produced here. Running it regenerates
every computed input consumed downstream by the notebook and the manuscript SI:

  ``data/combined_mean.csv``           integrated N-WDI (posterior mean/sd/HDI)
  ``data/combined_uncertainty.csv``    input rolling within-series uncertainties
  ``data/nwdi_distribution_fit.csv``   distribution-fit ranking of the N-WDI
  ``<ds.figs>/CollMemo_Tables.xlsx``   sheet "Table S10" (same ranking, MS format)

MCMC convergence diagnostics (posterior nu-hat, R-hat, ESS) are emitted to the
Hydra run log (``outputs/<timestamp>/``). Only the "Table S10" sheet of the
workbook is added/replaced; all hand-curated sheets are preserved.

Run:
  uv run python scripts/build_natural_index.py                # default ds=pure
  uv run python scripts/build_natural_index.py ds=best        # other scenarios
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from shifting_baseline.constants import START
from shifting_baseline.data import check_distribution, load_nat_data
from shifting_baseline.mc import combine_reconstructions
from shifting_baseline.utils.log import get_logger, setup_logger_from_hydra

# 分布拟合表(SI Table S10)的列名与四舍五入位数
_FIT_COLUMNS = {
    "sumsquare_error": ("Sum-sq. error", 4),
    "aic": ("AIC", 2),
    "bic": ("BIC", 2),
    "ks_statistic": ("KS statistic", 4),
    "ks_pvalue": ("KS p-value", 3),
}
_TABLE_SHEET = "Table S10"


def _build_fit_table(series: pd.Series) -> pd.DataFrame:
    """对整合后的 N-WDI 拟合候选分布，返回按 AIC 排序的整洁表格。"""
    summary = check_distribution(series.dropna(), only_best=False)
    table = summary[list(_FIT_COLUMNS)].copy()
    table = table.sort_values("aic")
    for raw, (nice, ndigits) in _FIT_COLUMNS.items():
        table[raw] = table[raw].round(ndigits)
    table = table.rename(columns={raw: nice for raw, (nice, _) in _FIT_COLUMNS.items()})
    table.index.name = "Distribution"
    return table.reset_index()


def _write_table_sheet(xlsx_path: Path, table: pd.DataFrame, log) -> None:
    """把分布拟合表写入工作簿的 Table S10 表,保留其它手工维护的表。"""
    if not xlsx_path.exists():
        log.warning("工作簿不存在,跳过写入 xlsx: %s", xlsx_path)
        return
    # mode="a" + if_sheet_exists="replace":仅新增/替换 Table S10,不动其它 sheet
    with pd.ExcelWriter(
        xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        table.to_excel(writer, sheet_name=_TABLE_SHEET, index=False)
    log.info("已写入分布拟合表到 %s [%s]", xlsx_path.name, _TABLE_SHEET)


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        raise ValueError("cfg 不能为空")
    OmegaConf.resolve(cfg)
    setup_logger_from_hydra(cfg)
    log = get_logger(__name__)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("整合自然重建数据开始;运行配置见 %s", out_dir / ".hydra/config.yaml")

    # 1. 读取筛选后的树轮重建数据(与 natural.ipynb 一致)
    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        start_year=START,
        standardize=True,
    )
    log.info(
        "载入 %d 套重建序列,时间跨度 %s-%s",
        datasets.shape[1],
        datasets.index.min(),
        datasets.index.max(),
    )

    # 2. 贝叶斯整合(combine_reconstructions 内部已打印 nu-hat / R-hat / ESS 诊断)
    combined, _trace = combine_reconstructions(
        datasets,
        uncertainties,
        standardize=True,
        random_seed=cfg.get("random_seed", 42),
    )

    # 3. 导出整合结果与输入不确定性
    tree_ring = Path(cfg.ds.out.tree_ring)
    tree_ring.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tree_ring)
    uncertainties.to_csv(cfg.ds.out.tree_ring_uncertainty)
    log.info("已导出整合数据 -> %s", tree_ring)

    # 4. 分布拟合表(SI Table S10):同时存 csv 与写入工作簿
    fit_table = _build_fit_table(combined["mean"])
    fit_csv = tree_ring.parent / "nwdi_distribution_fit.csv"
    fit_table.to_csv(fit_csv, index=False)
    log.info(
        "分布拟合(按 AIC 排序,最优 = %s):\n%s",
        fit_table.iloc[0]["Distribution"],
        fit_table.to_string(index=False),
    )
    _write_table_sheet(Path(cfg.ds.figs) / "CollMemo_Tables.xlsx", fit_table, log)

    log.info("完成。数据 -> %s ,表格 -> %s", tree_ring.parent, fit_csv.name)


if __name__ == "__main__":
    _main()
