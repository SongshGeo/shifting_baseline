#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Export the integrated N-WDI and H-WDI series as Supplementary Data 1.

Writes one workbook, ``<ds.longform>/CollMemo_Supplementary_Data_1.xlsx``, with a
"README" sheet (column definitions) and a "WDI series" sheet holding, per year
(1470–2000 CE):

- the N-WDI posterior summary (mean, SD, 94% HDI) and its five-level class;
- the regional H-WDI (mean of the standardised site grades, 1470–1900 CE), its
  five-level class, and the number of sites with a record that year.

Both series come from :func:`shifting_baseline.data.load_data` with the same
settings as the manuscript (``agg_method``, ``to_std``, ``random_seed``), and the
classes use the same :func:`shifting_baseline.filters.classify` thresholds, so
the file matches the numbers in ``results.json``.

Run:
  uv run python scripts/export_supplementary_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf

from shifting_baseline.constants import END, FINAL, STAGE1
from shifting_baseline.data import load_data
from shifting_baseline.filters import classify
from shifting_baseline.utils.log import get_logger, setup_logger_from_hydra

_README = [
    ("year", "Calendar year (CE)."),
    (
        "nwdi_mean",
        "Natural-proxy Wet/Dry Index (N-WDI): posterior mean of the latent standardised anomaly from the Bayesian integration of tree-ring reconstructions (Supplementary Note 2). Positive = wet.",
    ),
    ("nwdi_sd", "Posterior standard deviation of the N-WDI."),
    ("nwdi_hdi_3", "Lower bound of the 94% highest-density interval of the N-WDI."),
    ("nwdi_hdi_97", "Upper bound of the 94% highest-density interval of the N-WDI."),
    (
        "nwdi_class",
        "Five-level class of nwdi_mean (-2 severe dry, -1 moderate dry, 0 normal, 1 moderate wet, 2 severe wet; thresholds ±0.33 and ±1.17).",
    ),
    (
        "hwdi",
        "Historical Wet/Dry Index (H-WDI): mean across the 32 North China site columns of the Atlas of standardised archival grades (1470–1900 CE). Positive = wet. Blank where no site has a record.",
    ),
    (
        "hwdi_class",
        "Five-level class of hwdi, using the same thresholds as nwdi_class.",
    ),
    (
        "hwdi_n_sites",
        "Number of Atlas site columns (of 32) with an archival record in that year.",
    ),
]


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        raise ValueError("cfg 不能为空")
    OmegaConf.resolve(cfg)
    setup_logger_from_hydra(cfg)
    log = get_logger(__name__)

    combined, _, history = load_data(cfg)
    years = pd.RangeIndex(STAGE1, FINAL + 1, name="year")

    # N-WDI：贝叶斯整合后验摘要
    nwdi = combined.reindex(years)
    # H-WDI：与 compute_results2 相同——站点标准化等级的区域平均（连续值）
    history.setup()
    sites = history.data.loc[STAGE1:END]
    hwdi = history.aggregate(how=cfg.agg_method, to_int=False).loc[STAGE1:END]

    table = pd.DataFrame(
        {
            "nwdi_mean": nwdi["mean"],
            "nwdi_sd": nwdi["sd"],
            "nwdi_hdi_3": nwdi["hdi_3%"],
            "nwdi_hdi_97": nwdi["hdi_97%"],
            "nwdi_class": classify(nwdi["mean"], handle_na="skip"),
            "hwdi": hwdi.astype(float),
            "hwdi_class": classify(hwdi.astype(float), handle_na="skip"),
            "hwdi_n_sites": sites.notna().sum(axis=1).astype(int),
        },
        index=years,
    )
    table["hwdi_n_sites"] = table["hwdi_n_sites"].astype("Int64")

    out = Path(cfg.ds.longform) / "CollMemo_Supplementary_Data_1.xlsx"
    readme = pd.DataFrame(_README, columns=["Column", "Description"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        table.reset_index().to_excel(writer, sheet_name="WDI series", index=False)
    log.info(
        "已导出 %d 年 (N-WDI %d 年, H-WDI %d 年) -> %s",
        len(table),
        table["nwdi_mean"].notna().sum(),
        table["hwdi"].notna().sum(),
        out,
    )


if __name__ == "__main__":
    _main()
