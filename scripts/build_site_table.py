#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Build SI Table S2: historical-archive sites, their location and validation r/p.

Lists the 32 study-region sites with their coordinates, taken from the site layer
in ``cfg.ds.atlas.shp`` (the atlas authors' site centres), together with the
nearest-grid Pearson r, two-sided p and n behind Figure 2c. The r/p come from
``results.site_corr_table``, i.e. the same computation as ``results1``'s
``n_sites_sig005``, so the figure caption's "exact r and p in Supplementary
Information Table S2" and the main-text site count cannot drift apart.

Written to sheet "Table S2" of ``<ds.figs>/CollMemo_Tables.xlsx``.

Run:
  uv run python scripts/build_site_table.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf

from shifting_baseline.data import HistoricalRecords
from shifting_baseline.results import site_corr_table
from shifting_baseline.utils.io import write_table_sheet
from shifting_baseline.utils.log import get_logger, setup_logger_from_hydra

_TABLE_SHEET = "Table S2"
# 手稿原表的站点顺序（按省份自西向东），SI 表序号沿用此顺序
_SITE_ORDER = [
    "Lanzhou",
    "Tianshui",
    "Yinchuan",
    "Yulin",
    "Yan'an",
    "Baoji",
    "Xi'an",
    "Duolun",
    "Tangshan",
    "Beijing",
    "Tianjin",
    "Baoding",
    "Cangzhou",
    "Shijiazhuang",
    "Handan",
    "Dezhou",
    "Ji'nan",
    "Laiyang",
    "Linyi",
    "Heze",
    "Datong",
    "Taiyuan",
    "Linfen",
    "Changzhi",
    "Anyang",
    "Zhengzhou",
    "Luoyang",
    "Nanyang",
    "Fuyang",
    "Bengbu",
    "Pingliang",
    "Xuzhou",
]


def _dms(value: float, pos: str, neg: str) -> str:
    """Decimal degrees → 103°13′49″E."""
    total = round(abs(value) * 3600)
    deg, rem = divmod(total, 3600)
    minute, second = divmod(rem, 60)
    return f"{deg}°{minute}′{second}″{pos if value >= 0 else neg}"


def _fmt_p(value: float) -> str:
    """Two-sided p, 3 dp; below that report the threshold (Nature statistics style)."""
    return "$< 0.001$" if value < 0.001 else f"{value:.3f}"


def _build_site_table(sites: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    assert set(sites.index) == set(
        _SITE_ORDER
    ), f"站点图层与 _SITE_ORDER 不一致: {set(sites.index) ^ set(_SITE_ORDER)}"
    sites = sites.loc[_SITE_ORDER]
    stats = stats.loc[_SITE_ORDER]
    return pd.DataFrame(
        {
            "Index": np.arange(len(sites)),
            "Name": sites.index,
            "Longitude": [_dms(v, "E", "W") for v in sites["lon"]],
            "Latitude": [_dms(v, "N", "S") for v in sites["lat"]],
            "$r$": [f"{v:.2f}" for v in stats["r"]],
            "$p$": [_fmt_p(v) for v in stats["p"]],
            "$n$": stats["n"].to_numpy(),
        }
    )


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        raise ValueError("cfg 不能为空")
    OmegaConf.resolve(cfg)
    setup_logger_from_hydra(cfg)
    log = get_logger(__name__)

    sites = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp, data_path=cfg.ds.atlas.file
    ).shp.set_index("name_en")[["lon", "lat"]]
    stats = site_corr_table(cfg)
    log.info("站点数: %d; p<0.05 的站点数: %d", len(sites), int((stats["p"] < 0.05).sum()))
    write_table_sheet(cfg.ds.figs, _build_site_table(sites, stats), _TABLE_SHEET)


if __name__ == "__main__":
    _main()
