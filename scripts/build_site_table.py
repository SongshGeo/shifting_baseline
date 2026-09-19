#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Build SI Table S2: historical-archive sites and their location.

Lists the 32 study-region sites with their coordinates, taken from the site layer
in ``cfg.ds.atlas.shp`` (the atlas authors' site centres). Only one validation
correlation is reported in the paper (the regional series, results1), so no
per-site r/p is listed here. The sites are laid out in two side-by-side column
blocks, as in the manuscript, and written to sheet "Table S2" of
``<ds.figs>/CollMemo_Tables.xlsx``.

Run:
  uv run python scripts/build_site_table.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf

from shifting_baseline.data import HistoricalRecords
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


def _build_site_table(sites: pd.DataFrame) -> pd.DataFrame:
    assert set(sites.index) == set(
        _SITE_ORDER
    ), f"站点图层与 _SITE_ORDER 不一致: {set(sites.index) ^ set(_SITE_ORDER)}"
    sites = sites.loc[_SITE_ORDER]
    table = pd.DataFrame(
        {
            "Index": np.arange(len(sites)),
            "Name": sites.index,
            "Longitude": [_dms(v, "E", "W") for v in sites["lon"]],
            "Latitude": [_dms(v, "N", "S") for v in sites["lat"]],
        }
    )
    # 两栏并排排版（与手稿原表一致）：前一半站点在左，后一半在右
    half = int(np.ceil(len(table) / 2))
    left = table.iloc[:half].reset_index(drop=True)
    right = table.iloc[half:].reset_index(drop=True)
    return pd.concat([left, right], axis=1)


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
    log.info("站点数: %d", len(sites))
    write_table_sheet(cfg.ds.figs, _build_site_table(sites), _TABLE_SHEET)


if __name__ == "__main__":
    _main()
