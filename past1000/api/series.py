#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/


"""
时间序列数据
"""

from functools import cached_property
from pathlib import Path
from typing import Literal, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from past1000.api.io import PathLike

Region: TypeAlias = Literal[
    "华北地区",
    "西北内陆区",
    "西南地区",
    "东北地区",
    "华南地区",
    "长江中下游地区",
    "青藏高原区",
]


def classify_by_std(
    data: np.ndarray | pd.Series | pd.DataFrame, thresholds: list[int] | None = None
) -> np.ndarray:
    """使用标准差法划分等级

    Args:
        data: 径流时间序列数据
        thresholds: 标准差倍数阈值，默认[-2, -1, 1, 2]

    Returns:
        等级序列(1-5):
        1 - 极端低值 (<-2σ)
        2 - 低值 (-2σ ~ -1σ)
        3 - 正常 (-1σ ~ 1σ)
        4 - 高值 (1σ ~ 2σ)
        5 - 极端高值 (>2σ)
    """
    if thresholds is None:
        thresholds = [-2, -1, 1, 2]
    if isinstance(data, pd.DataFrame):
        df = pd.DataFrame()
        for col in data.columns:
            df[col] = classify_by_std(data[col])
        return df
    # 标准化数据
    z_scores = stats.zscore(data)
    levels = np.full_like(data, 3, dtype=int)  # 默认为正常值(3)

    # 从两端向中间划分
    levels[z_scores < thresholds[0]] = 5  # 极端低值
    levels[(z_scores >= thresholds[0]) & (z_scores < thresholds[1])] = 4  # 低值
    levels[(z_scores > thresholds[2]) & (z_scores <= thresholds[3])] = 2  # 高值
    levels[z_scores > thresholds[3]] = 1  # 极端高值
    if isinstance(data, pd.Series):
        levels = pd.Series(levels, index=data.index)
    return levels


class HistoricalRecords:
    """历史记录数据"""

    def __init__(
        self,
        shp_path: PathLike,
        data_path: PathLike,
        region: Region = "华北地区",
    ):
        self.shp_path = Path(shp_path)
        self.data_path = Path(data_path)
        # 读取数据
        self.shp = gpd.read_file(shp_path).dropna(how="any")
        self.shp = self.shp[self.shp["region"] == region]
        # TODO: 这里读取不同地区的数据还需要实现
        self._data = pd.read_excel(
            data_path,
            sheet_name=region,
            index_col=0,
            header=1,
        ).replace(0, pd.NA)

    @cached_property
    def data(self) -> pd.DataFrame:
        """根据地区筛选后的历史记录数据"""
        return self._data

    def __getattr__(self, item: str):
        return getattr(self.data, item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data.iloc[index]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item: str):
        return item in self.data["region"].values
