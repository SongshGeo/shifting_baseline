#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/


"""
时间序列数据
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from past1000.ci.corr import calc_corr

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike

# 常量定义
GRADE_VALUES = [5, 4, 3, 2, 1]  # 原始等级值
STD_THRESHOLDS = [-1.17, -0.33, 0, 0.33, 1.17]  # 标准差阈值

Region: TypeAlias = Literal[
    "华北地区",
    "西北内陆区",
    "西南地区",
    "东北地区",
    "华南地区",
    "长江中下游地区",
    "青藏高原区",
]

STAGES_BINS = [1469, 1649, 1949, 2021]


def classify_by_std(
    data: np.ndarray | pd.Series | pd.DataFrame, thresholds: list[int] | None = None
) -> np.ndarray:
    """使用标准差法划分等级

    Args:
        data: 径流时间序列数据
        thresholds: 标准差倍数阈值，默认[-2, -1, 1, 2]

    Returns:
        等级序列(1-5):
        1 - 极端低值 (<-1.17σ)
        2 - 低值 (-1.17σ ~ -0.33σ)
        3 - 正常 (-0.33σ ~ 0.33σ)
        4 - 高值 (0.33σ ~ 1.17σ)
        5 - 极端高值 (>1.17σ)
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
        symmetrical_level: bool = True,
    ):
        self.shp_path = Path(shp_path)
        self.data_path = Path(data_path)
        # 读取数据
        self.shp = gpd.read_file(shp_path).dropna(how="any")
        self.shp = self.shp[self.shp["region"] == region]
        # TODO: 这里读取不同地区的数据还需要实现
        self._data = self._read_data(region)
        if symmetrical_level:
            self._data = 3 - self._data

    def _read_data(self, region: Region):
        full_index = np.arange(1000, 2021)
        df = pd.read_excel(
            self.data_path,
            sheet_name=region,
            index_col=0,
            header=1,
        ).replace(0, pd.NA)
        df.index.name = "year"
        return df.reindex(full_index)

    @property
    def data(self) -> pd.DataFrame:
        """根据地区筛选后的历史记录数据"""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame | pd.Series):
        """设置数据"""
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise ValueError("数据必须是DataFrame或Series")
        self._data = value

    @property
    def stage4(self) -> pd.DataFrame:
        """第四阶段的数据"""
        return self.data.loc[1949:2010]

    def rescale_to_std(self, target_std=None):
        """
        将原始等级值重新映射到标准差倍数

        Args:
            target_std: 目标标准差倍数，默认使用预定义的 STD_THRESHOLDS

        Returns:
            DataFrame/Series: 映射到标准差尺度的数据
        """
        if target_std is None:
            target_std = STD_THRESHOLDS
        return self.data.replace(dict(zip(GRADE_VALUES, target_std)))

    def __repr__(self) -> str:
        print("<Historical Records>")
        return repr(self.data)

    def __getattr__(self, item: str):
        return getattr(self._data, item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data.iloc[index]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item: str):
        return item in self.data["region"].values

    @property
    def is_series(self) -> bool:
        """当前数据是否已经转化为 Series"""
        return isinstance(self.data, pd.Series)

    def get_series(self, col: Optional[str] = None) -> pd.Series:
        """获取序列"""
        if col is not None:
            return self.data[col]
        if not self.is_series:
            raise TypeError("尚未转化为 pd.Series")
        return self.data

    def to_series(
        self,
        how: Literal["mean", "median"] = "mean",
        interpolate: str | None = None,
        inplace: bool = False,
        name: str | None = None,
        **kwargs,
    ) -> pd.Series | None:
        """转换为Series"""
        # data = self.rescale_to_std()
        data = self.data
        if how == "mean":
            result = data.mean(axis=1)
        elif how == "median":
            result = data.median(axis=1)
        elif how == "mode":
            result = data.mode(axis=1)[0]
        else:
            raise ValueError(f"无效的聚合方法: {how}")
        if interpolate:
            result = result.interpolate(method=interpolate, **kwargs)
        if name is None:
            func_name = str(how).lower()
            name = f"{func_name}_{interpolate}" if interpolate else func_name
        result.name = name
        if inplace:
            self.data = result
            return None
        return result

    def merge_with(
        self, other: pd.Series | pd.DataFrame, time_range=None
    ) -> pd.DataFrame:
        """合并两个数据集"""
        if time_range is None:
            time_range = self.data.index
            data = self.data.copy()
        else:
            data = self.data.reindex(time_range)
        return pd.concat([data, other.reindex(time_range)], axis=1)

    def corr_with(
        self,
        arr2: pd.Series,
        col: Optional[str] = None,
        how: Literal["pearson", "kendall", "spearman"] = "pearson",
    ) -> Tuple[float, float, int]:
        """历史记录和其它数据之间的相关系数

        Args:
            arr2: 要比较的另一个序列
            col: 要使用的列名，如果为None则使用默认列

        Returns:
            Tuple[float, float, int]: (相关系数, p值, 有效样本数)
        """
        arr1 = self.get_series(col=col)
        return calc_corr(arr1, arr2, how)
