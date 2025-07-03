#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files

from past1000.api.mc import standardize_both
from past1000.ci.corr import calc_corr
from past1000.core.constants import GRADE_VALUES, STD_THRESHOLDS, Region

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike


def load_nat_data(
    folder: str,
    includes: list[str],
    index_name: str = "year",
    start_year: int = 1000,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载自然数据，并进行标准化处理，返回数据和不确定性

    Args:
        folder (str): 数据文件夹，其中包含多个
        includes (list[str]): 包含的字符串
        index_name (str): 索引名称
        start_year (int): 开始年份
        standardize (bool): 是否标准化
    Returns:
        datasets (pd.DataFrame): 数据
        uncertainties (pd.DataFrame): 不确定性
    """
    datasets = []
    uncertainties = []
    # 匹配包含included中任意一个字符串的文件
    pattern = f"(?:{'|'.join(map(re.escape, includes))})"
    # 读取树轮重建数据
    paths = get_files(folder, iter_subdirs=True, wildcard="*.txt")
    paths = filter_files(paths, pattern)
    for path in paths:
        lino_1st = find_first_uncommented_line(path)
        sep = r"\t" if check_tab_sep(path) else r"\s+"
        df = pd.read_csv(path, sep=sep, skiprows=lino_1st - 1, index_col=0)
        df.index.name = index_name
        if standardize:
            # TODO：这里怎么没有输入uncertainty？
            ser, uncertainty = standardize_both(df.iloc[:, 0])
            uncertainty.name = path.stem
        else:
            ser = df.iloc[:, 0]
            uncertainty = pd.Series(
                np.ones(shape=ser.shape) * ser.std(), index=ser.index
            )
        ser.name = path.stem
        datasets.append(ser)
        uncertainties.append(uncertainty)
    datasets = pd.concat(datasets, axis=1).sort_index().loc[start_year:]
    uncertainties = pd.concat(uncertainties, axis=1).sort_index().loc[start_year:]
    return datasets, uncertainties


class HistoricalRecords:
    """历史记录数据

    Args:
        shp_path (PathLike): 矢量图路径
        data_path (PathLike): 数据路径
        region (Region): 地区
        symmetrical_level (bool): 是否对称等级

    Returns:
        HistoricalRecords: 历史记录数据
    """

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
