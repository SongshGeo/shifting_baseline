#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from fitter import Fitter, get_common_distributions
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files
from omegaconf import DictConfig

from past1000.calibration import MismatchReport
from past1000.constants import GRADE_VALUES, STAGES_BINS, STD_THRESHOLDS
from past1000.mc import standardize_both
from past1000.utils.calc import calc_corr

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike, Region

    from past1000.utils.types import HistoricalAggregateType, Stages

log = logging.getLogger(__name__)
# 常用的分布
common_distributions = get_common_distributions()
common_distributions.append("t")


def check_distribution(
    data: pd.Series | pd.DataFrame,
    only_best: bool = True,
) -> pd.DataFrame | pd.Series:
    """检查数据分布"""
    if isinstance(data, pd.DataFrame):
        results = []
        for col in data.columns:
            best = check_distribution(data[col], only_best=True)
            best["best_dist"] = best.name
            best.name = col
            results.append(best)
        results = pd.concat(results, axis=1).T
        return results
    f = Fitter(data.dropna().values, distributions=common_distributions)
    f.fit()
    summary = f.summary(clf=False, plot=False)
    best = summary.iloc[0]
    log.info("最佳分布: %s", best)
    if only_best:
        return best
    return summary


def load_nat_data(
    folder: str,
    includes: list[str],
    index_name: str = "year",
    start_year: int = 1000,
    standardize: bool = True,
    end_year: int = 2021,
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
    includes_str = ", ".join(includes)
    log.info("从 %s 加载自然数据: %s", folder, includes_str)
    log.debug("年份范围: %s-%s", start_year, end_year)

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
        region: Region | None = "华北地区",
        symmetrical_level: bool = True,
    ):
        """
        历史千年旱涝记录数据，参考：
        https://news.fudan.edu.cn/2024/0420/c5a140208/page.htm

        Args:
            shp_path (PathLike): 矢量图路径
            data_path (PathLike): 数据路径
            region (Region): 地区，可供选择的有：
                - 华北地区
                - 东北地区
                - 华东地区
                - 华中地区
                - 华南地区
                - 西南地区
                - 西北地区
                如果为 None，则读取所有地区.
            symmetrical_level (bool): 是否对称等级.
                如果为 True，则将数据转换为对称等级.
                如果为 False，则不进行转换.
        """
        self.shp_path = Path(shp_path)
        self.data_path = Path(data_path)
        # 读取地理空间数据
        self.shp = gpd.read_file(shp_path).dropna(how="any")
        if region is not None:
            self.shp = self.shp[self.shp["region"] == region]
        self._data = self._read_data(region)
        # 处理对称等级
        if symmetrical_level:
            self._data = 3 - self._data
        self._sym_level = symmetrical_level
        assert isinstance(self._sym_level, bool), "symmetrical_level 必须是布尔值"

    @property
    def sym(self) -> bool:
        """是否对称等级"""
        return self._sym_level

    @property
    def data(self) -> pd.DataFrame | pd.Series:
        """根据地区筛选后的历史记录数据"""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame | pd.Series):
        """设置数据"""
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise ValueError("数据必须是DataFrame或Series")
        self._data = value

    def get_time_slice(self, stage: Stages) -> slice:
        """Convert stage input to a time slice.

        This method accepts multiple convenient notations and returns a slice object:
        - Integer stage index: 1, 2, 3, 4
        - Stage slice: ``slice(1, 3)`` meaning stages 1 through 2 (inclusive of
          the end year of stage 2). If ``start`` or ``stop`` is ``None``, they
          default to 1 and 4 respectively. ``step`` is not supported.
        - String forms:
          - ``"stage1"``, ``"stage2"``, ...
          - ``"1:3"`` or ``"1-3"`` for stages
          - ``"stage1:stage3"`` or ``"stage1-stage3"``
          - ``"1000:1469"`` or ``"1000-1469"`` for explicit year ranges
          - ``"all"`` / ``"full"`` / ``"total"`` for the whole series

        Returns a slice object representing the time range.

        Examples:
            >>> history.get_time_slice(1)  # slice(1000, 1469)
            >>> history.get_time_slice(slice(1, 2))  # slice(1000, 1669)
            >>> history.get_time_slice("1:4")  # slice(1000, 2010)
            >>> history.get_time_slice("stage3")  # slice(1669, 1889)
            >>> history.get_time_slice("1000:2010")  # slice(1000, 2010)
        """
        bins = STAGES_BINS

        def _bounds_from_stage_indices(
            start_stage: int, stop_stage: int
        ) -> tuple[int, int]:
            statement = (
                f"Stage index out of range: {start_stage}..{stop_stage}. "
                f"Valid is 1..{len(bins) - 1}"
            )
            if not (
                1 <= start_stage <= len(bins) - 1 and 1 <= stop_stage <= len(bins) - 1
            ):
                raise ValueError(statement)
            if start_stage > stop_stage:
                start_stage, stop_stage = stop_stage, start_stage
            return bins[start_stage - 1], bins[stop_stage]

        # Integer stage
        if isinstance(stage, int):
            start_year, end_year = _bounds_from_stage_indices(stage, stage)
            return slice(start_year, end_year)

        # Slice: interpret as stage indices if within 1..4, otherwise pass through as year slice
        if isinstance(stage, slice):
            if stage.step is not None:
                raise ValueError("Slice step is not supported for stage selection")

            # Detect stage-style slice: values within valid stage indices or None
            start_is_stage = stage.start is None or (
                isinstance(stage.start, int) and 1 <= stage.start <= len(bins) - 1
            )
            stop_is_stage = stage.stop is None or (
                isinstance(stage.stop, int) and 1 <= stage.stop <= len(bins) - 1
            )

            if start_is_stage and stop_is_stage:
                start_stage = 1 if stage.start is None else stage.start
                stop_stage = (len(bins) - 1) if stage.stop is None else stage.stop
                start_year, end_year = _bounds_from_stage_indices(
                    start_stage, stop_stage
                )
                return slice(start_year, end_year)

            # Otherwise treat as a raw year slice
            return stage

        # String patterns
        if isinstance(stage, str):
            s = stage.strip().lower()
            if s in {"all", "full", "total", "whole"}:
                return slice(None)  # slice(None) selects all data

            # "stageN"
            m = re.fullmatch(r"stage\s*(\d)", s)
            if m:
                n = int(m.group(1))
                start_year, end_year = _bounds_from_stage_indices(n, n)
                return slice(start_year, end_year)

            # "stageA:stageB" or "stageA-stageB"
            m = re.fullmatch(r"stage\s*(\d)\s*[:\-]\s*stage\s*(\d)", s)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                start_year, end_year = _bounds_from_stage_indices(a, b)
                return slice(start_year, end_year)

            # "A:B" or "A-B" where A,B are stage indices (single digit)
            m = re.fullmatch(r"(\d)\s*[:\-]\s*(\d)", s)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                start_year, end_year = _bounds_from_stage_indices(a, b)
                return slice(start_year, end_year)

            # Explicit years "YYYY:YYYY" or "YYYY-YYYY"
            m = re.fullmatch(r"(\d{3,4})\s*[:\-]\s*(\d{3,4})", s)
            if m:
                start_year, end_year = int(m.group(1)), int(m.group(2))
                if start_year > end_year:
                    start_year, end_year = end_year, start_year
                return slice(start_year, end_year)

            raise ValueError(
                f"Invalid stage expression: {stage}. Expected like 'stage1', '1:3', or '1000-1469'."
            )

        raise TypeError("stage must be int, slice, or str representing stage/years")

    def select_data(self, time_slice: slice) -> pd.DataFrame | pd.Series:
        """Select data using a time slice.

        Args:
            time_slice: A slice object representing the time range

        Returns:
            The selected data subset with the same type as self.data
        """
        return self.data.loc[time_slice]

    def period(self, stage: Stages) -> pd.DataFrame | pd.Series:
        """Select data by historical stage(s) or explicit year range.

        This is a convenience method that combines get_time_slice() and select_data().
        See get_time_slice() for detailed parameter documentation.

        Examples:
            >>> history.period(1)  # stage 1
            >>> history.period(slice(1, 2))  # stages 1-2
            >>> history.period("1:4")  # stages 1-4
            >>> history.period("stage3")
            >>> history.period("1000:2010")  # explicit years
        """
        time_slice = self.get_time_slice(stage)
        return self.select_data(time_slice)

    def _read_data(self, region: Region) -> pd.DataFrame:
        """读取数据，并统一为逐年索引"""
        full_index = np.arange(1000, 2021)
        df = pd.read_excel(
            self.data_path,
            sheet_name=region,
            index_col=0,
            header=1,
        ).replace(0, pd.NA)
        df.index.name = "year"
        return df.reindex(full_index)

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
        return f"<Historical Records: {self.data.shape}>"

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

    def aggregate(
        self,
        how: HistoricalAggregateType | Callable = "mean",
        inplace: bool = False,
        name: str | None = None,
        **kwargs,
    ) -> pd.Series | "HistoricalRecords":
        """转换为Series

        If ``self.data`` is already a Series, it will be returned (optionally
        interpolated/renamed). If it is a DataFrame, it will be aggregated along
        rows (time) using ``how`` into a Series.

        Args:
            how: 聚合方法，可以是字符串或函数
            inplace: 是否在原地修改数据
            name: 结果的名称
            **kwargs: 传递给函数的参数

        Examples:
            >>> history.aggregate('mean')
            >>> history.aggregate('median')
            >>> history.aggregate('mode')
            >>> history.aggregate(lambda x: x.mean(axis=1).astype(float).round(0))
            >>> history.aggregate(lambda x: x.mean(axis=1).astype(float).round(0), inplace=True)

        Raises:
            ValueError: 如果聚合方法无效

        Returns:
            pd.Series: 聚合后的结果
            HistoricalRecords: 如果 inplace 为 True，则返回自身
        """
        # data = self.rescale_to_std()
        data = self.data
        if isinstance(data, pd.Series):
            result = data.copy()
        else:
            if how == "mean":
                result = data.mean(axis=1).astype(float).round(0)
            elif how == "median":
                result = data.median(axis=1)
            elif how == "mode":
                result = data.mode(axis=1)[0]
            elif callable(how):
                result = how(data, **kwargs)
            else:
                raise ValueError(f"无效的聚合方法: {how}")
        if name is None:
            if isinstance(data, pd.Series) and data.name:
                name = data.name
            else:
                name = str(how).lower()
        result.name = "history_" + name
        if inplace:
            self.data = result
            return self
        return result

    @overload
    def merge_with(
        self,
        other: pd.Series | pd.DataFrame,
        time_range: Stages = "all",
        split: Literal[False] = False,
    ) -> pd.DataFrame:
        ...

    @overload
    def merge_with(
        self,
        other: pd.Series | pd.DataFrame,
        time_range: Stages = "all",
        split: Literal[True] = True,
    ) -> tuple[pd.Series, pd.Series]:
        ...

    def merge_with(
        self,
        other: pd.Series | pd.DataFrame,
        time_range: Stages = "all",
        split: bool = False,
    ) -> pd.DataFrame | tuple[pd.Series, pd.Series]:
        """合并两个数据集

        Args:
            other: 要合并的数据集
            time_range: 时间范围
            split: 是否拆分

        Returns:
            pd.DataFrame | tuple[pd.Series, pd.Series]: 合并后的数据集，如果 split 为 True，则返回两个 Series
        """
        # 获取历史记录数据
        data = self.period(time_range)
        # 合并两个数据集，并返回一个DataFrame
        df = pd.merge(
            left=data,
            right=other,
            left_index=True,
            right_index=True,
            how="inner",
        )
        if split:
            return df.iloc[:, 0], df.iloc[:, 1]
        return df

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


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalRecords]:
    """读取自然和历史数据，以及不确定性"""
    start_year = cfg.years.start
    end_year = cfg.years.end
    log.info("加载自然数据 [%s-%s]...", start_year, end_year)
    log.debug("数据路径: %s", cfg.ds.noaa)
    log.debug("数据包括: %s", cfg.ds.includes)
    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        index_name="year",
        start_year=start_year,
    )
    log.info("加载历史数据 ...")
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
    )
    return datasets, uncertainties, history
