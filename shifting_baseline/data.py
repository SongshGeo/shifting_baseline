#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from fitter import Fitter, get_common_distributions
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files
from omegaconf import DictConfig

from shifting_baseline.constants import (
    END,
    FINAL,
    GRADE_VALUES,
    LEVELS,
    LEVELS_PROB,
    MAP,
    STAGES_BINS,
    START,
    STD_THRESHOLDS,
)
from shifting_baseline.filters import classify
from shifting_baseline.mc import standardize_both
from shifting_baseline.utils.calc import calc_corr, rand_generate_from_std_levels

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike, Region

    from shifting_baseline.utils.types import (
        HistoricalAggregateType,
        Stages,
        ToStdMethod,
    )

from shifting_baseline.utils.log import get_logger

log = get_logger(__name__)
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
    start_year: int = START,
    standardize: bool = True,
    end_year: int = FINAL,
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
        df = pd.read_csv(
            path,
            sep=sep,
            skiprows=lino_1st - 1,
            index_col=0,
            engine="python",
        )
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
        to_std: Optional[ToStdMethod] = None,
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
        self.region = region
        self.to_std = to_std
        # 读取地理空间数据
        self.shp = gpd.read_file(shp_path).dropna(how="any")
        self._symmetrical_level = symmetrical_level
        self.region = region
        self.setup()

    def setup(self):
        if self.region is not None:
            self.shp = self.shp[self.shp["region"] == self.region]
        self._data = self._read_data(self.region)
        # 处理对称等级
        self._setup_level_data(to_level=self.sym, to_std=self.to_std)

    def _setup_level_data(self, to_level: bool, to_std: Optional[ToStdMethod]):
        """处理对称等级和标准化"""
        if to_level:
            self._data = 3 - self._data
        if to_std is None:
            return
        assert to_level, "to_std 必须设置 to_level 同时为 True"
        if to_std == "mapping":
            self._data = self._data.replace(MAP)
        elif to_std == "sampling":
            data = rand_generate_from_std_levels(
                self._data,
                mu=0.0,
                sigma=1.0,
                n_samples=100,
            )
            self._data = pd.DataFrame(
                np.nanmean(data, axis=0),
                index=self._data.index,
                columns=self._data.columns,
            )
            self._std = pd.DataFrame(
                np.nanstd(data, axis=0),
                index=self._data.index,
                columns=self._data.columns,
            )
        else:
            raise ValueError(f"无效的 to_std 方法: {to_std}")

    def get_bounds(
        self,
        lon_name: str = "lon",
        lat_name: str = "lat",
        resolution: float | None = None,
    ) -> dict[str, slice]:
        """获取所有数据点的范围"""
        mins = self.shp.bounds.min()
        maxs = self.shp.bounds.max()

        return {
            lat_name: slice(maxs.maxy, mins.miny, resolution),
            lon_name: slice(mins.minx, maxs.maxx, resolution),
        }

    @property
    def sym(self) -> bool:
        """是否对称等级"""
        return self._symmetrical_level

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

    def _probability_weighted_aggregation(
        self, weights: Optional[dict] = None
    ) -> pd.Series:
        """
        基于概率分布的加权平均聚合

        将每个站点的等级数据转换为连续值时，考虑等级的概率分布，
        使用概率密度函数来估计连续值
        """

        # 获取所有非NA数据
        data_clean = self.data.dropna(how="all")

        if data_clean.empty:
            return pd.Series(index=self.data.index, dtype=float)

        # 设置站点权重
        if weights is None:
            weights = {col: 1.0 for col in data_clean.columns}

        # 标准化权重
        total_weight = sum(weights.get(col, 1.0) for col in data_clean.columns)
        normalized_weights = {
            col: weights.get(col, 1.0) / total_weight for col in data_clean.columns
        }

        result = pd.Series(index=self.data.index, dtype=float)

        for year in data_clean.index:
            year_data = data_clean.loc[year]
            valid_data = year_data.dropna()

            if valid_data.empty:
                result.loc[year] = np.nan
                continue

            # 对每个有效站点，计算概率加权的连续值
            weighted_values = []
            total_weight_sum = 0

            for station in valid_data.index:
                level = valid_data[station]
                if level in MAP:
                    # 使用等级对应的标准差值作为中心
                    center_value = MAP[level]

                    # 使用该等级的概率作为权重
                    level_prob = LEVELS_PROB[LEVELS.index(level)]
                    station_weight = normalized_weights.get(station, 1.0)

                    # 概率加权
                    weighted_value = center_value * level_prob * station_weight
                    weighted_values.append(weighted_value)
                    total_weight_sum += level_prob * station_weight

            if weighted_values:
                result.loc[year] = sum(weighted_values) / total_weight_sum
            else:
                result.loc[year] = np.nan

        return result

    def _simple_mean_aggregation(self, weights: Optional[dict] = None) -> pd.Series:
        """简单平均聚合"""
        from .constants import MAP

        data_std = self.data.replace(MAP)

        if weights is None:
            return data_std.mean(axis=1)
        else:
            # 应用权重
            weighted_data = data_std * pd.Series(weights)
            return weighted_data.sum(axis=1) / sum(weights.values())

    def _median_aggregation(self, weights: Optional[dict] = None) -> pd.Series:
        """中位数聚合"""
        from .constants import MAP

        data_std = self.data.replace(MAP)
        return data_std.median(axis=1)

    def _mode_aggregation(self, weights: Optional[dict] = None) -> pd.Series:
        """众数聚合"""
        from .constants import MAP

        data_std = self.data.replace(MAP)
        return (
            data_std.mode(axis=1).iloc[:, 0]
            if not data_std.empty
            else pd.Series(index=self.data.index)
        )

    def bayesian_aggregation(
        self,
        spatial_correlation: float | np.ndarray | pd.DataFrame = 0.5,
        uncertainty_factor: float = 1.0,
        distance_matrix: np.ndarray | pd.DataFrame | None = None,
        correlation_decay: float = 0.1,
    ) -> pd.Series:
        """
        基于贝叶斯方法的站点数据聚合

        考虑站点间的空间相关性和不确定性，使用贝叶斯组合方法
        来整合多个站点的离散等级数据

        Args:
            spatial_correlation: 站点间空间相关性，可以是：
                - float: 常数相关性系数 (0-1)
                - np.ndarray: 相关性矩阵 (n_stations x n_stations)
                - pd.DataFrame: 相关性矩阵，索引为站点名
            uncertainty_factor: 不确定性因子，用于调整置信区间
            distance_matrix: 站点间距离矩阵，用于计算空间相关性
            correlation_decay: 距离衰减参数，用于基于距离计算相关性

        Returns:
            pd.Series: 贝叶斯聚合的区域连续时间序列
        """
        import numpy as np

        from .constants import LEVELS, LEVELS_PROB, MAP

        data_clean = self.data.dropna(how="all")

        if data_clean.empty:
            return pd.Series(index=self.data.index, dtype=float)

        # 处理空间相关性参数
        correlation_matrix = self._process_spatial_correlation(
            spatial_correlation,
            distance_matrix,
            correlation_decay,
            data_clean.columns,
        )

        result = pd.Series(index=self.data.index, dtype=float)
        uncertainty = pd.Series(index=self.data.index, dtype=float)

        for year in data_clean.index:
            year_data = data_clean.loc[year]
            valid_data = year_data.dropna()

            if valid_data.empty:
                result.loc[year] = np.nan
                uncertainty.loc[year] = np.nan
                continue

            n_stations = len(valid_data)
            station_names = valid_data.index.tolist()

            # 计算每个站点的似然函数
            station_values = []
            station_uncertainties = []

            for station in station_names:
                level = valid_data[station]
                if level in MAP:
                    center_value = MAP[level]
                    level_prob = LEVELS_PROB[LEVELS.index(level)]

                    # 基于等级概率计算不确定性
                    base_uncertainty = 0.5  # 基础标准差
                    level_uncertainty = base_uncertainty * (
                        1 - level_prob
                    )  # 概率越高，不确定性越低

                    station_values.append(center_value)
                    station_uncertainties.append(level_uncertainty)

            if not station_values:
                result.loc[year] = np.nan
                uncertainty.loc[year] = np.nan
                continue

            # 贝叶斯组合
            if n_stations == 1:
                # 单站点情况
                result.loc[year] = station_values[0]
                uncertainty.loc[year] = station_uncertainties[0]
            else:
                # 多站点贝叶斯组合
                # 获取当前有效站点的相关性子矩阵
                station_indices = [
                    list(data_clean.columns).index(name) for name in station_names
                ]
                corr_submatrix = correlation_matrix[
                    np.ix_(station_indices, station_indices)
                ]

                # 计算贝叶斯权重
                weights = self._calculate_bayesian_weights(
                    station_values, station_uncertainties, corr_submatrix
                )

                # 计算后验均值和方差
                posterior_mean = np.average(station_values, weights=weights)

                # 计算后验不确定性（考虑空间相关性）
                posterior_uncertainty = self._calculate_posterior_uncertainty(
                    station_uncertainties, corr_submatrix, weights
                )

                result.loc[year] = posterior_mean
                uncertainty.loc[year] = posterior_uncertainty * uncertainty_factor

        # 存储不确定性信息
        self._aggregation_uncertainty = uncertainty

        return result

    def _process_spatial_correlation(
        self,
        spatial_correlation,
        distance_matrix,
        correlation_decay,
        station_names,
    ):
        """处理空间相关性参数，生成相关性矩阵"""

        n_stations = len(station_names)

        if isinstance(spatial_correlation, (int, float)):
            # 常数相关性
            if spatial_correlation == 0:
                return np.eye(n_stations)  # 完全独立
            else:
                # 创建常数相关性矩阵
                corr_matrix = np.full((n_stations, n_stations), spatial_correlation)
                np.fill_diagonal(corr_matrix, 1.0)  # 对角线为1
                return corr_matrix

        elif isinstance(spatial_correlation, (np.ndarray, pd.DataFrame)):
            # 提供相关性矩阵
            if isinstance(spatial_correlation, pd.DataFrame):
                # 确保索引和列名匹配
                corr_matrix = spatial_correlation.reindex(
                    index=station_names, columns=station_names
                ).values
            else:
                corr_matrix = spatial_correlation

            # 确保矩阵是对称的且对角线为1
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix, 1.0)
            return corr_matrix

        elif distance_matrix is not None:
            # 基于距离矩阵计算相关性
            if isinstance(distance_matrix, pd.DataFrame):
                dist_matrix = distance_matrix.reindex(
                    index=station_names, columns=station_names
                ).values
            else:
                dist_matrix = distance_matrix

            # 使用指数衰减函数计算相关性
            corr_matrix = np.exp(-correlation_decay * dist_matrix)
            np.fill_diagonal(corr_matrix, 1.0)
            return corr_matrix

        else:
            # 默认：基于距离的简单相关性
            # 这里可以添加基于地理距离的默认计算
            return np.eye(n_stations)  # 默认独立

    def _calculate_bayesian_weights(
        self,
        station_values: np.ndarray,
        station_uncertainties: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """计算贝叶斯权重"""
        n_stations = len(station_values)

        # 构建协方差矩阵
        # 对角线元素是各站点的方差
        variances = np.array(station_uncertainties) ** 2

        # 非对角线元素考虑空间相关性
        cov_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * corr_matrix
        np.fill_diagonal(cov_matrix, variances)

        # 计算贝叶斯权重
        try:
            # 使用协方差矩阵的逆来计算权重
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n_stations)
            weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)

            # 确保权重为正且和为1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)

        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用简单平均
            weights = np.ones(n_stations) / n_stations

        return weights

    def __repr__(self) -> str:
        return f"<Historical Records: {self.data.shape}>"

    def __getattr__(self, item: str):
        """获取属性"""
        try:
            return getattr(self._data, item)
        except AttributeError:
            raise AttributeError(
                f"HistoricalRecords object or its data has no attribute '{item}'"
            )

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
        to_int: bool = True,
        weights: pd.Series | None = None,
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
            to_int: 是否转换为整数
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
                result = data.mean(axis=1)
            elif how == "median":
                result = data.median(axis=1)
            elif how == "mode":
                result = data.mode(axis=1)[0]
            elif how == "weighted_mean":
                assert weights is not None, "带权重的平均值需要提供权重"
                result = self.weighted_mean(weights)
            elif how == "probability_weighted":
                result = self._probability_weighted_aggregation(weights, **kwargs)
            elif how == "bayesian":
                result = self.bayesian_aggregation(weights, **kwargs)
            else:
                raise ValueError(f"无效的聚合方法: {how}")
        if name is None:
            if isinstance(data, pd.Series) and data.name:
                name = data.name
            else:
                name = str(how).lower()
        # 是否转换为整数
        if to_int:
            # 确保结果是数值类型
            if not pd.api.types.is_numeric_dtype(result):
                result = pd.to_numeric(result, errors="coerce")
            result = classify(result, handle_na="skip")
        result.name = "history_" + name
        if inplace:
            self.data = result.astype(float)
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
            pd.DataFrame | tuple[pd.Series, pd.Series]: 合并后的数据集，如果 split 为 True，则返回两个 Series，第一个是历史数据，第二个是其它数据
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

    def weighted_mean(self, weights: pd.Series) -> pd.Series:
        """加权平均"""
        data = self.data.T
        levels = []
        for year in data.columns:
            mask = data[year].notnull()
            if mask.sum() == 0:
                level = np.nan
            else:
                w = weights.loc[mask] / weights.loc[mask].sum()
                level = (data.loc[mask, year] * w).sum()
            levels.append(level)
        return pd.Series(levels, index=data.columns, name="levels")


def load_validation_data(data_path: PathLike, resolution: float = 0.25) -> xr.DataArray:
    """加载验证数据

    Args:
        data_path: 数据路径

    Returns:
        pd.DataFrame: 验证数据
    """
    summer_precip = rxr.open_rasterio(data_path)
    return summer_precip.rio.reproject(
        dst_crs="EPSG:4326",
        resolution=resolution,
    )


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalRecords]:
    """读取自然和历史数据，以及不确定性"""
    start_year = START
    end_year = END
    log.info("加载自然数据 [%s-%s]...", start_year, end_year)
    log.debug("数据路径: %s", cfg.ds.noaa)
    log.debug("数据包括: %s", cfg.ds.includes)
    if cfg.recalculate_data:
        log.info("重新计算自然数据 ...")
        datasets, uncertainties = load_nat_data(
            folder=cfg.ds.noaa,
            includes=cfg.ds.includes,
            index_name="year",
            start_year=start_year,
        )
    else:
        log.info("从文件加载处理后的自然数据 ...")
        datasets = pd.read_csv(cfg.ds.out.tree_ring, index_col=0)
        uncertainties = pd.read_csv(cfg.ds.out.tree_ring_uncertainty, index_col=0)
    log.info("加载历史数据 ...")
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
        to_std=cfg.to_std,
    )
    return datasets, uncertainties, history
