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
import xarray as xr
from fitter import Fitter, get_common_distributions
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files
from omegaconf import DictConfig

from shifting_baseline.constants import (
    END,
    FINAL,
    MAP,
    NEUTRAL_GRADE,
    STAGES_BINS,
    START,
)
from shifting_baseline.filters import classify
from shifting_baseline.utils.calc import calc_corr, rand_generate_from_std_levels

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike

    from shifting_baseline.utils.types import (
        HistoricalAggregateType,
        Region,
        Stages,
        ToStdMethod,
    )

from shifting_baseline.utils.log import get_logger

# 使用主logger，避免重复设置
log = get_logger()
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
    from shifting_baseline.mc import standardize_both

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
        random_seed: int | None = 42,
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
        self.random_seed = random_seed
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
            log.info("处理为对称等级 ...")
            self._data = NEUTRAL_GRADE - self._data
        if to_std is None:
            return
        assert to_level, "to_std 必须设置 to_level 同时为 True"
        log.info("处理为标准化等级 ...")
        if to_std == "mapping":
            log.info("通过映射表处理为标准化等级 ...")
            self._data = self._data.replace(MAP)
        elif to_std == "sampling":
            log.info("通过随机采样处理为标准化等级，生成 100 个样本 ...")
            data = rand_generate_from_std_levels(
                self._data,
                mu=0.0,
                sigma=1.0,
                n_samples=100,
                random_seed=self.random_seed,
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
        lat_monotonic_increasing: bool = True,
    ) -> dict[str, slice]:
        """获取所有数据点的范围

        Args:
            lon_name: 经度名称
            lat_name: 纬度名称
            resolution: 分辨率
            lat_monotonic_increasing: 纬度是否单调递增。同样的区域，有的要“正着”切，有的要“反着”切，这不是数据值不同，而是坐标排序不同（常见于不同机构/产品的 CF-Conventions 写法差异）。
                - 如果为 True，则纬度范围为从最小纬度到最大纬度
                - 如果为 False，则纬度范围为从最大纬度到最小纬度
        Returns:
            dict[str, slice]: 范围
                - lon_name: 经度范围
                - lat_name: 纬度范围
        """
        mins = self.shp.bounds.min()
        maxs = self.shp.bounds.max()
        if lat_monotonic_increasing:
            lat_slice = slice(mins.miny, maxs.maxy, resolution)
        else:
            lat_slice = slice(maxs.maxy, mins.miny, resolution)
        return {
            lat_name: lat_slice,
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
          - ``"1000:1470"`` or ``"1000-1470"`` for explicit year ranges
          - ``"all"`` / ``"full"`` / ``"total"`` for the whole series

        Returns a slice object representing the time range.

        Examples:
            >>> history.get_time_slice(1)  # slice(1000, 1470)
            >>> history.get_time_slice(slice(1, 2))  # slice(1000, 1659)
            >>> history.get_time_slice("1:4")  # slice(1000, 2000)
            >>> history.get_time_slice("stage3")  # slice(1659, 1900)
            >>> history.get_time_slice("1000:2000")  # slice(1000, 2000)
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
                f"Invalid stage expression: {stage}. Expected like 'stage1', '1:3', or '1000-1470'."
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
        # 原始图集数据覆盖 START 至 2020 年（2021 为 arange 的开区间上界，
        # 比分析窗口 FINAL=2000 更晚，属于数据本身的范围，暂无对应常数）。
        full_index = np.arange(START, 2021)
        df = pd.read_excel(
            self.data_path,
            sheet_name=region,
            index_col=0,
            header=1,
        ).replace(0, pd.NA)
        df.index.name = "year"
        return df.reindex(full_index)

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


def regional_precip_z(
    summer_precip: xr.DataArray, sel_dict: Optional[dict | HistoricalRecords] = None
) -> pd.Series:
    """计算区域降水 z-score"""
    if sel_dict is None:
        sel_dict = {}
    if isinstance(sel_dict, HistoricalRecords):
        sel_dict = sel_dict.get_bounds(
            lon_name="x",
            lat_name="y",
            lat_monotonic_increasing=summer_precip.indexes["y"].is_monotonic_increasing,
        )
    series = summer_precip.sel(sel_dict).mean(dim=["x", "y"]).to_series()
    series.name = "pre"
    return (series - series.mean()) / series.std()


def load_validation_data(
    data_path: PathLike,
    csv_save_to: PathLike,
    nc_save_to: PathLike,
    resolution: float = 0.25,
    crs: str = "EPSG:4326",
    recalculate_zscore: bool = False,
    sel_bound_by: Optional[dict | HistoricalRecords] = None,
) -> tuple[xr.DataArray, pd.Series]:
    """加载验证数据

    Args:
        data_path: 数据路径
        resolution: 分辨率
        crs: 坐标系
        recalculate_zscore: 是否重新计算zscore
    Returns:
        pd.DataFrame: 验证数据
    """
    log.info("从 %s 加载验证数据 ...", data_path)
    if recalculate_zscore is False:
        log.info("从文件加载处理后的 z-score 验证数据 ...")
        assert Path(csv_save_to).exists(), f"文件不存在: {csv_save_to}"
        assert Path(data_path).exists(), f"文件不存在: {data_path}"
        summer_precip_z = xr.open_dataarray(data_path).rio.write_crs(crs)
        resolution_loaded = summer_precip_z.rio.resolution()[0]
        if resolution_loaded != resolution:
            log.warning("加载的数据分辨率是: %s，与期望的分辨率: %s 不一致！", resolution_loaded, resolution)
        log.info("从 %s 加载区域降水 z-score 验证数据 ...", csv_save_to)
        regional_z = pd.read_csv(csv_save_to, index_col=0)["pre"]
        return summer_precip_z, regional_z
    # 重新计算 z-score
    log.info("加载原始验证数据（非 z-score） ...")
    lower_data_path = data_path.lower()
    if "china" in lower_data_path:
        log.info("加载 China 数据 ...")
        summer_precip = xr.open_dataarray(data_path).rio.set_spatial_dims("lon", "lat")
    elif "gpcc" in lower_data_path:
        log.info("加载 GPCC 数据 ...")
        summer_precip = xr.open_dataset(data_path, decode_times=True, engine="netcdf4")[
            "precip"
        ].rio.set_spatial_dims("lon", "lat")
    elif "cru" in lower_data_path:
        log.info("加载 CRU 数据 ...")
        summer_precip = xr.open_dataarray(
            data_path, engine="netcdf4"
        ).rio.set_spatial_dims("lon", "lat")
    else:
        raise ValueError(f"未知数据路径: {data_path}")
    summer_precip.rio.write_crs(crs, inplace=True)
    log.debug("验证数据形状: %s", summer_precip.shape)
    log.debug("验证数据分辨率: %s", str(summer_precip.rio.resolution()))
    reprojected = summer_precip.rio.reproject(
        dst_crs=crs,
        resolution=resolution,
    )
    log.info("对原始数据计算 z-score ...")
    regional_z = regional_precip_z(reprojected, sel_bound_by)
    summer_precip_z = (reprojected - reprojected.mean(dim="year")) / reprojected.std(
        dim="year"
    )
    summer_precip_z.name = "summer_precip_z"
    if nc_save_to is not None:
        log.info("保存 z-score 验证数据到 %s ...", nc_save_to)
        summer_precip_z.to_netcdf(nc_save_to)
    if csv_save_to is not None:
        log.info("保存区域降水 z-score 验证数据到 %s ...", csv_save_to)
        regional_z.to_csv(csv_save_to)
    return summer_precip_z, regional_z


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalRecords]:
    """读取自然和历史数据，以及不确定性"""
    start_year = START
    end_year = END
    log.info("加载自然数据 [%s-%s]...", start_year, end_year)
    log.debug("数据路径: %s", cfg.ds.noaa)
    log.debug("数据包括: %s", cfg.ds.includes)
    if cfg.recalculate_data:
        from shifting_baseline.mc import combine_reconstructions

        log.info("重新计算自然数据 ...")
        datasets, uncertainties = load_nat_data(
            folder=cfg.ds.noaa,
            includes=cfg.ds.includes,
            index_name="year",
            start_year=start_year,
        )
        datasets, _ = combine_reconstructions(
            datasets,
            uncertainties,
            standardize=True,
            random_seed=cfg.get("random_seed", 42),
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
        random_seed=cfg.get("random_seed", 42),
    )
    return datasets, uncertainties, history
