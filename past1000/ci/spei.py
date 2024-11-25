#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, Optional, Tuple, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import indices
from climate_indices.compute import Periodicity  # 导入周期类型枚举
from climate_indices.indices import Distribution  # 导入分布类型枚举
from scipy.stats import kendalltau

DistributionType: TypeAlias = Literal["pearson", "gamma"]

# 创建分类
BINS = [-np.inf, -1.17, -0.33, 0.33, 1.17, np.inf]
LEVELS = [5, 4, 3, 2, 1]
LABELS = ["Severe drought", "Moderate drought", "Normal", "Wet", "Very wet"]

# 全部旱涝情况按照5级分别给定。
# 1级为涝，2级为偏涝，3级为正常，4级偏旱，5级为旱。无资料依据者判定为0。


# 2. 定义计算单个网格点 SPEI 的函数
def calc_single_spei(
    pr: np.ndarray,
    pet: np.ndarray,
    scale: int = 1,
    distribution: DistributionType = "pearson",
    years: Tuple[int, int, int] = (850, 850, 1850),
) -> np.ndarray:
    """根据气候序列计算 SPEI

    Args:
        pr:
            月尺度降水量。
        pet:
            月尺度潜在蒸散发量。
        scale:
            SPEI 计算的尺度，默认 1 个月。
            通常为 1 个月、3 个月、6 个月、12 个月。
            物理意义为：
                1 个月尺度：反映当月干旱情况。
                3 个月尺度：反映一个季度内的干旱情况。
                6 个月尺度：反映半年内的干旱情况。
                12 个月尺度：反映一年内的干旱情况。
        distribution:
            分布类型，可选 "pearson" 或 "gamma"。
            gamma 为 Gamma 分布，pearson 为 PearsonIII 分布。
            常用 PearsonIII 分布。
        years:
            数据起始年份、校准期起始年份、校准期结束年份。
            默认 (850, 850, 1850)。

    Returns:
        np.ndarray: 月尺度的 SPEI 值。
    """
    if np.all(np.isnan(pr)) or np.all(np.isnan(pet)):
        return np.full_like(pr, np.nan)
    start_year, calibration_start_year, calibration_end_year = years

    # 计算 SPEI，使用 PearsonIII 分布
    return indices.spei(
        precips_mm=pr,
        pet_mm=pet,
        scale=scale,
        distribution=getattr(Distribution, distribution),
        periodicity=Periodicity.monthly,
        data_start_year=start_year,
        calibration_year_initial=calibration_start_year,
        calibration_year_final=calibration_end_year,
    )


def spei_to_level(spei: xr.DataArray) -> xr.DataArray:
    """根据 SPEI 值判断干旱等级"""
    return xr.apply_ufunc(
        lambda x: pd.cut(x, bins=BINS, labels=LEVELS),
        spei,
        vectorize=True,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        output_dtypes=[float],
    )


def vectorize_summary(
    data: np.ndarray,
    method: Literal["mean", "drought"] = "mean",
    axis=None,
) -> np.ndarray:
    """向量化统计年度干旱特征"""
    func = {"mean": mean_drought_level, "drought": drought_summary}
    return xr.apply_ufunc(
        func[method],
        data,
        vectorize=True,
        input_core_dims=[["time"]],
        output_dtypes=[float],
    )


def drought_summary(data: np.ndarray, **kwargs):
    """统计年度干旱特征

    Args:
        data: 月尺度的干旱等级数据
        axis: numpy axis 参数（xarray reduce 操作需要）
        **kwargs: 其他可能的关键字参数

    Returns:
        int: 最显著的干旱等级
    """
    if np.isnan(data).all():
        return np.nan

    # 计算各等级的月数
    counts = pd.Series(data.flatten()).value_counts()
    extreme_droughts = counts.get(1, 0)
    extreme_wets = counts.get(5, 0)

    if extreme_droughts - extreme_wets > 0:
        return 1
    if extreme_wets - extreme_droughts > 0:
        return 5

    moderate_droughts = counts.get(2, 0)
    moderate_wets = counts.get(4, 0)

    if moderate_droughts - moderate_wets > 0:
        return 2
    if moderate_wets - moderate_droughts > 0:
        return 4
    return 3


def mean_drought_level(data: np.ndarray, **kwargs) -> float:
    """计算平均干旱等级"""
    if np.isnan(data).all():
        return np.nan
    return np.mean(data).round(0)


def get_city_lon_lat(city, data):
    """获取城市经纬度"""
    row = data.loc[city]
    return row.geometry.x, row.geometry.y


def get_ESM_time_series(city, xda, gdf, method="nearest"):
    """获取ESM干旱等级"""
    lon, lat = get_city_lon_lat(city, gdf)
    # 1~5 转化为 -2~2
    level = xda.sel(lon=lon, lat=lat, method=method).to_pandas()
    level = level.rename_axis(index="year").rename(lambda x: x.year)
    level.name = city + "_ESM"
    return level - 3


def get_real_time_series(city, df):
    """获取实际干旱等级"""
    return df.loc[:, city] - 3


def match_levels(predicted, history) -> pd.DataFrame:
    """将预测的干旱等级与实际干旱等级匹配"""
    data = pd.concat([predicted, history], axis=1)
    data.columns = ["ESM", "Records"]
    return data


def calc_kendall(data, model_col="ESM", record_col="Records"):
    """计算 Kendall's tau

    :param data: pd.DataFrame
    :param model_col: str, 预测干旱等级列名
    :param record_col: str, 实际干旱等级列名
    """
    # 1. 计算 Kendall's tau（适合等级数据）
    tau, p_value = kendalltau(
        data[model_col],
        data[record_col],
        nan_policy="omit",
        variant="b",
        alternative="two-sided",
    )
    return tau, p_value


def calc_kendall_for_all_cities(
    xda: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    col_name: Optional[str] = None,
):
    """计算所有城市的 Kendall's tau

    Args:
        index: 城市列表
        xda: ESM 干旱等级数据
        gdf: 城市经纬度数据
        df: 实际干旱等级数据
    """
    result = pd.DataFrame()
    if col_name is None:
        cities = gdf.index
    else:
        cities = gdf[col_name]
    for city in cities:
        level = get_ESM_time_series(city, xda, gdf)
        real_level = get_real_time_series(city, df)
        data = match_levels(level, real_level)
        corr, p_value = calc_kendall(data)
        result.loc[city, "corr"] = corr
        result.loc[city, "p_value"] = p_value
    return result.reset_index(names="city")
