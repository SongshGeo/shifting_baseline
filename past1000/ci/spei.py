#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, Tuple, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import indices
from climate_indices.compute import Periodicity  # 导入周期类型枚举
from climate_indices.indices import Distribution  # 导入分布类型枚举

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


def vectorize_summary(data: np.ndarray, axis=None) -> np.ndarray:
    """向量化统计年度干旱特征"""
    return xr.apply_ufunc(
        drought_summary,
        data,
        vectorize=True,
        input_core_dims=[["time"]],
        output_dtypes=[float],
    )


def drought_summary(data: np.ndarray, axis=None, **kwargs):
    """统计年度干旱特征

    Args:
        data: 月尺度的干旱等级数据
        axis: numpy axis 参数（xarray reduce 操作需要）
        **kwargs: 其他可能的关键字参数

    Returns:
        int: 最显著的干旱等级
    """

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

    if np.isnan(data).all():
        return np.nan
    return 3
