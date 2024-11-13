#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import indices
from climate_indices.compute import Periodicity  # 导入周期类型枚举
from climate_indices.indices import Distribution  # 导入分布类型枚举

SPEI_LEVEL: Dict[Tuple[float, float], int] = {
    (-np.inf, -1.5): 5,
    (-1.5, -1.0): 4,
    (-1.0, 1.0): 3,
    (1.0, 1.5): 2,
    (1.5, np.inf): 1,
}

LEVEL_CATE: Dict[int, str] = {
    0: "No data",
    1: "Very wet",
    2: "Wet",
    3: "Normal",
    4: "Drought",
    5: "Very drought",
}

# 创建分类
BINS = [-np.inf, -1.5, -1.0, 1.0, 1.5, np.inf]
LEVELS = [5, 4, 3, 2, 1]
LABELS = ["Severe drought", "Moderate drought", "Normal", "Wet", "Very wet"]

# 全部旱涝情况按照5级分别给定。
# 1级为涝，2级为偏涝，3级为正常，4级偏旱，5级为旱。无资料依据者判定为0。


# 2. 定义计算单个网格点 SPEI 的函数
def calc_single_spei(precip: np.ndarray, pet: np.ndarray) -> np.ndarray:
    """计算单个网格点的 SPEI"""
    if np.all(np.isnan(precip)) or np.all(np.isnan(pet)):
        return np.full_like(precip, np.nan)

    return indices.spei(
        precips_mm=precip,
        pet_mm=pet,
        scale=1,
        distribution=Distribution.gamma,
        periodicity=Periodicity.monthly,
        data_start_year=850,
        calibration_year_initial=850,
        calibration_year_final=1850,
    )


def spei_to_level(value: float) -> int:
    """根据SPEI值判断干旱等级

    Args:
        value: SPEI值

    Returns:
        str: 干旱等级描述
    """
    for (lower, upper), level in SPEI_LEVEL.items():
        if lower < value <= upper:
            return level
    raise ValueError(f"Invalid SPEI value: {value}")


def vectorize_summary(data: np.ndarray, axis=None) -> np.ndarray:
    """向量化统计年度干旱特征"""
    return xr.apply_ufunc(
        drought_summary,
        data,
        vectorize=True,
        input_core_dims=[["time"]],
        output_dtypes=[float],
    )


def drought_summary(data, axis=None, **kwargs):
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

    return 3
