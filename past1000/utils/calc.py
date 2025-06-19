#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
import xclim.indices as xci
from scipy.signal import detrend

from past1000.ci.spei import DistributionType, calc_single_spei, spei_to_level

# from past1000.core.models import _EarthSystemModel
# from past1000.utils.units import (
#     MONTH_DAYS,
#     YEAR_DAYS,
#     TimeUnit,
#     flux_kg_to_mm,
#     has_unit,
# )


# def calc_pet(
#     tasmin: xr.DataArray,
#     tasmax: xr.DataArray,
#     input_freq: TimeUnit = "month",
#     output_freq: TimeUnit = "month",
# ) -> xr.DataArray:
#     """计算潜在蒸发蒸腾量 (PET)

#     使用Hargreaves方法计算潜在蒸发蒸腾量。

#     Args:
#         input_freq: 输入数据的时间频率 ('day', 'month', 'year')
#         output_freq: 输出数据的时间频率 ('day', 'month', 'year')

#     Returns:
#         包含PET值的数据数组，单位为毫米

#     Note:
#         计算基于最高温度(tasmax)和最低温度(tasmin)
#         结果会根据指定的时间频率进行单位转换
#     """
#     days = {
#         "day": 1,
#         "month": MONTH_DAYS,
#         "year": YEAR_DAYS,
#     }
#     if has_unit(tasmin):
#         tasmin = tasmin.pint.dequantify(format="unit")
#     if has_unit(tasmax):
#         tasmax = tasmax.pint.dequantify(format="unit")
#     pet_per_unit = xci.potential_evapotranspiration(
#         tasmin=tasmin,
#         tasmax=tasmax,
#         method="HG85",
#     )
#     pet = pet_per_unit * days[input_freq]
#     return flux_kg_to_mm(pet, flux_frequency=output_freq)


# def calc_spei(
#     pr: xr.DataArray,
#     pet: xr.DataArray,
#     to_level: bool = True,
#     scale: int = 1,
#     distribution: DistributionType = "pearson",
#     years: Tuple[int, int, int] = (850, 850, 1850),
# ) -> xr.DataArray:
#     """计算标准化降水蒸发指数 (SPEI)

#     Args:
#         to_level: 是否将SPEI值转换为干旱等级
#         scale: SPEI计算的时间尺度（月）
#         distribution: 概率分布类型，可选 "pearson"、"gamma" 等
#         years: 用于拟合分布的年份范围，格式为(起始年,校准起始年,校准结束年)

#     Returns:
#         包含SPEI值的数据数组

#     Note:
#         计算需要降水量(pr)和最高最低温度(tasmax, tasmin)数据
#         如果to_level=True，返回的是干旱等级而不是SPEI值
#     """
#     # 如果输入数据是日尺度，则需要先月平均：
#     pr = pr.resample(time="ME").mean()
#     pet = pet.resample(time="ME").mean()

#     # 对齐时间，删除不匹配的时间
#     pr, pet = xr.align(pr, pet, join="inner")
#     # 计算 SPEI
#     spei = xr.apply_ufunc(
#         partial(
#             calc_single_spei,
#             scale=scale,
#             distribution=distribution,
#             years=years,
#         ),
#         pr,
#         pet,
#         input_core_dims=[["time"], ["time"]],
#         output_core_dims=[["time"]],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[float],
#     )
#     if to_level:
#         return spei_to_level(spei)
#     return spei


# def calc_spei_for_model(model: _EarthSystemModel):
#     """计算模型SPEI"""
#     # 检查模型是否包含所需变量
#     model.check_variables(variables=["pr", "tasmin", "tasmax"])
#     pr = model["pr"]
#     # 计算 PET
#     pet = calc_pet(
#         tasmin=model["tasmin"],
#         tasmax=model["tasmax"],
#         input_freq=model.time_resolutions["pr"],
#         output_freq="month",
#     )
#     return calc_spei(pr, pet)


def get_coords(mask: np.ndarray) -> list[tuple[int, ...]]:
    """
    获取数组中符合条件的坐标列表。

    Args:
        mask: 布尔数组

    Returns:
        list[tuple[int, ...]]: 符合条件的坐标列表
    """
    coords = np.where(mask)
    if coords[0].size == 0:
        return []
    return list(zip(*coords))


def detrend_with_nan(data: pd.Series) -> pd.Series:
    """
    去除数据中的 NaN 值，并进行去趋势处理。

    Args:
        data: 输入数据

    Returns:
        np.ndarray: 去趋势处理后的数据
    """
    dropped_nan_data = data.dropna()
    detrended_data = pd.Series(
        detrend(dropped_nan_data.values),
        index=dropped_nan_data.index,
    ).reindex(data.index)
    return detrended_data
