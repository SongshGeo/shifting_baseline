#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
from climate_indices import indices
from climate_indices.compute import Periodicity  # 导入周期类型枚举
from climate_indices.indices import Distribution  # 导入分布类型枚举


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
