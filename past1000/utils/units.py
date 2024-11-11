#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, TypeAlias

import pint
import xarray as xr
from loguru import logger
from pint_xarray import unit_registry as ureg

from past1000.api.log import setup_logger

setup_logger()

WATER_DENSITY = 1000 * ureg("kg/m^3")
TimeUnit: TypeAlias = Literal["second", "day", "month", "year"]
Number: TypeAlias = int | float


def has_unit(numeric: Number | xr.DataArray) -> bool:
    """检查数值是否具有单位。"""
    if isinstance(numeric, xr.DataArray):
        return bool(numeric.pint.units)
    return isinstance(numeric, pint.Quantity)


# https://earthscience.stackexchange.com/questions/20733/fluxnet15-how-to-convert-latent-heat-flux-to-actual-evapotranspiration
def hfls_to_evapo(
    hfls: Number | pint.Quantity,
    time: TimeUnit = "day",
) -> pint.Quantity:
    """将潜热通量转换为实际蒸发量。

    计算公式：E = LE / (ρw * λ)
    其中：
    - E: 蒸发量 (mm/day)
    - LE: 潜热通量 (W/m²)
    - λ: 水的汽化潜热 (≈ 2.45 MJ/kg)

    Args:
        hfls: 潜热通量，单位为 W/m²

    Returns:
        蒸发量，单位为 mm/day
    """
    # 确保输入单位正确
    if not has_unit(hfls):
        logger.warning("没有单位，自动添加单位W/m^2.")
        hfls = hfls * ureg("W/m^2")
    # 水的汽化潜热
    lambda_water = 2.45 * ureg("MJ/kg")  # 约2.45 MJ/kg at 20°C
    # 转换为每日总量 (W = J/s)
    daily_energy = hfls * 86400 * ureg("s/day")  # 转换为 J/(m²·day)
    # 计算蒸发量 E = LE / (ρw * λ)
    evap = daily_energy / (WATER_DENSITY * lambda_water)
    # 转换为mm/day (1 mm = 1 kg/m²)
    unit = "mm/" + time
    if isinstance(evap, pint.Quantity):
        return evap.to(unit)
    return evap.pint.to(unit)


def pr_kg_to_mm(
    pr: Number | pint.Quantity,
    time: TimeUnit = "second",
) -> pint.Quantity:
    """将以质量为单位的降水率转换为以体积为单位的降水强度。

    Args:
        pr (pint.Quantity): 降水率，单位为 kg/(m^2 s)。

    Returns:
        pint.Quantity: 降水强度，单位为 mm/time。
    """
    if not has_unit(pr):
        logger.warning("没有单位，自动添加单位kg/(m^2*s).")
        pr = pr * ureg("kg/(m^2*s)")

    volume = pr / WATER_DENSITY
    unit = "mm/" + time
    if isinstance(volume, pint.Quantity):
        return volume.to(unit)
    return volume.pint.to(unit)


def mrso_kg_to_mm(
    mrso: Number | pint.Quantity,
) -> pint.Quantity:
    """将土壤水分转换为体积单位。"""
    if not has_unit(mrso):
        logger.warning("没有单位，自动添加单位kg/m^2.")
        mrso = mrso * ureg("kg/m^2")
    return mrso / WATER_DENSITY


def convert_cmip_units(
    data: xr.DataArray,
    variable: str,
    output_units: str,
) -> xr.DataArray:
    """
    转换CMIP数据单位。
    """
    if variable == "pr":
        data = pr_kg_to_mm(data)
    if variable == "hfls":
        data = hfls_to_evapo(data)
    if variable == "mrso":
        data = mrso_kg_to_mm(data)
    return data.pint.quantify().pint.to(output_units)
