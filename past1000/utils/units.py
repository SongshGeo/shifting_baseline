#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, Optional, TypeAlias

import pint
import xarray as xr
from loguru import logger
from pint_xarray import unit_registry as ureg

from past1000.api.log import setup_logger

setup_logger()

TimeUnit: TypeAlias = Literal[
    "second",
    "minute",
    "hour",
    "day",
    "month",
    "year",
]
Number: TypeAlias = int | float
YEAR_DAYS: Number = 365.2425
MONTH_DAYS: Number = YEAR_DAYS / 12
WATER_DENSITY: pint.Quantity = 1000 * ureg("kg/m^3")
SECONDS: dict[TimeUnit, Number] = {
    "second": 1,
    "minute": 60,
    "hour": 60 * 60,
    "day": 24 * 60 * 60,
    "month": MONTH_DAYS * 24 * 60 * 60,
    "year": YEAR_DAYS * 24 * 60 * 60,
}


def has_unit(numeric: Number | xr.DataArray) -> bool:
    """检查数值是否具有单位。"""
    if isinstance(numeric, xr.DataArray):
        return bool(numeric.pint.units)
    return isinstance(numeric, pint.Quantity)


def _get_time_resolution(da: xr.DataArray) -> TimeUnit:
    """获取 cftime 时间序列的分辨率。

    Parameters
    ----------
    da : xr.DataArray
        带有时间维度的数组

    Returns
    -------
    TimeUnit
        时间分辨率：'month', 'day', 'year' 等
    """
    if "time" not in da.dims:
        raise ValueError(f"时间维度不存在: {da.dims}")
    # 获取前两个时间点
    times = da.time.values
    if len(times) < 2:
        raise ValueError(f"时间序列长度小于2: {da.time}")

    # 计算时间差
    dt = times[1] - times[0]

    # 判断分辨率
    if dt.days == 1:
        return "day"
    if dt.days >= 28 and dt.days <= 31:
        return "month"
    if dt.days >= 365 and dt.days <= 366:
        return "year"
    if dt.days * 24 == dt.total_seconds() / 3600:  # 整数小时
        return "hour"
    raise ValueError(f"无法确定时间分辨率: {dt}")


# https://earthscience.stackexchange.com/questions/20733/fluxnet15-how-to-convert-latent-heat-flux-to-actual-evapotranspiration
def hfls_to_evapo(
    hfls: Number | pint.Quantity,
    flux_frequency: TimeUnit = "day",
    output_units: str = "mm",
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
        logger.warning("蒸散发数据没有单位，自动添加单位W/m^2.")
        hfls = hfls * ureg("W/m^2")
    # 水的汽化潜热
    lambda_water = 2.45 * ureg("MJ/kg")  # 约2.45 MJ/kg at 20°C
    # 转换为每日总量 (W = J/s)
    daily_energy = hfls * SECONDS[flux_frequency] * ureg("s")
    # 计算蒸发量 E = LE / (ρw * λ)
    evap = daily_energy / (WATER_DENSITY * lambda_water)
    # 转换为mm/day (1 mm = 1 kg/m²)
    if isinstance(evap, pint.Quantity):
        return evap.to(output_units)
    return evap.pint.to(output_units)


def flux_kg_to_mm(
    pr: Number | pint.Quantity,
    flux_frequency: TimeUnit = "day",
    output_units: str = "mm",
) -> pint.Quantity:
    """将以质量为单位的降水率转换为以体积为单位的降水强度。

    Args:
        pr (pint.Quantity): 降水率，单位为 kg/(m^2 s)。

    Returns:
        pint.Quantity: 降水强度，单位为 mm/time。
    """
    if not has_unit(pr):
        logger.warning("降水数据没有单位，自动添加单位kg/(m^2*s).")
        pr = pr * ureg("kg/(m^2*s)")

    volume = pr / WATER_DENSITY * SECONDS[flux_frequency] * ureg("s")
    if isinstance(volume, pint.Quantity):
        return volume.to(output_units)
    return volume.pint.to(output_units)


def storage_kg_to_mm(
    mrso: Number | pint.Quantity,
) -> pint.Quantity:
    """将土壤水分转换为体积单位。"""
    if not has_unit(mrso):
        logger.warning("土壤水分数据没有单位，自动添加单位kg/m^2.")
        mrso = mrso * ureg("kg/m^2")
    return mrso / WATER_DENSITY


def convert_cmip_units(
    data: xr.DataArray,
    variable: str,
    unit_to: str,
    flux_frequency: Optional[TimeUnit] = None,
) -> xr.DataArray:
    """
    转换CMIP数据单位。
    """
    if flux_frequency is None:
        flux_frequency = _get_time_resolution(data)
    if variable in ("pr", "mrro"):
        data = flux_kg_to_mm(data, flux_frequency)
    if variable == "hfls":
        data = hfls_to_evapo(data, flux_frequency)
    if variable == "mrso":
        data = storage_kg_to_mm(data)
    return data.pint.quantify().pint.to(unit_to)
