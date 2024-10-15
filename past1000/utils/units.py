#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from numbers import Number
from typing import Literal, TypeAlias

import pint
import xarray as xr
from loguru import logger
from pint_xarray import unit_registry as ureg

from past1000.api.log import setup_logger

setup_logger()

WATER_DENSITY = 1000 * ureg("kg/m^3")
TimeUnit: TypeAlias = Literal["second", "day", "month", "year"]


def has_unit(numeric: Number | xr.DataArray) -> bool:
    """检查数值是否具有单位。"""
    if isinstance(numeric, xr.DataArray):
        return bool(numeric.pint.units)
    return isinstance(numeric, pint.Quantity)


def pr_kg_to_mm(
    pr: pint.Quantity,
    time: TimeUnit = "second",
    density: pint.Quantity = WATER_DENSITY,
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

    volume = pr / density
    unit = "mm/" + time
    if isinstance(volume, pint.Quantity):
        return volume.to(unit)
    return volume.pint.to(unit)


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
    return data.pint.quantify().pint.to(output_units)
