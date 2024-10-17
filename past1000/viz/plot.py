#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Optional

import xarray as xr
from matplotkit import with_axes
from matplotlib.axes import Axes


@with_axes(figsize=(12, 3))
def plot_single_time_series(
    data: xr.DataArray,
    freq: str = "YE",
    attrs: Optional[dict] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> None:
    """
    绘制单个时间序列图。

    Args:
        data: xr.DataArray
            输入数据。
        freq: str
            时间频率，默认每年。
        attrs: dict
            属性字典，默认None。
        ax: matplotlib.axes.Axes

    Returns:
        ax: matplotlib.axes.Axes
            绘图的Axes对象。
    """
    if attrs is None:
        attrs = {}
    display_name = attrs.get("display_name", data.name)
    output_units = attrs.get("output_units", "dimensionless")
    color = attrs.get("color", "black")
    resampled = data.resample(time=freq).mean().mean(dim=["lat", "lon"])
    resampled.plot(ax=ax, color=color, **kwargs)

    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    ax.set_title(f"Annual Mean {display_name}")
    ax.set_xlabel("Time (850-1849 CE)")
    ax.set_ylabel(f"{display_name} ({output_units})")
    return ax
