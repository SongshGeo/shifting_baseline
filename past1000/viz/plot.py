#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from functools import wraps
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes


def with_axes(
    decorated_func: Callable | None = None, figsize: Tuple[int, int] = (6, 4)
) -> Callable:
    """装饰一个函数/方法，如果该方法接受一个参数叫'ax'并且为None，为其增加一个默认的绘图布。

    Parameters:
        decorated_func:
            被装饰的函数，检查是否有参数传递给装饰器，若没有则返回装饰器本身。
        figsize:
            图片画布的大小，默认宽度为6，高度为4。

    Returns:
        被装饰的函数
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ax = kwargs.get("ax", None)
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
                kwargs["ax"] = ax
                result = func(*args, **kwargs)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    # 检查是否有参数传递给装饰器，若没有则返回装饰器本身
    return decorator(decorated_func) if decorated_func else decorator


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
