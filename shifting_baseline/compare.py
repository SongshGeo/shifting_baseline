#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""对比两个序列的相关性"""
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shifting_baseline.utils.calc import calc_corr, find_top_max_indices
from shifting_baseline.utils.plot import plot_corr_heatmap

if TYPE_CHECKING:
    from shifting_baseline.utils.types import CorrFunc, FilterSide

# A logger for this file
log = logging.getLogger(__name__)


def compare_corr(
    data1: pd.Series,
    data2: pd.Series,
    filter_func: Callable | None = None,
    filter_side: Literal["both", "left", "right"] = "both",
    corr_method: CorrFunc = "pearson",
    window_error: str = "raise",
    n_diff_w: int | float = 2,
    **rolling_kwargs,
) -> tuple[float, float, int]:
    """
    对比两个序列的相关性，并返回相关性系数

    Args:
        data (pd.DataFrame): 数据
        filter_func (Callable | None): 滤波函数
        corr_func (Callable | None): 相关性函数
        rolling_kwargs (dict): 滚动窗口参数

    Returns:
        tuple[float, float, int]: 相关性系数，p值，样本数
    """
    r_benchmark, p_value, n = calc_corr(data1, data2, how=corr_method)
    if filter_func is None:
        return r_benchmark, p_value, n
    default_kwargs = {
        "window": n // 10,  # 默认窗口为样本数的1/10
        "center": False,  # 默认不居中
        "min_periods": min(np.log2(n), 2),  # 默认最小窗口为样本数的对数
        "closed": "both",  # 默认闭合方式为both
    }
    default_kwargs.update(rolling_kwargs)
    if default_kwargs["window"] <= default_kwargs["min_periods"] + n_diff_w:
        if window_error == "raise":
            raise ValueError(
                f"窗口{default_kwargs['window']}太小，n_periods={default_kwargs['min_periods']}，n_diff_w={n_diff_w}，请增大窗口范围"
            )
        elif window_error == "nan":
            return np.nan, np.nan, n
        else:
            raise ValueError(f"无效的窗口错误处理方式: {window_error}")
        default_kwargs["min_periods"] = default_kwargs["window"]
    if filter_side == "both":
        filtered_data1 = data1.rolling(**default_kwargs).apply(filter_func)
        filtered_data2 = data2.rolling(**default_kwargs).apply(filter_func)
    elif filter_side == "left":
        filtered_data1 = data1.rolling(**default_kwargs).apply(filter_func)
        filtered_data2 = data2
    elif filter_side == "right":
        filtered_data1 = data1
        filtered_data2 = data2.rolling(**default_kwargs).apply(filter_func)
    else:
        raise ValueError(f"无效的过滤侧: {filter_side}")
    r, p, n = calc_corr(filtered_data1, filtered_data2, how=corr_method)
    return r, p, n


def compare_corr_2d(
    data1: pd.Series,
    data2: pd.Series,
    windows: np.ndarray,
    min_periods: np.ndarray,
    filter_func: Callable | None = None,
    corr_method: CorrFunc = "pearson",
    n_diff_w: int | float = 2,
    **rolling_kwargs,
) -> tuple[float | np.ndarray, float | np.ndarray, int | np.ndarray]:
    """
    批量对比两个序列的相关性，并返回相关性系数，p值，样本数

    Args:
        data1: 数据1
        data2: 数据2
        filter_func: 滤波函数
        corr_method: 相关性方法
        window_slice: 窗口范围
        min_periods_slice: 最小样本数范围
        rolling_kwargs: 滚动窗口参数

    Returns:
        tuple[float, float, int]: 相关性系数，p值，样本数
    """
    if windows.shape != min_periods.shape:
        raise ValueError("窗口和最小样本数数组形状不一致")
    if windows.size > 5000:
        raise ValueError("窗口数太多，请缩小窗口范围")

    partial_compare_corr = partial(
        compare_corr,
        data1=data1,
        data2=data2,
        filter_func=filter_func,
        corr_method=corr_method,
        window_error="nan",
        n_diff_w=n_diff_w,
        **rolling_kwargs,
    )
    # 批量计算相关性
    rs, ps, ns = np.vectorize(partial_compare_corr)(
        window=windows,
        min_periods=min_periods,
    )
    return rs, ps, ns


def get_filtered_corr(
    rs: np.ndarray,
    ps: np.ndarray,
    ns: np.ndarray,
    windows: np.ndarray,
    sample_threshold: float = 2,
    p_threshold: float = 0.01,
) -> np.ndarray:
    """
    根据相关性系数、p值和样本数，获取过滤后的相关性系数
    """
    enough_n_samples = (ns / windows) > sample_threshold
    significant = ps < p_threshold
    return np.where(enough_n_samples & significant, rs, np.nan)


def experiment_corr_2d(
    data1: pd.Series,
    data2: pd.Series,
    corr_method: CorrFunc = "pearson",
    time_slice: slice = slice(None),
    filter_side: FilterSide = "right",
    filter_func: Callable | None = None,
    sample_threshold: float = 1,
    std_offset: float = 0.2,
    p_threshold: float = 5e-2,
    n_diff_w: int | float = 2,
    ax: plt.Axes | None = None,
) -> tuple[pd.DataFrame, float, plt.Axes]:
    """做一次完整的实验

    Args:
        data1: 数据1
        data2: 数据2
        corr_method: 相关性方法
        time_slice: 时间切片
        filter_side: 过滤侧
        filter_func: 过滤函数
        sample_threshold: 样本数阈值
        std_offset: 标准差偏移
        p_threshold: p值阈值
        penalty: 是否使用惩罚
        n_diff_w: 窗口差值
        ax: 绘图轴

    Returns:
        tuple[pd.DataFrame, float, plt.Axes]: 过滤后的相关性系数，基准相关性系数，绘图轴
    """
    log.info("开始计算相关性 %s，时间切片: %s", corr_method, time_slice)
    # 计算基准相关系数
    base_corr = compare_corr(
        data1.loc[time_slice],
        data2.loc[time_slice],
        corr_method=corr_method,
    )
    r_benchmark = base_corr[0]
    log.info("基准相关系数: %.5f", r_benchmark)
    windows = np.arange(2, 100, 2)
    min_periods = np.arange(2, 20, 1)
    windows_mesh, min_periods_mesh = np.meshgrid(windows, min_periods)

    # 计算相关系数
    rs, ps, ns = compare_corr_2d(
        data1.loc[time_slice],
        data2.loc[time_slice],
        corr_method=corr_method,
        filter_func=filter_func,
        windows=windows_mesh,
        min_periods=min_periods_mesh,
        filter_side=filter_side,
        n_diff_w=n_diff_w,
    )
    filtered = get_filtered_corr(
        rs=rs,
        ps=ps,
        ns=ns,
        windows=windows,
        sample_threshold=sample_threshold,
        p_threshold=p_threshold,
    )
    filtered_df = pd.DataFrame(filtered, index=min_periods, columns=windows)
    ax = plot_corr_heatmap(
        filtered=filtered_df,
        r_benchmark=r_benchmark,
        std_offset=std_offset,
        ax=ax,
    )
    ax.set_title(f"{corr_method.capitalize()} Corr. Coef.")
    return filtered_df, r_benchmark, ax


def sweep_slices(
    start_year: int,
    window_size: int,
    step_size: int,
    end_year: int,
) -> tuple[list[slice], list[int], list[str]]:
    """
    遍历生成所有可能的窗口，并返回切片和对应的标签

    Args:
        start_year: 开始年份
        window_size: 窗口大小
        step_size: 步长
        end_year: 结束年份

    Returns:
        tuple[list[slice], list[int], list[str]]: 切片，中间年份，切片标签
    """
    slices = []
    slice_labels = []
    current_start = start_year

    # 生成所有可能的窗口
    while current_start + window_size <= end_year:
        current_end = current_start + window_size
        slices.append(slice(current_start, current_end))
        slice_labels.append(f"{current_start}-{current_end}")
        current_start += step_size

    mid_year = [np.mean([s.start, s.stop]).astype(int) for s in slices]
    # mid_year = [s.stop for s in slices]
    return slices, mid_year, slice_labels


def sweep_max_corr_year(
    data1: pd.Series,
    data2: pd.Series,
    slices: list[slice],
    windows: np.ndarray,
    min_periods: np.ndarray,
    ratio: float = 0.1,
    **compare_kwargs,
) -> tuple[list, list, list]:
    """
    遍历所有切片，计算相关性系数，并返回最大相关性系数和对应的窗口
    """
    assert 0 < ratio < 1, "ratio must be between 0 and 1"
    max_corr = []
    max_corr_year = []
    r_benchmark_list = []
    for slice_now in slices:
        base_corr = compare_corr(
            data1.loc[slice_now],
            data2.loc[slice_now],
            **compare_kwargs,
        )
        r_benchmark = base_corr[0]
        r_benchmark_list.append(r_benchmark)
        rs, _, _ = compare_corr_2d(
            data1=data1.loc[slice_now],
            data2=data2.loc[slice_now],
            windows=windows,
            min_periods=min_periods,
            **compare_kwargs,
        )
        assert isinstance(rs, np.ndarray), "rs must be a numpy array"
        top_max_indices = find_top_max_indices(rs, ratio)
        max_corr_year.append(top_max_indices)
        max_corr.append(rs[top_max_indices])

    return max_corr_year, max_corr, r_benchmark_list
