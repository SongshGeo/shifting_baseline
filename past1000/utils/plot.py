#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from itertools import product
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotkit import with_axes
from matplotlib.axes import Axes

from past1000.constants import LEVELS, TICK_LABELS
from past1000.utils.calc import fill_star_matrix, get_coords


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


@with_axes(figsize=(4, 3.5))
def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str | None = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """绘制混淆矩阵

    Args:
        y_true: Union[pd.Series, np.ndarray]
            True labels
        y_pred: Union[pd.Series, np.ndarray]
            Predicted labels
        ax: Optional[Axes]
            Matplotlib axes object
        dropna: bool
            Whether to drop NA values
        **kwargs: dict
            kwargs for seaborn.heatmap

    Returns:
        Axes: 返回 axes 对象
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    assert isinstance(cm_df, pd.DataFrame), "cm_df must be a pandas DataFrame"

    # 构造对角线 mask
    mask = np.eye(len(cm_df), dtype=bool)
    zero_mask = cm_df == 0

    # 绘制热力图，mask 掉对角线
    sns.heatmap(
        cm_df,
        annot=True,  # 先不显示数字
        fmt="d",
        cmap="Reds",
        ax=ax,
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Mismatches"},
        mask=mask | zero_mask,  # 对角线不显示颜色
        **kwargs,
    )

    # 手动在对角线上写黑色数字
    for i in range(len(cm_df)):
        value = cm_df.iloc[i, i]
        if value == 0:
            continue
        ax.text(
            i + 0.5,
            i + 0.5,  # 热力图格子中心
            f"{value:d}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
            fontweight="bold",
        )

    ax.grid(True, linestyle=":", color="gray", alpha=0.3)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_yticklabels(TICK_LABELS)
    ax.set_xlabel("Natural")
    ax.set_ylabel("Recorded")
    if title is not None:
        ax.set_title(title, fontsize=9)
    return ax


def plot_corr_2d(
    ns: np.ndarray,
    rs: np.ndarray,
    windows: np.ndarray,
    min_periods: np.ndarray,
) -> tuple[plt.Figure, tuple[Axes, Axes]]:
    """绘制相关性2D热图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5), tight_layout=True)
    sns.heatmap(
        pd.DataFrame(ns, index=min_periods, columns=windows),
        ax=ax1,
        cmap="coolwarm",
        annot=False,
        linewidths=0.1,
        linecolor="white",
        vmin=0,
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax1.set_title("N Samples")
    sns.heatmap(
        pd.DataFrame(rs, index=min_periods, columns=windows),
        ax=ax2,
        cmap="coolwarm",
        annot=False,
        linewidths=0.05,
        linecolor="white",
        vmin=-0.1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax2.set_title("Correlation Coefficient")
    ax1.set_xlabel("Window Size")
    ax2.set_xlabel("Window Size")
    ax1.set_ylabel("Minimum Periods")
    ax2.set_ylabel("Minimum Periods")
    return fig, (ax1, ax2)


@with_axes(figsize=(3, 2.5))
def plot_corr_heatmap(
    filtered: pd.DataFrame | np.ndarray,
    r_benchmark: float = 0.5,
    std_offset: float = 0,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """绘制相关性热图"""
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    v_max = np.nanmax(filtered)
    v_min = r_benchmark * 2 - v_max

    sns.heatmap(
        filtered,
        cmap="vlag",
        annot=False,
        linewidths=0.05,
        linecolor="white",
        vmin=v_min,
        vmax=v_max,
        cbar_kws={
            "shrink": 0.4,  # 缩小 colorbar
            # "location": "bottom",  # 移到上方
            "pad": 0.05,  # 调整间距
        },
        ax=ax,
        square=True,
        **kwargs,
    )
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Minimum Samples")

    # 设置边框
    sns.despine(
        ax=ax,
        top=False,
        right=False,
        left=False,
        bottom=False,
        trim=False,
    )
    ax.locator_params(axis="both", nbins=9)

    if r_benchmark is not None:
        # 获取 colorbar 对象
        cbar = ax.collections[0].colorbar
        # 添加垂直参考线（因为现在是水平的colorbar）
        cbar.ax.axhline(y=r_benchmark, color="black", linewidth=2)
        cbar.ax.set_yticks(np.linspace(v_min, v_max, 5))
        # 2位小数点
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(v_min, v_max, 5)])

    # 绘制对角线
    # min_period = filtered.index.values.min()
    # max_period = filtered.index.values.max()
    # min_window = filtered.columns.values.min()
    # max_window = filtered.columns.values.max()
    # ax.plot(
    #     [min_window, max_period],
    #     [min_period, max_window],
    #     color="black",
    #     linestyle=":",
    #     linewidth=1,
    #     label="window = period",
    # )

    # 标记最大值点
    std_value = np.nanstd(filtered)
    lower_bound = v_max - std_offset * std_value
    points = get_coords(filtered >= lower_bound)

    for i, point in enumerate(points):
        if i == 0:
            ax.scatter(
                point[1],
                point[0],
                s=10,
                c="r",
                alpha=0.8,
                label=f"~Max: {v_max:.2f} - {std_offset:.2f}σ",
            )
        else:
            ax.scatter(point[1], point[0], s=10, c="r", alpha=0.8)
        ax.axvline(point[1], color="lightgray", linestyle="--", alpha=0.8, lw=0.5)
        ax.axhline(point[0], color="lightgray", linestyle="--", alpha=0.8, lw=0.5)

    # 将图例移到上方
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.5, 1.15),  # 移到图的上方
        ncol=2,  # 水平排列
        fontsize=8,
        frameon=False,  # 去掉边框
    )
    return ax


@with_axes(figsize=(3, 2.5))
def enhanced_corr_plot(
    df: pd.DataFrame,
    ax: Optional[Axes] = None,
    **kwargs,
) -> None:
    """绘制增强相关系数热图

    Args:
        df: pd.DataFrame
            输入数据。
        ax: matplotlib.axes.Axes
            绘图的Axes对象。
        **kwargs: dict
            kwargs for sns.heatmap
            cmap: 颜色映射
            annot: 是否显示注释
            annot_kws: 注释的样式
            fmt: 注释的格式
            vmin: 颜色映射的最小值
            vmax: 颜色映射的最大值
            cbar_kws: 颜色条的样式
            square: 是否将热图的单元格设置为正方形

    Returns:
        ax: matplotlib.axes.Axes
            绘图的Axes对象。
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    default_kwargs = {
        "cmap": "vlag",
        "annot": True,
        "annot_kws": {"size": 8},
        "fmt": ".1f",
        "vmin": -1,
        "vmax": 1,
        "cbar_kws": {"shrink": 0.8},
    }
    default_kwargs.update(kwargs)
    sns.heatmap(df.corr(numeric_only=True), ax=ax, **default_kwargs)
    ax.set_title("Correlation Coefficient")
    return ax


@with_axes(figsize=(2, 3.5))
def plot_mismatch_matrix(
    actual_diff_aligned: pd.DataFrame,
    p_value_matrix: pd.DataFrame,
    false_count_matrix: pd.DataFrame,
    ax: Optional[Axes] = None,
) -> Axes:
    """绘制不匹配矩阵

    Args:
        actual_diff_aligned: pd.DataFrame
            实际差异矩阵
        p_value_matrix: pd.DataFrame
            显著性矩阵
        false_count_matrix: pd.DataFrame
            不匹配矩阵
        ax: Optional[Axes]
            绘图的Axes对象。

    Returns:
        Axes: 返回 axes 对象
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"

    def is_significant(p_value: float) -> bool:
        if np.isnan(p_value):
            return False
        return p_value < 0.1

    # 1. 设置渐变色和归一化
    vmax = np.nanmax(np.abs(actual_diff_aligned.values))
    cmap = mpl.cm.coolwarm  # 或 mpl.cm.RdBu
    # 归一化，使用幂函数归一化，gamma=0.5 使得颜色分布更均匀
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=-vmax, vmax=vmax)

    for l1, l2 in product(LEVELS, LEVELS):
        value = actual_diff_aligned.loc[l1, l2]
        p_value = p_value_matrix.loc[l1, l2]
        false_count = false_count_matrix.loc[l1, l2]
        color = cmap(norm(value))
        lw = false_count * 0.5
        alpha = 0.9 if is_significant(p_value) else 0.4
        ax.plot([0, 1], [l2, l1], lw=lw, color=color, alpha=alpha)

    ax.set_xlim(0, 1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Natural", "Recorded"])
    ax.set_yticklabels(TICK_LABELS)
    sns.despine(ax=ax, left=False, right=False, top=False, bottom=False)

    # 2. 创建渐变色 colorbar，横向放在上方
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.15, alpha=0.4
    )
    # cbar.set_label('Standardized difference', labelpad=8, fontsize=10, loc='center')
    cbar.ax.xaxis.set_label_position("bottom")  # 标签放到上方
    cbar.ax.set_xlabel("Std. diff. between last/current")

    # 不显示轴线
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)

    # 同时显示 y 轴左侧和右侧坐标刻度、注释
    ax.yaxis.set_ticks_position("both")
    ax.yaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.grid(True, linestyle="--", alpha=0.8, color="black")

    ax.text(
        0.5, 2.5, "Expected diff. = 0", color="gray", fontsize=9, ha="center", va="top"
    )
    ax.set_xlabel("Recorded level diff.")

    return ax


@with_axes(figsize=(3, 2.5))
def heatmap_with_annot(
    matrix: pd.DataFrame,
    p_value: pd.DataFrame | None = None,
    annot: pd.DataFrame | None = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """以热力图的形式绘制不匹配情况前后对比图

    Args:
        actual_diff: pd.DataFrame
            实际差异矩阵
        p_value: pd.DataFrame
            显著性矩阵
        annot: pd.DataFrame
            注释矩阵
        ax: Optional[Axes]
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    # 如果 p_value 不为 None，但 annot 为 None，则自动生成注释矩阵
    if p_value is not None and annot is None:
        annot = fill_star_matrix(p_value, matrix)
    # 如果 p_value 和 annot 都为 None，则抛出错误
    elif p_value is None and annot is None:
        raise ValueError("p_value and annot must be provided together")
    # 否则不执行任何操作，绘制热力图
    sns.heatmap(
        matrix,
        annot=annot,  # 使用我们自定义的标签矩阵
        fmt="s",  # "s" 表示我们提供的是字符串格式
        cmap="coolwarm",
        square=True,
        ax=ax,
        linewidths=0.5,
        # 给颜色条加个标签
        cbar_kws={"label": "Standardized difference"},
        center=0,
        linecolor="lightgray",
    )

    ax.set_title("False Estimation")
    ax.set_xlabel("Classified")
    ax.set_ylabel("Expect")
    ax.set_xticklabels(TICK_LABELS)
    ax.set_yticklabels(TICK_LABELS)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_xlabel("Natural data")
    ax.set_ylabel("Historical data")
    return ax


@with_axes(figsize=(14, 3))
def plot_std_times(
    data: pd.Series,
    ax: Optional[Axes] = None,
    color_options: Optional[dict[str, str]] = None,
) -> None:
    """分段绘制正负值出现次数图

    Args:
        data: pd.Series
            输入数据。
        ax: matplotlib.axes.Axes
            绘图的Axes对象。
        colors: dict[str, str]
            颜色映射。
    """
    assert isinstance(data, pd.Series), "data must be a pandas Series"
    assert data.index.is_monotonic_increasing, "index must be monotonic increasing"
    assert data.index.is_unique, "index must be unique"
    assert ax is not None, "ax must be provided"

    # 使用numpy条件设置颜色：正数蓝色，负数红色，零值灰色
    if color_options is None:
        color_options = {
            "positive": "#689B8A",
            "negative": "#E43636",
            "zero": "gray",
        }
    colors = np.where(
        data.values > 0,
        color_options["positive"],  # 正数用蓝色
        np.where(
            data.values < 0, color_options["negative"], color_options["zero"]
        ),  # 负数用红色，零值用灰色
    )

    # 绘制垂直线和散点
    for x, y, color in zip(data.index, data.values, colors):
        ax.vlines(x, 0, y, colors=color, linewidth=1)
        ax.scatter(x, y, c=color, s=30, zorder=3)

    # 添加基线
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(True, alpha=0.3, ls=":", color="gray")

    ax.set_ylabel("Times of STD")
    ax.set_xlabel("Year")
    return ax
