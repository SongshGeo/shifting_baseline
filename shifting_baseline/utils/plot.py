#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from itertools import product
from typing import Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotkit import with_axes
from matplotlib.axes import Axes
from pyproj import CRS
from sklearn.metrics import root_mean_squared_error

from shifting_baseline.constants import LEVELS, TICK_LABELS
from shifting_baseline.utils.calc import fill_star_matrix, low_pass_filter


def is_significant(p_value: float, threshold: float = 0.1) -> bool:
    """判断p值是否显著"""
    if np.isnan(p_value):
        return False
    return p_value < threshold


def get_marker(p_value: float, threshold: float = 0.1) -> str:
    """获取显著性标记"""
    if is_significant(p_value, threshold=0.01):
        marker = "***"
    elif is_significant(p_value, threshold=0.05):
        marker = "**"
    elif is_significant(p_value, threshold=0.10):
        marker = "*"
    else:
        marker = ""
    return marker


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
        cbar_kws={"shrink": 0.6, "label": "Number of Mismatches"},
        mask=mask | zero_mask,  # 对角线不显示颜色
        **kwargs,
    )
    ax.figure.axes[-1].yaxis.label.set_size(9)
    ax.figure.axes[-1].yaxis.set_ticks(np.arange(0, 31, 10))

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
    ax.set_xlabel("Natural Proxies")
    ax.set_ylabel("Historical Archives")
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
            # "location": "bottom",  # 移到上方
            "pad": 0.05,  # 调整间距
            "shrink": 0.7,
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

    # 1. 设置渐变色和归一化
    vmax = np.nanmean(np.abs(actual_diff_aligned.values)) * 1.2
    cmap = mpl.cm.coolwarm  # 或 mpl.cm.RdBu
    # 使用线性归一化，对于对称的数据分布更合适
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    for l1, l2 in product(LEVELS, LEVELS):
        value = actual_diff_aligned.loc[l1, l2]
        if l1 == l2 or np.isnan(value):
            continue
        p_value = p_value_matrix.loc[l1, l2]
        false_count = false_count_matrix.loc[l1, l2]

        color = cmap(norm(value))
        lw = false_count * 0.7
        va = "bottom" if l1 > l2 else "top"
        marker = get_marker(p_value, threshold=0.1)
        if marker:
            ax.text(
                0,
                l2,
                marker,
                ha="left",
                va=va,
                fontsize=10,
                fontweight="bold",
                color=color,
                zorder=10,
            )
        ax.plot([0, 1], [l2, l1], lw=lw, color=color, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Natural", "Historical"])
    ax.set_yticklabels(TICK_LABELS)
    sns.despine(ax=ax, left=False, right=False, top=False, bottom=False)

    # 2. 创建渐变色 colorbar，横向放在上方
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.15)
    # cbar.set_label('Standardized difference', labelpad=8, fontsize=10, loc='center')
    cbar.ax.xaxis.set_label_position("bottom")  # 标签放到上方
    cbar.ax.set_xlabel("WDIs' diff. (current - last)")

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
    ax.set_xlabel("Wet/Dry Index diff.", fontsize=9)

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
    **kwargs,
) -> None:
    """分段绘制正负值出现次数图

    Args:
        data: pd.Series
            输入数据。
        ax: matplotlib.axes.Axes
            绘图的Axes对象。
        color_options: dict[str, str]
            颜色映射。
        add_legend: bool
            是否添加图例。
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
            "zero": "lightgray",
        }
    colors = np.where(
        data.values > 0.33,
        color_options["positive"],  # 正数用蓝色
        np.where(
            data.values < -0.33, color_options["negative"], color_options["zero"]
        ),  # 负数用红色，零值用灰色
    )

    # 跟踪已添加的标签类型
    added_labels = set()

    # 绘制垂直线和散点
    for x, y, color in zip(data.index, data.values, colors):
        ax.vlines(x, 0, y, colors=color, linewidth=1, **kwargs)

        # 根据数值类型确定标签
        if y > 0.33:
            label = "Wet Year" if "positive" not in added_labels else None
            if label:
                added_labels.add("positive")
        elif y < -0.33:
            label = "Dry Year" if "negative" not in added_labels else None
            if label:
                added_labels.add("negative")
        else:
            label = "Normal Year" if "zero" not in added_labels else None
            if label:
                added_labels.add("zero")
            else:
                label = None
        ax.scatter(x, y, c=color, s=30, zorder=3, edgecolors="white", label=label)

    # 添加基线
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(True, alpha=0.3, ls=":", color="gray")

    ax.set_ylabel("Times of STD")
    ax.set_xlabel("Year")

    return ax


@with_axes(figsize=(8, 3.5))
def plot_correlation_windows(
    max_corr_year: list[np.ndarray],
    max_corr_improvment: list[np.ndarray],
    mid_years: list[int],
    slice_labels: list[str] | None = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    绘制时间窗口的最优相关性年份图

    Parameters:
    -----------
    max_corr_year : list of arrays
        每个时间窗口的最大相关性年份数据
    max_corr_improvment : list of arrays
        每个时间窗口的最大相关性改进值数据
    slice_labels : list, optional
        时间窗口标签
    figsize : tuple
        图形大小
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    # 数据预处理
    means = np.array([arr.mean() for arr in max_corr_year])
    stds = np.array([arr.std() for arr in max_corr_year])

    # 过滤负相关性，转换为改进百分比
    corr_improvements = []
    for improved_ratio in max_corr_improvment:
        mean_improvement = improved_ratio.mean()
        corr_improvements.append(mean_improvement if mean_improvement >= 0 else np.nan)

    corr_improvements = np.array(corr_improvements) * 100
    valid_mask = ~np.isnan(corr_improvements)

    # 设置颜色映射
    if np.any(valid_mask):
        vmin, vmax = 0, np.nanmax(corr_improvements) + 1
        # 确保 vmax > vmin 避免颜色映射问题
        if vmax <= vmin:
            vmax = vmin + 1
    else:
        vmin, vmax = 0, 1

    cmap = cm.OrRd
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 收集有效点用于趋势线
    valid_points = []

    # 绘制数据点
    i = 0
    for mid, mean_y, std_y, improvement in zip(
        mid_years, means, stds, corr_improvements
    ):
        if valid_mask[i]:
            point_color = cmap(norm(improvement))
            alpha = 0.8
            valid_points.append({"x": mid, "y": mean_y})

            # 绘制带彩色点和黑色误差棒
            ax.errorbar(
                mid,
                mean_y,
                yerr=std_y,
                fmt="o",
                color=point_color,  # 点的颜色
                ecolor="black",  # 误差棒颜色
                elinewidth=2,  # 误差棒线宽
                capsize=5,  # 误差棒端帽大小
                capthick=2,  # 误差棒端帽粗细
                markersize=8,  # 点的大小
                alpha=alpha,
                markeredgecolor="black",  # 点的边框
                markeredgewidth=0.5,  # 点边框宽度
            )
        else:
            # 对于无效数据，可以选择不绘制或用灰色
            ax.errorbar(
                mid,
                mean_y,
                yerr=std_y,
                fmt="o",
                color="lightgray",
                ecolor="lightgray",
                alpha=0.3,
                capsize=5,
                capthick=2,
                elinewidth=2,
                markersize=8,
            )

        i += 1

    # 添加趋势线
    if len(valid_points) > 1:
        valid_df = pd.DataFrame(valid_points)
        sns.regplot(
            data=valid_df,
            x="x",
            y="y",
            scatter=False,
            color="red",
            line_kws={"linewidth": 2, "alpha": 0.8, "linestyle": "--"},
            ax=ax,
            truncate=False,
        )

    # 设置坐标轴
    ax.set_xticks(mid_years)
    ax.set_xticklabels(slice_labels, rotation=30)
    ax.set_xlabel("Periods applied the filter (AD)")
    ax.set_ylabel("Window Size with Optimal $Tau$")
    # ax.set_title("Optimal Year of Max Correlation with Error Bars\n(Color indicates correlation improvement)")
    ax.grid(True, alpha=0.3)

    # 添加颜色条（修复白色问题）
    if np.any(valid_mask):
        # 只为有效数据创建颜色映射
        valid_improvements = corr_improvements[valid_mask]
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(valid_improvements)  # 设置实际数据数组

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Avg. Improvement of $Tau$ (%)", rotation=270, labelpad=15)
    lims = ax.get_xlim()
    ax.set_xlim(lims)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


@with_axes(figsize=(10, 4))
def plot_time_series_with_lowpass(
    data: pd.Series,
    filtered_data: pd.Series | None = None,
    window_size: int = 30,
    filter_method: str = "rolling_mean",
    baseline: float | None = None,
    rmse_data: pd.Series | None = None,
    ax: Optional[Axes] = None,
    show_annual: bool = True,
    show_filtered: bool = True,
    show_baseline: bool = True,
    show_rmse: bool = True,
    colors: dict[str, str] | None = None,
) -> Axes:
    """绘制带低通滤波的时间序列图，支持基准线着色和误差范围显示。

    这个函数可以绘制类似你描述的时间序列图，包括：
    - 年度原始数据（垂直线）
    - 30年低通滤波后的平滑趋势（粗线）
    - 基准线以上的蓝色区域和以下的红色区域
    - ±1 RMSE误差范围（浅灰色阴影）

    Args:
        data: 原始时间序列数据
        filtered_data: 已滤波的数据，如果为None则自动计算
        window_size: 滤波窗口大小，默认30年
        filter_method: 滤波方法，'rolling_mean', 'gaussian', 'butterworth'
        baseline: 基准线值，如果为None则使用数据均值
        rmse_data: RMSE数据，如果为None则自动计算
        ax: matplotlib坐标轴对象
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        unit: 数据单位
        show_annual: 是否显示年度数据
        show_filtered: 是否显示滤波数据
        show_baseline: 是否显示基准线着色
        show_rmse: 是否显示RMSE误差范围
        colors: 颜色配置字典
        **kwargs: 其他绘图参数

    Returns:
        Axes: matplotlib坐标轴对象

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # 创建示例数据
        >>> years = np.arange(1100, 2001)
        >>> np.random.seed(42)
        >>> data = pd.Series(
        ...     np.random.randn(len(years)) * 200 + 1600 +
        ...     np.sin(years / 50) * 100,
        ...     index=years
        ... )
        >>> # 绘制时间序列图
        >>> ax = plot_time_series_with_lowpass(data, title="Streamflow Reconstruction")
    """

    if colors is None:
        colors = {
            "annual": "darkgray",
            "filtered": "black",
            "baseline": "gray",
            "above_baseline": "lightblue",
            "below_baseline": "lightcoral",
            "rmse": "lightgray",
        }

    # 计算滤波数据
    if filtered_data is None:
        filtered_data = low_pass_filter(
            data, window_size=window_size, method=filter_method
        )

    # 计算基准线
    if baseline is None:
        baseline = filtered_data.mean()

    # 计算RMSE
    if rmse_data is None and show_rmse:
        # 使用原始数据与滤波数据的差异作为RMSE的近似
        valid_mask = ~filtered_data.isna()
        if valid_mask.any():
            rmse_value = root_mean_squared_error(
                y_true=data[valid_mask],
                y_pred=filtered_data[valid_mask],
            )
            rmse_data = pd.Series([rmse_value] * len(data), index=data.index)

    assert isinstance(ax, Axes), "ax must be an instance of Axes"

    # 绘制年度数据（垂直线）
    if show_annual:
        ax.plot(
            data.index,
            data.values,
            color=colors["annual"],
            linewidth=0.5,
            alpha=0.6,
            label="Annual data",
        )

    # 绘制RMSE误差范围
    if show_rmse and rmse_data is not None:
        valid_mask = ~filtered_data.isna()
        if valid_mask.any():
            ax.fill_between(
                filtered_data.index[valid_mask],
                filtered_data[valid_mask] - rmse_data[valid_mask],
                filtered_data[valid_mask] + rmse_data[valid_mask],
                color=colors["rmse"],
                alpha=0.3,
                label="±1 RMSE",
            )

    # 绘制基准线着色区域
    if show_baseline:
        valid_mask = ~filtered_data.isna()
        if valid_mask.any():
            # 找到高于和低于基准线的区域
            above_mask = (filtered_data >= baseline) & valid_mask
            below_mask = (filtered_data < baseline) & valid_mask

            # 绘制高于基准线的区域
            if above_mask.any():
                ax.fill_between(
                    filtered_data.index[above_mask],
                    baseline,
                    filtered_data[above_mask],
                    color=colors["above_baseline"],
                    alpha=0.6,
                    label="Above baseline",
                )

            # 绘制低于基准线的区域
            if below_mask.any():
                ax.fill_between(
                    filtered_data.index[below_mask],
                    filtered_data[below_mask],
                    baseline,
                    color=colors["below_baseline"],
                    alpha=0.6,
                    label="Below baseline",
                )

    # 绘制滤波后的数据
    if show_filtered:
        valid_mask = ~filtered_data.isna()
        if valid_mask.any():
            ax.plot(
                filtered_data.index[valid_mask],
                filtered_data[valid_mask],
                color=colors["filtered"],
                linewidth=2,
                label=f"Reconstruction ({window_size}-year low-pass filter)",
            )

    # 绘制基准线
    if show_baseline:
        ax.axhline(
            y=baseline,
            color=colors["baseline"],
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"Baseline ({baseline:.1f})",
        )

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle=":", color="gray")

    # 美化坐标轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


@with_axes
def plot_corr_map(
    corr: xr.DataArray,
    p_value: xr.DataArray,
    threshold: float = 0.05,
    ax: plt.Axes | None = None,
    mask: bool = True,
    base_maps: dict[str, str] | None = None,
    crs: str | None | CRS = None,
    add_colorbar: bool = True,
    **kwargs,
) -> plt.Axes:
    """绘制相关性地图

    Args:
        corr (xr.DataArray): 相关性
        p_value (xr.DataArray): p值
        threshold (float, optional): 显著性阈值. Defaults to 0.05.
        ax (plt.Axes | None, optional): 坐标轴. Defaults to None.

    Returns:
        plt.Axes: 坐标轴
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    if crs is None:
        assert hasattr(corr, "rio"), "corr must have rio attribute"
        crs = corr.rio.crs
    corr_map = corr
    # 创建显著性掩码
    colors = ["black", "gray", "lightgray"]
    linewidths = [0.8, 0.8, 1.5]
    levels = [0.1, 0.05, 0.01]
    for level, color, linewidth in zip(levels, colors, linewidths):
        contour = p_value.plot.contour(
            ax=ax,
            levels=[level],
            colors=[color],
            linewidths=[linewidth],
            linestyles=["--"],
            alpha=0.8,
        )
        # 添加简洁标签
        ax.clabel(
            contour, inline=True, fontsize=7, fmt=f"{level:.2f}", colors=color  # 只显示数值
        )
    if mask:
        significant_mask = p_value < threshold
        corr_map = corr_map.where(significant_mask)
    if add_colorbar:
        cbar_kwargs = {"shrink": 0.8, "aspect": 20}
    else:
        cbar_kwargs = None
    corr_map.plot(
        ax=ax,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
    )
    # 绘制底图
    if base_maps is None:
        base_maps = {}
    zorder = -1
    for name, shp in base_maps.items():
        gpd.read_file(shp).to_crs(crs).plot(
            ax=ax,
            color="gray",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.4,
            label=name.capitalize(),
            zorder=zorder,
        )
        zorder -= 1
    sns.despine(ax=ax, left=False, bottom=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
    ax.set_title("")  # 清空标题
    ax.set_xlabel("")  # 清空x轴标签
    return ax
