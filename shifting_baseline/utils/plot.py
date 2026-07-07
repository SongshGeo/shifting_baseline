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
from matplotlib.patches import Patch, Rectangle
from mksci_font import config_font
from pyproj import CRS
from sklearn.metrics import root_mean_squared_error

from shifting_baseline.constants import COLORS, LEVELS, THRESHOLDS, TICK_LABELS
from shifting_baseline.utils.calc import (
    fill_star_matrix,
    get_significance_stars,
    low_pass_filter,
)

# 全局设置
# 设置seaborn风格
sns.set_style("ticks")
sns.set_context("paper")
# 设置字体大小
config_font({"font.size": 9})


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
        cbar.ax.set_yticks(np.linspace(v_min, v_max, 3))
        # 2位小数点
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(v_min, v_max, 3)])
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
    vmax = np.nanmean(np.abs(actual_diff_aligned.values)) * 1.1
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
        marker = get_significance_stars(p_value)
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
    n_levels: int = 5,
    legend_kwargs: Optional[dict] = None,
    **kwargs,
) -> Axes:
    """分段绘制历史档案标准差时间序列。

    Args:
        data: 输入数据（连续 z-score 或档案代表性标准差）。
        ax: 绘图的 Axes 对象。
        color_options: 3 级模式下的颜色映射（``n_levels=3`` 时使用）。
        n_levels: 分级数量，5 为旱涝 5 档（默认），3 为旱/正常/涝。
        legend_kwargs: 若提供，则按 SW→SD 顺序绘制 5 级图例。
    """
    assert isinstance(data, pd.Series), "data must be a pandas Series"
    assert data.index.is_monotonic_increasing, "index must be monotonic increasing"
    assert data.index.is_unique, "index must be unique"
    assert ax is not None, "ax must be provided"
    assert n_levels in (3, 5), "n_levels must be 3 or 5"

    added_labels: set[int | str] = set()

    if n_levels == 5:
        level_colors = [COLORS[0], COLORS[1], COLORS[2], "#5CB8E8", COLORS[3]]
        level_labels = TICK_LABELS
        bin_idx = np.digitize(data.values, THRESHOLDS)
        for x, y, idx in zip(data.index, data.values, bin_idx):
            color = level_colors[idx]
            ax.vlines(x, 0, y, colors=color, linewidth=1, **kwargs)
            label = level_labels[idx] if idx not in added_labels else None
            if label:
                added_labels.add(idx)
            ax.scatter(x, y, c=color, s=30, zorder=3, edgecolors="white", label=label)
        if legend_kwargs is not None:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_order = list(reversed(TICK_LABELS))  # SW, MW, N, MD, SD
            ordered_handles = [by_label[lab] for lab in legend_order if lab in by_label]
            ordered_labels = [lab for lab in legend_order if lab in by_label]
            if ordered_labels:
                ax.legend(ordered_handles, ordered_labels, **legend_kwargs)
    else:
        if color_options is None:
            color_options = {
                "positive": "#689B8A",
                "negative": "#E43636",
                "zero": "lightgray",
            }
        colors = np.where(
            data.values > 0.33,
            color_options["positive"],
            np.where(
                data.values < -0.33, color_options["negative"], color_options["zero"]
            ),
        )
        for x, y, color in zip(data.index, data.values, colors):
            ax.vlines(x, 0, y, colors=color, linewidth=1, **kwargs)
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


def _filter_improvements(max_corr_improvment: list[np.ndarray]) -> np.ndarray:
    """每个窗口取平均改进值；超出 [0, 1] 的置为 NaN（视为无效）。
    返回数组长度与输入严格一致——每个输入恰好对应一个输出。

    Args:
        max_corr_improvment: 每个时间窗口的改进值数组列表。

    Returns:
        np.ndarray: 各窗口的平均改进值，越界处为 NaN。
    """
    out = [
        m if 0 <= m <= 1 else np.nan
        for m in (float(np.asarray(a).mean()) for a in max_corr_improvment)
    ]
    return np.array(out, dtype=float)


@with_axes(figsize=(8, 3.5))
def plot_correlation_windows(
    max_corr_year: list[np.ndarray],
    max_corr_improvment: list[np.ndarray],
    mid_years: list[int],
    slice_labels: list[str] | None = None,
    p_value_list: list[float] | None = None,
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

    corr_improvements = _filter_improvements(max_corr_improvment)
    valid_mask = ~np.isnan(corr_improvements)

    # 设置颜色映射
    if np.any(valid_mask):
        vmin, vmax = 0, np.nanmax(corr_improvements)
        # 确保 vmax > vmin 避免颜色映射问题
        if vmax <= vmin:
            vmax = vmin + 0.1
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
        if p_value_list is not None:
            ax.text(
                mid,
                0,
                get_significance_stars(p_value_list[i]),
                fontsize=8,
                ha="center",
                va="bottom",
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
        cbar.set_label("Avg. Improvement of $Tau$", rotation=270, labelpad=15)
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
    band_label: str = "±1 RMSE",
    band_center: pd.Series | None = None,
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

    # 绘制误差/置信带：默认围绕低通线，可用 band_center 指定围绕的中心序列
    # （例如传入后验均值序列以画出指数本身的可信区间）
    if show_rmse and rmse_data is not None:
        center = filtered_data if band_center is None else band_center
        valid_mask = ~center.isna()
        if valid_mask.any():
            ax.fill_between(
                center.index[valid_mask],
                center[valid_mask] - rmse_data[valid_mask],
                center[valid_mask] + rmse_data[valid_mask],
                color=colors["rmse"],
                alpha=0.3,
                label=band_label,
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
    cbar_label: str | None = "Pearson r",
) -> plt.Axes:
    """绘制相关性地图

    Args:
        corr (xr.DataArray): 相关性
        p_value (xr.DataArray): p值
        threshold (float, optional): 显著性阈值. Defaults to 0.05.
        ax (plt.Axes | None, optional): 坐标轴. Defaults to None.
        mask (bool, optional): 是否仅显示显著相关格点. Defaults to True.
        base_maps (dict[str, str] | None, optional): 底图 shapefile 路径.
        crs (str | None | CRS, optional): 坐标参考系.
        add_colorbar (bool, optional): 是否添加颜色条. Defaults to True.
        cbar_label (str | None, optional): 颜色条标题；覆盖数据变量自带名称.

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
    corr_map = corr_map.copy()
    corr_map.name = "pearson_r"
    corr_map.attrs.pop("long_name", None)
    corr_map.attrs.pop("units", None)
    if add_colorbar:
        cbar_kwargs = {"shrink": 0.8, "aspect": 20, "pad": 0.12}
    else:
        cbar_kwargs = None
    mesh = corr_map.plot(
        ax=ax,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
    )
    if add_colorbar and cbar_label:
        cbar = mesh.colorbar
        cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.tick_right()
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


def _prepare_mismatch_data(
    diff_df: pd.DataFrame,
    count_df: pd.DataFrame,
    pval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare and merge mismatch data into a long-format DataFrame.

    Args:
        diff_df: Difference values per (true, pred) pair.
        count_df: Count values per (true, pred) pair.
        pval_df: P-values per (true, pred) pair.

    Returns:
        Long-format DataFrame with all pairs and computed metrics.
    """
    # Rename axes to avoid level_0/level_1 after reset_index
    diff_df = diff_df.rename_axis(index="true", columns="pred")
    count_df = count_df.rename_axis(index="true", columns="pred")
    pval_df = pval_df.rename_axis(index="true", columns="pred")

    # Create base with all 20 pairs (exclude diagonal)
    all_pairs = [(t, p) for t in LEVELS for p in LEVELS if t != p]
    base = pd.DataFrame(all_pairs, columns=["true", "pred"])

    # Convert to long format and merge
    diff_long = diff_df.stack().reset_index(name="diff")
    cnt_long = count_df.stack().reset_index(name="count")
    p_long = pval_df.stack().reset_index(name="p_value")

    long = (
        base.merge(diff_long, on=["true", "pred"], how="left")
        .merge(cnt_long, on=["true", "pred"], how="left")
        .merge(p_long, on=["true", "pred"], how="left")
    )

    # Compute offset and labels
    long = long.assign(
        offset=lambda d: d["true"] - d["pred"],
        abs_offset=lambda d: (d["true"] - d["pred"]).abs(),
        label=lambda d: d["true"].astype(str) + "-" + d["pred"].astype(str),
    )

    # Fill missing values
    long["diff"] = long["diff"].fillna(0)
    long["count"] = long["count"].fillna(0)

    # Sort by absolute offset, then signed offset
    long = long.sort_values(["abs_offset", "offset", "pred", "true"])

    return long


def _assign_groups_and_labels(long: pd.DataFrame) -> pd.DataFrame:
    """Assign group buckets and pair labels to long-format data.

    Args:
        long: Long-format DataFrame from _prepare_mismatch_data.

    Returns:
        DataFrame with group, group_label, and pair_label columns added.
    """
    # Define groups excluding extreme offsets
    groups = [o for o in range(-len(LEVELS) + 1, len(LEVELS)) if o not in [-4, 4]]
    group_labels = {o: ("-" * abs(o) if o < 0 else "+" * abs(o)) for o in groups}
    level_to_name = {lvl: name for lvl, name in zip(LEVELS, TICK_LABELS)}

    long = long.assign(
        group=lambda d: d["offset"],
        group_label=lambda d: d["offset"].map(group_labels),
        pair_label=lambda d: d.apply(
            lambda r: f"{level_to_name[r['pred']]}\n→\n{level_to_name[r['true']]}",
            axis=1,
        ),
    )

    # Keep only defined groups
    long = long[long["group"].isin(groups)]
    long = long.sort_values(["group", "true", "pred"])

    return long


def _compute_bar_positions(long: pd.DataFrame, groups: list) -> tuple[dict, dict]:
    """Compute x-axis positions for bars within each group.

    Args:
        long: Long-format DataFrame with group column.
        groups: List of group identifiers.

    Returns:
        Tuple of (centers dict, offsets dict) for bar positioning.
    """
    group_counts = long.groupby("group").size().reindex(groups, fill_value=0)
    centers = {g: i for i, g in enumerate(groups)}
    bar_width = 0.8
    half = bar_width / 2
    offsets = {}

    for g in groups:
        n = int(group_counts.loc[g])
        if n <= 0:
            continue
        xs = np.linspace(
            centers[g] - half + bar_width / (2 * n),
            centers[g] + half - bar_width / (2 * n),
            n,
        )
        offsets[g] = list(xs)

    return centers, offsets


def _compute_y_value(
    count_val: float,
    diff_val: float,
    y_metric: str,
    y_weight: float,
    weighted_signed: bool,
    total_count: float,
    max_abs_diff: float,
    contrib_den: float,
) -> float:
    """Compute y-axis value based on selected metric.

    Args:
        count_val: Count for this pair.
        diff_val: Diff value for this pair.
        y_metric: Metric type (proportion, count, diff, etc.).
        y_weight: Weight for weighted metric.
        weighted_signed: Whether weighted metric keeps sign.
        total_count: Total mismatch count.
        max_abs_diff: Maximum absolute diff value.
        contrib_den: Denominator for normalized contribution.

    Returns:
        Computed y-axis value.
    """
    if y_metric == "count":
        return count_val
    elif y_metric == "diff":
        return diff_val
    elif y_metric == "abs_diff":
        return abs(diff_val)
    elif y_metric == "weighted":
        prop = count_val / total_count
        norm_abs_diff = 0.0 if max_abs_diff == 0 else abs(diff_val) / max_abs_diff
        y = float(y_weight) * prop + (1.0 - float(y_weight)) * norm_abs_diff
        if weighted_signed:
            y = (1 if diff_val >= 0 else -1) * y
        return y
    elif y_metric == "contribution":
        return (count_val / total_count) * diff_val
    elif y_metric == "abs_contribution":
        return (count_val / total_count) * abs(diff_val)
    elif y_metric == "norm_contribution":
        return (count_val * diff_val) / contrib_den
    else:  # "proportion"
        return count_val / total_count


def _draw_single_bar(
    ax: Axes,
    x: float,
    y: float,
    bar_width: float,
    n_bars: int,
    diff_val: float,
    p_value: float,
    pair_label: str,
    show_pair_labels: bool,
    pos_color: str = "#D73027",
    neg_color: str = "#225EA8",
) -> None:
    """Draw a single bar with appropriate styling.

    Args:
        ax: Matplotlib axes.
        x: X-axis position.
        y: Y-axis value (bar height).
        bar_width: Total width allocated to this group.
        n_bars: Number of bars in this group.
        diff_val: Diff value to determine color.
        p_value: P-value to determine significance.
        pair_label: Label to show on bar.
        show_pair_labels: Whether to show pair labels.
        pos_color: Color for positive values.
        neg_color: Color for negative values.
    """
    face = pos_color if diff_val >= 0 else neg_color
    marker = get_significance_stars(p_value)
    is_sig = marker != ""
    width = bar_width / max(1, n_bars)

    if is_sig:
        # Filled bar for significant results
        ax.bar(
            x, y, width=width, facecolor=face, edgecolor=face, linewidth=1.0, alpha=1.0
        )
    else:
        # Hollow bar for non-significant results
        ax.bar(
            x,
            y,
            width=width,
            facecolor="none",
            edgecolor=face,
            linewidth=1.2,
            alpha=1.0,
        )

    # Add pair label if requested
    if show_pair_labels:
        txt = ax.text(
            x, max(0, y), pair_label, ha="center", va="bottom", fontsize=6, rotation=0
        )
        txt.set_linespacing(0.8)

    # Add significance marker
    if marker:
        va = "top" if y < 0 else "bottom"
        ax.text(x, y, marker, ha="center", va=va, fontsize=9, color="black")


def _set_y_axis_label(
    ax: Axes, y_metric: str, y_weight: float, weighted_signed: bool
) -> None:
    """Set appropriate y-axis label based on metric type.

    Args:
        ax: Matplotlib axes.
        y_metric: Metric type.
        y_weight: Weight for weighted metric.
        weighted_signed: Whether weighted metric keeps sign.
    """
    labels = {
        "count": "Mismatch count",
        "diff": "Diff value",
        "abs_diff": "Absolute diff value",
        "contribution": "Signed contribution: P(pair) × diff",
        "abs_contribution": "Magnitude contribution: P(pair) × |diff|",
        "norm_contribution": "Normalized signed contribution",
        "proportion": "Proportion of mismatches",
    }

    if y_metric == "weighted":
        prefix = "Signed " if weighted_signed else ""
        ax.set_ylabel(f"{prefix}Weighted (w={y_weight:.2f})")
    else:
        ax.set_ylabel(labels.get(y_metric, "Proportion of mismatches"))


def _add_legend(ax: Axes, legend_loc: str) -> None:
    """Add legend showing color and style meanings.

    Args:
        ax: Matplotlib axes.
        legend_loc: Location string for legend.
    """
    handles = [
        Patch(facecolor="#D73027", edgecolor="#D73027", label="Positive (sig.)"),
        Patch(facecolor="#225EA8", edgecolor="#225EA8", label="Negative (sig.)"),
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="#D73027",
            linewidth=1.2,
            label="Positive",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="#225EA8",
            linewidth=1.2,
            label="Negative",
        ),
    ]
    ax.legend(
        title="Legend",
        title_fontsize=8,
        handles=handles,
        loc=legend_loc,
        frameon=True,
        fontsize=7,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.8,
        labelspacing=0.4,
        borderpad=0.2,
    )


@with_axes(figsize=(9, 2.6))
def plot_mismatch_bar(
    diff_df: pd.DataFrame,
    count_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    ax: Optional[Axes] = None,
    show_pair_labels: bool = False,
    show_legend: bool = False,
    legend_loc: str = "upper right",
    y_metric: str = "proportion",
    y_weight: float = 0.5,
    weighted_signed: bool = False,
    **kwargs,
) -> Axes:
    """Plot mismatch bars for all (true, pred) pairs without aggregation.

    This function visualizes mismatch patterns by drawing 20 bars (excluding
    diagonal pairs), grouped by the absolute level difference between predicted
    and true values. The bars are colored based on the sign of the difference
    (red for positive, blue for negative), with filled bars indicating
    statistically significant results and hollow bars for non-significant ones.

    Args:
        diff_df: Values to display on the y-axis per (true, pred).
        count_df: Count matrix (kept for completeness; not used for color).
        pval_df: Per-cell p-values to determine significance.
        ax: Matplotlib axes.
        show_pair_labels: Whether to annotate each bar with label like
            "SD→MW" on top of the bar.
        show_legend: Whether to show legend.
        legend_loc: Location of legend.
        y_metric: Which quantity to plot on y-axis. One of:
            - "proportion" (default): count / total mismatches
            - "count": raw count
            - "diff": signed diff value
            - "abs_diff": absolute diff value
            - "weighted": composite of proportion and |diff| normalized
              as: y = w * proportion + (1-w) * (|diff|/max_abs_diff).
              If `weighted_signed` is True, multiply by sign(diff).
            - "contribution": (count/total) * diff; signed expected
              mismatch contribution per event
            - "abs_contribution": (count/total) * |diff|; magnitude
              contribution ignoring sign
            - "norm_contribution": (count*diff) / sum(count*|diff|);
              signed, normalized to [-1, 1] when diff is bounded
        y_weight: Weight w in [0,1] for the composite metric when
            y_metric="weighted".
        weighted_signed: Whether the weighted metric keeps the sign of diff.

    Returns:
        The axes with the plot.
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"

    # Prepare data
    long = _prepare_mismatch_data(diff_df, count_df, pval_df)
    long = _assign_groups_and_labels(long)

    # Define groups and compute positions
    groups = [o for o in range(-len(LEVELS) + 1, len(LEVELS)) if o not in [-4, 4]]
    group_labels = {o: ("-" * abs(o) if o < 0 else "+" * abs(o)) for o in groups}
    centers, offsets = _compute_bar_positions(long, groups)

    # Compute normalization constants
    max_abs_diff = (
        np.nanmax(np.abs(diff_df.values))
        if np.isfinite(np.nanmax(np.abs(diff_df.values)))
        else 1.0
    )
    total_count = np.nansum(count_df.values)
    if not np.isfinite(total_count) or total_count == 0:
        total_count = 1.0
    contrib_den = np.nansum(np.abs(count_df.values * diff_df.values))
    if not np.isfinite(contrib_den) or contrib_den == 0:
        contrib_den = 1.0

    # Set y-axis limits if specified
    if "ylim" in kwargs:
        ax.set_ylim(*kwargs.get("ylim"))

    # Draw bars for each group
    bar_width = 0.8
    for g, sub in long.groupby("group", sort=False):
        if g not in offsets:
            continue
        xs = offsets[g]
        for i, (_, row) in enumerate(sub.iterrows()):
            x = xs[i]
            count_val = 0.0 if pd.isna(row["count"]) else float(row["count"])
            diff_val = 0.0 if pd.isna(row["diff"]) else float(row["diff"])

            # Compute y-value based on metric
            y = _compute_y_value(
                count_val,
                diff_val,
                y_metric,
                y_weight,
                weighted_signed,
                total_count,
                max_abs_diff,
                contrib_den,
            )

            # Draw the bar
            _draw_single_bar(
                ax,
                x,
                y,
                bar_width,
                len(xs),
                diff_val,
                row["p_value"],
                row["pair_label"],
                show_pair_labels,
            )

    # Configure x-axis
    tick_positions = [centers[g] for g in groups]
    tick_labels_list = [group_labels[g] for g in groups]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list)
    ax.set_xlabel("Relative level difference (pred - true)")

    # Add reference line and separators
    ax.axhline(0, color="black", linewidth=0.6)
    ymin, ymax = ax.get_ylim()
    for i in range(len(groups) - 1):
        x = (centers[groups[i]] + centers[groups[i + 1]]) / 2
        ax.vlines(
            x, ymin, ymax, colors="lightgray", linestyles=":", linewidth=1.6, alpha=0.7
        )
    ax.set_ylim(ymin, ymax)

    # Set y-axis label
    _set_y_axis_label(ax, y_metric, y_weight, weighted_signed)

    # Add legend if requested
    if show_legend:
        _add_legend(ax, legend_loc)

    return ax
