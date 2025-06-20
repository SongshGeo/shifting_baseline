#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotkit import with_axes
from matplotlib.axes import Axes
from scipy.stats import kendalltau
from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score

from past1000.utils.calc import get_coords

TICK_LABELS = ["SD", "MD", "N", "MW", "SW"]


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
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    ax: Optional[Axes] = None,
    dropna: bool = False,
    **kwargs,
) -> None:
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
            kwargs for ConfusionMatrixDisplay.from_predictions
    """
    assert isinstance(ax, Axes), "ax must be an instance of Axes"
    labels = [-2, -1, 0, 1, 2]
    # TODO: 简化这个逻辑
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series) and dropna:
        combined = pd.concat([y_true, y_pred], axis=1).dropna(axis=0)
        y_true = combined.iloc[:, 0]
        y_pred = combined.iloc[:, 1]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if dropna:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=labels,
        ax=ax,
        cmap=plt.cm.Reds,
        # normalize="pred",
    )
    ax.set_xticklabels(TICK_LABELS)
    ax.set_yticklabels(TICK_LABELS)

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=labels,
        weights="quadratic",
        **kwargs,
    )

    # 3. 计算 Kendall's Tau
    tau, p_value = kendalltau(y_true, y_pred)

    title = f"Kappa: {kappa:.2f}, Kendall's Tau: {tau:.2f}"
    if p_value < 0.05:
        title += "**"
    ax.set_title(title)
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
        vmin=v_min,  # 让基准值在中间
        # vmin=r_benchmark,  # 让基准值在最下面
        vmax=v_max,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        square=True,
        **kwargs,
    )
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Minimum Periods")
    # 设置边框
    sns.despine(
        ax=ax,
        top=False,
        right=False,
        left=False,
        bottom=False,
        trim=False,
    )
    ax.locator_params(axis="both", nbins=9)  # x轴最多7个主刻度
    if r_benchmark is not None:
        # 获取 colorbar 对象
        cbar = ax.collections[0].colorbar
        # 添加水平参考线
        cbar.ax.axhline(y=r_benchmark, color="black", linewidth=2)  # 在中间位
        cbar.ax.set_yticks(np.linspace(v_min, v_max, 5))
        # 2位小数点
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(v_min, v_max, 5)])
    min_period = filtered.index.values.min()
    max_period = filtered.index.values.max()
    min_window = filtered.columns.values.min()
    max_window = filtered.columns.values.max()
    ax.plot(
        [min_window, max_period],
        [min_period, max_window],
        color="black",
        linestyle=":",
        linewidth=1,
        label="window = period",
    )
    std_value = np.nanstd(filtered)
    lower_bound = v_max - std_offset * std_value
    points = get_coords(filtered >= lower_bound)
    # points = get_coords(filtered == max_value)
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
    ax.legend(loc="lower left", fontsize=8)
    return ax
