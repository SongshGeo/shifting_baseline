#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
计算两个数据集之间的相关系数
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from rasterio.enums import Resampling
from scipy import stats

if TYPE_CHECKING:
    from past1000.core.types import CorrFunc


def summer_precipitation(pr_data: xr.DataArray):
    """获取夏季（6-8月）降水量 # TODO: 6-9月

    Args:
        data: 数据集

    Returns:
        xr.DataArray: 夏季降水量数据，每年都是 3 个月数据的和
    """
    basin_mask = pr_data.isel(time=0).notnull()
    # 选择 6-8 月的数据
    return (
        pr_data.sel(time=pr_data.time.dt.month.isin([6, 7, 8]))
        .groupby("time.year")
        .sum(dim="time", skipna=True)
        .where(basin_mask)
        .reset_coords(names="time", drop=True)
        .rename({"year": "time"})
    )


def corr(x, y) -> np.ndarray:
    """计算两个数组之间的相关系数"""
    return np.corrcoef(x, y)[0, 1]


def reproject_match(
    da: xr.DataArray,
    target: xr.DataArray,
    resampling: str = "bilinear",
    dims_name: Tuple[str, str] = ("lat", "lon"),
) -> xr.DataArray:
    """重采样数据集以匹配目标数据集

    Args:
        da: 需要重采样的数据集，维度为 (time, lat, lon)
        target: 目标数据集，维度为 (time, lat, lon)
        resampling: 重采样方法
    """
    y_dim, x_dim = dims_name
    # 重命名维度并确保顺序正确
    return da.rio.reproject_match(
        target,
        resampling=Resampling[resampling],
    ).rename(
        {
            "y": y_dim,
            "x": x_dim,
        }
    )


def xarray_corr(
    da1: xr.DataArray,
    da2: xr.DataArray,
    resampling: str = "nearest",
    reproject_to: Optional[Literal["da1", "da2"]] = None,
    time_dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    """计算两个数据集之间的相关系数"""
    # 选择共同年份
    da1, da2 = xr.align(da1, da2, join="inner")
    # 重投影
    if reproject_to is not None:
        # pass
        if reproject_to == "da1":
            da2 = reproject_match(da2, da1, resampling)
        elif reproject_to == "da2":
            da1 = reproject_match(da1, da2, resampling)
        else:
            raise ValueError(f"Invalid reproject value: {reproject_to}")

    if da1.shape != da2.shape:
        raise ValueError("The two datasets have different shapes.")
    if da1[time_dim].size != da2[time_dim].size:
        raise ValueError("The two datasets have different time dimensions.")

    return xr.apply_ufunc(
        corr,
        da1,
        da2,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[]],  # 输出没有时间维度
        vectorize=True,
        **kwargs,
    )


def merge_comparable_data(
    *data: pd.DataFrame | pd.Series, index: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """合并可比较的数据"""
    if index is None:
        index = np.arange(850, 2000)
    df = pd.DataFrame(index=index)
    for d in data:
        df = pd.merge(
            df,
            right=d,
            how="left",
            left_index=True,
            right_index=True,
        )
    return df


def plot_corr(df: pd.DataFrame) -> None:
    """绘制相关系数"""
    dataplot = sns.heatmap(
        df.corr(numeric_only=True),
        cmap="vlag",
        annot=True,
        annot_kws={"size": 8},
        fmt=".1f",
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    dataplot.set_title("Correlation Coefficient")
    return dataplot


def effective_sample_size(n, arr1, arr2):
    acf1 = arr1.autocorr(lag=1)
    acf2 = arr2.autocorr(lag=1)
    # 如果自相关无法计算，直接返回1
    if np.isnan(acf1) or np.isnan(acf2):
        return 1
    denom = 1 + acf1 * acf2
    if denom == 0:
        return 1
    neff = n * (1 - acf1 * acf2) / denom
    # 防止neff为负或为nan
    if not np.isfinite(neff) or neff <= 0:
        return 1
    return int(neff)


def calc_corr(
    arr1: pd.Series,
    arr2: pd.Series,
    how: CorrFunc = "pearson",
    penalty: bool = False,
) -> tuple[float, float, int]:
    """计算两个序列之间的相关系数"""
    # 确保两个序列都是数值类型
    arr1 = pd.to_numeric(arr1, errors="coerce")
    arr2 = pd.to_numeric(arr2, errors="coerce")
    # 使用pandas的isna()方法
    mask = ~arr1.isna() & ~arr2.isna()
    n = mask.sum()
    # 计算相关系数
    valid_arr1 = arr1[mask]
    valid_arr2 = arr2[mask]
    if how == "pearson":
        r, p = stats.pearsonr(valid_arr1, valid_arr2)
    elif how == "kendall":
        r, p = stats.kendalltau(valid_arr1, valid_arr2)
    elif how == "spearman":
        r, p = stats.spearmanr(valid_arr1, valid_arr2)
    else:
        raise ValueError(f"无效的相关系数计算方法: {how}")
    if penalty:
        neff = effective_sample_size(n, valid_arr1, valid_arr2)
        penalty = np.sqrt(neff / n)
        r = r * penalty
    return r, p, n
