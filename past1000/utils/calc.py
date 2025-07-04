#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import detrend

if TYPE_CHECKING:
    from past1000.utils.types import CorrFunc


def get_coords(mask: np.ndarray) -> list[tuple[int, ...]]:
    """
    获取数组中符合条件的坐标列表。

    Args:
        mask: 布尔数组

    Examples:
        >>> mask = np.array([[True, False, True], [False, True, False]])
        >>> get_coords(mask)
        [(0, 0), (0, 2), (1, 1)]

    Returns:
        list[tuple[int, ...]]: 符合条件的坐标列表
    """
    coords = np.where(mask)
    if coords[0].size == 0:
        return []
    return list(zip(*coords))


def detrend_with_nan(data: pd.Series, **kwargs) -> pd.Series:
    """
    去除数据中的 NaN 值，并进行去趋势处理。

    Args:
        data: 输入数据

    Returns:
        np.ndarray: 去趋势处理后的数据
    """
    dropped_nan_data = data.dropna()
    detrend_data = pd.Series(
        detrend(dropped_nan_data.values, **kwargs),
        index=dropped_nan_data.index,
    ).reindex(data.index)
    return detrend_data


def get_significance_stars(p_value: float) -> str:
    """
    根据 p 值决定添加什么星号

    Args:
        p_value: p 值，0-1 之间

    Returns:
        str: 星号
    """
    if pd.isna(p_value):
        return ""
    if not 0 <= p_value <= 1:
        raise ValueError(f"p must be between 0 and 1, but got {p_value}")
    if p_value < 0.05:
        return "**"  # p < 0.05, 两个星号
    if p_value < 0.1:
        return "*"  # 0.05 <= p < 0.1, 一个星号
    return ""


def align_matrices(*matrices: pd.DataFrame) -> list[pd.DataFrame]:
    """
    对齐一个或多个 DataFrame 矩阵，使它们具有相同的行和列索引。

    这个函数会找到所有输入矩阵的索引和列的并集，
    然后使用 .reindex() 方法将每个矩阵扩展到这个统一的维度。

    Parameters:
    -----------
    *matrices : pd.DataFrame
        一个或多个需要对齐的 pandas DataFrame。

    Returns:
    --------
    list[pd.DataFrame]
        一个包含所有已对齐的 DataFrame 的列表。
    """
    if not matrices:
        return []

    # 使用 reduce 和 set.union 高效地找到所有索引和列的并集
    all_indices = reduce(lambda x, y: x.union(y.index), matrices, pd.Index([]))
    all_columns = reduce(lambda x, y: x.union(y.columns), matrices, pd.Index([]))

    # 找到行和列的全集
    all_levels = sorted(list(all_indices.union(all_columns)))

    # 对每个矩阵应用 .reindex
    aligned_matrices = [
        m.reindex(index=all_levels, columns=all_levels) for m in matrices
    ]

    return aligned_matrices


def fill_star_matrix(p_values: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """
    根据 p 值矩阵填充星号矩阵
    """
    annot_labels = pd.DataFrame(index=values.index, columns=values.columns, dtype=str)

    # 填充标签矩阵
    for idx in values.index:
        for col in values.columns:
            value = values.loc[idx, col]
            p_value = p_values.loc[idx, col]

            if pd.isna(value):
                annot_labels.loc[idx, col] = ""  # 如果没有 diff 值，则不显示任何内容
            else:
                stars = get_significance_stars(p_value)
                annot_labels.loc[idx, col] = f"{value:.2f}{stars}"  # 格式：数值+星号
    return annot_labels


def effective_sample_size(n, arr1, arr2):
    """计算有效样本量
    计算公式：
    neff = n * (1 - acf1 * acf2) / (1 + acf1 * acf2)
    其中，n 是样本量，acf1 和 acf2 是两个序列的自相关系数。参考：https://www.cnblogs.com/yongh/p/11060111.html

    Args:
        n: 样本量
        arr1: 序列1
        arr2: 序列2

    Returns:
        int: 有效样本量
    """
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
