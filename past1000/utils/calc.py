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
from scipy.stats import truncnorm

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


def effective_sample_size(n: int, arr1: pd.Series, arr2: pd.Series) -> int:
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
    arr1: pd.Series | np.ndarray,
    arr2: pd.Series | np.ndarray,
    how: CorrFunc = "pearson",
    penalty: bool = False,
) -> tuple[float, float, int]:
    """计算两个序列之间的相关系数

    Args:
        arr1: 第一个序列，支持pandas Series或numpy数组
        arr2: 第二个序列，支持pandas Series或numpy数组
        how: 相关系数计算方法，可选'pearson', 'kendall', 'spearman'
        penalty: 是否应用有效样本量惩罚

    Returns:
        tuple[float, float, int]: (相关系数, p值, 有效样本数)

    Raises:
        ValueError: 当输入数据无效或计算方法不支持时
        TypeError: 当输入类型不支持时
    """
    # 输入验证
    if arr1 is None or arr2 is None:
        raise ValueError("输入序列不能为None")

    # 确保两个序列长度相同
    if len(arr1) != len(arr2):
        raise ValueError(f"两个序列长度必须相同，但得到 {len(arr1)} 和 {len(arr2)}")

    # 高效处理：根据输入类型选择最优路径
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        # 两个都是numpy数组，使用高效路径
        return _calc_corr_numpy(arr1, arr2, how, penalty)
    else:
        # 包含pandas Series，使用pandas路径
        return _calc_corr_pandas(arr1, arr2, how, penalty)


def _calc_corr_numpy(
    arr1: np.ndarray,
    arr2: np.ndarray,
    how: CorrFunc,
    penalty: bool,
) -> tuple[float, float, int]:
    """高效处理numpy数组的相关系数计算"""
    # 转换为数值类型（如果还不是的话）
    try:
        arr1 = np.asarray(arr1, dtype=float)
        arr2 = np.asarray(arr2, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"无法将输入转换为数值类型: {e}") from e

    # 找到有效数据点（使用numpy的高效操作）
    mask = np.isfinite(arr1) & np.isfinite(arr2)
    n = np.sum(mask)

    # 检查有效样本数
    if n <= 2:
        return np.nan, np.nan, n

    # 提取有效数据
    valid_arr1 = arr1[mask]
    valid_arr2 = arr2[mask]

    # 检查数据是否全为常数（使用numpy的高效操作）
    if np.var(valid_arr1) == 0 or np.var(valid_arr2) == 0:
        return np.nan, np.nan, n

    # 计算相关系数
    try:
        if how == "pearson":
            r, p = stats.pearsonr(valid_arr1, valid_arr2)
        elif how == "kendall":
            r, p = stats.kendalltau(valid_arr1, valid_arr2)
        elif how == "spearman":
            r, p = stats.spearmanr(valid_arr1, valid_arr2)
        else:
            raise ValueError(f"无效的相关系数计算方法: {how}")
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise ValueError(f"计算相关系数时出错: {e}") from e

    # 应用有效样本量惩罚
    if penalty:
        try:
            # 对于numpy数组，需要转换为pandas Series来计算自相关
            valid_series1 = pd.Series(valid_arr1)
            valid_series2 = pd.Series(valid_arr2)
            neff = effective_sample_size(n, valid_series1, valid_series2)
            penalty_factor = np.sqrt(neff / n)
            r = r * penalty_factor
        except (ValueError, RuntimeError, FloatingPointError) as e:
            # 如果惩罚计算失败，记录警告但继续返回原始结果
            import warnings

            warnings.warn(f"有效样本量惩罚计算失败: {e}")

    return r, p, n


def _calc_corr_pandas(
    arr1: pd.Series | np.ndarray,
    arr2: pd.Series | np.ndarray,
    how: CorrFunc,
    penalty: bool,
) -> tuple[float, float, int]:
    """处理包含pandas Series的相关系数计算"""
    # 统一转换为pandas Series进行处理，保持索引对齐
    if isinstance(arr1, np.ndarray):
        arr1 = pd.Series(arr1)
    if isinstance(arr2, np.ndarray):
        arr2 = pd.Series(arr2)

    # 确保两个序列都是数值类型
    try:
        arr1 = pd.to_numeric(arr1, errors="coerce")
        arr2 = pd.to_numeric(arr2, errors="coerce")
    except (TypeError, ValueError) as e:
        raise ValueError(f"无法将输入转换为数值类型: {e}") from e

    # 找到有效数据点
    mask = ~arr1.isna() & ~arr2.isna()
    n = mask.sum()

    # 检查有效样本数
    if n <= 2:
        return np.nan, np.nan, n

    # 提取有效数据
    valid_arr1 = arr1[mask]
    valid_arr2 = arr2[mask]

    # 检查数据是否全为常数
    if valid_arr1.nunique() <= 1 or valid_arr2.nunique() <= 1:
        return np.nan, np.nan, n

    # 计算相关系数
    try:
        if how == "pearson":
            r, p = stats.pearsonr(valid_arr1, valid_arr2)
        elif how == "kendall":
            r, p = stats.kendalltau(valid_arr1, valid_arr2)
        elif how == "spearman":
            r, p = stats.spearmanr(valid_arr1, valid_arr2)
        else:
            raise ValueError(f"无效的相关系数计算方法: {how}")
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise ValueError(f"计算相关系数时出错: {e}") from e

    # 应用有效样本量惩罚
    if penalty:
        try:
            neff = effective_sample_size(n, valid_arr1, valid_arr2)
            penalty_factor = np.sqrt(neff / n)
            r = r * penalty_factor
        except (ValueError, RuntimeError, FloatingPointError) as e:
            # 如果惩罚计算失败，记录警告但继续返回原始结果
            import warnings

            warnings.warn(f"有效样本量惩罚计算失败: {e}")

    return r, p, n


def find_top_max_indices(
    arr: np.ndarray,
    ratio: float = 0.1,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    找到numpy数组中前10%大值所在的索引（忽略nan值）

    参数:
    arr: numpy数组

    返回:
    numpy数组，包含前10%大值的索引
    """
    # 将数组展平
    flat_arr = arr.flatten()

    # 找到非nan值的索引
    valid_mask = ~np.isnan(flat_arr)
    valid_indices = np.where(valid_mask)[0]
    valid_values = flat_arr[valid_mask]

    # 如果没有有效值，返回空数组
    if len(valid_values) == 0:
        if arr.ndim > 1:
            return tuple(np.array([]) for _ in range(arr.ndim))
        else:
            return np.array([])

    # 计算前10%的元素个数（基于有效值）
    top_10_percent_count = max(1, int(len(valid_values) * ratio))

    # 使用argpartition找到前10%大值的索引（在有效值中）
    top_indices_in_valid = np.argpartition(valid_values, -top_10_percent_count)[
        -top_10_percent_count:
    ]

    # 映射回原数组的索引
    flat_indices = valid_indices[top_indices_in_valid]

    # 如果需要返回原数组形状的索引，可以使用unravel_index
    if arr.ndim > 1:
        # 返回多维索引
        return np.unravel_index(flat_indices, arr.shape)
    else:
        # 返回一维索引
        return flat_indices


def low_pass_filter(
    data: pd.Series | np.ndarray,
    window_size: int = 30,
    method: str = "rolling_mean",
    center: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Apply low-pass filter to time series data.

    This function applies a low-pass filter to remove high-frequency noise
    and reveal long-term trends in time series data. The default 30-year
    window is commonly used in paleoclimate studies.

    Args:
        data: Input time series data
        window_size: Size of the rolling window for filtering (default: 30)
        method: Filtering method ('rolling_mean', 'gaussian', 'butterworth')
        center: Whether to center the rolling window (default: True)
        min_periods: Minimum number of observations in window required to have a value

    Returns:
        pd.Series: Filtered time series data

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.Series(np.random.randn(100), index=range(1000, 1100))
        >>> filtered = low_pass_filter(data, window_size=10)
        >>> print(filtered.head())
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if min_periods is None:
        min_periods = max(1, window_size // 2)

    if method == "rolling_mean":
        return data.rolling(
            window=window_size, center=center, min_periods=min_periods
        ).mean()
    elif method == "gaussian":
        # Gaussian filter using scipy
        from scipy.ndimage import gaussian_filter1d

        sigma = window_size / 6  # 3-sigma rule
        filtered_values = gaussian_filter1d(data.dropna().values, sigma=sigma)
        result = pd.Series(index=data.index, dtype=float)
        result.loc[data.dropna().index] = filtered_values
        return result
    elif method == "butterworth":
        # Butterworth low-pass filter
        from scipy.signal import butter, filtfilt

        nyquist = 0.5  # Assuming yearly data
        cutoff = 1.0 / window_size  # Cutoff frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        filtered_values = filtfilt(b, a, data.dropna().values)
        result = pd.Series(index=data.index, dtype=float)
        result.loc[data.dropna().index] = filtered_values
        return result
    else:
        raise ValueError(f"Unknown filtering method: {method}")


def calculate_rmse(
    observed: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    axis: int | None = None,
) -> float:
    """Calculate Root Mean Square Error (RMSE).

    Args:
        observed: Observed values
        predicted: Predicted values
        axis: Axis along which to calculate RMSE

    Returns:
        float: RMSE value
    """
    if isinstance(observed, pd.Series):
        observed = observed.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values

    mse = np.mean((observed - predicted) ** 2, axis=axis)
    return np.sqrt(mse)


def get_interval(level: int) -> tuple[float, float]:
    """根据等级返回标准化下限和上限（标准差倍数）。"""
    boundaries = {
        -2: (-2, -1.17),
        -1: (-1.17, -0.33),
        0: (-0.33, 0.33),
        1: (0.33, 1.17),
        2: (1.17, 2),
    }
    if level not in boundaries:
        raise ValueError("无效等级，必须为 {-2, -1, 0, 1, 2}")
    return boundaries[level]


def rand_generate_from_std_levels(
    grade_matrix: np.ndarray | pd.Series | pd.DataFrame,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    为任意形状的等级数组生成对应的原始值矩阵，每个值从截断正态分布采样。
    跳过 NA 值，在输出中保留 np.nan。

    参数:
    - grade_matrix: 任意形状的 numpy 数组、pandas Series 或 DataFrame。
                    元素为 {-2, -1, 0, 1, 2} 或 NA（字符串/None/pandas NA 均可）
    - mu: 正态分布均值 (默认 0.0)
    - sigma: 正态分布标准差 (默认 1.0)

    返回:
    - numpy 数组，形状与输入相同，非 NA 元素为采样值，NA 元素为 np.nan
    """
    # 统一取 numpy array 视图与原始形状
    if isinstance(grade_matrix, pd.DataFrame):
        arr = grade_matrix.to_numpy()
        orig_shape = arr.shape
    elif isinstance(grade_matrix, pd.Series):
        arr = grade_matrix.to_numpy()
        orig_shape = arr.shape
    elif isinstance(grade_matrix, np.ndarray):
        arr = grade_matrix
        orig_shape = arr.shape
    else:
        raise ValueError("输入必须是 numpy.ndarray、pandas.Series 或 pandas.DataFrame")

    # 为适配任意形状，扁平化到 1D
    flat = arr.ravel()

    # 若为 object 或混合类型，使用 pandas to_numeric 安全转换（非数值与 NA 变为 np.nan）
    if flat.dtype == object:
        coerced = pd.to_numeric(pd.Series(flat), errors="coerce").to_numpy()
    else:
        # 已经是数值类型，复制为浮点
        coerced = flat.astype(float, copy=False)

    # NA 掩膜
    na_mask = np.isnan(coerced)

    # 检查非 NA 元素的有效性
    valid_grades = np.array([-2, -1, 0, 1, 2], dtype=float)
    non_na = coerced[~na_mask]
    if non_na.size > 0 and not np.all(np.isin(non_na, valid_grades)):
        raise ValueError("非 NA 元素包含无效等级，必须为 {-2, -1, 0, 1, 2}")

    # 结果扁平向量
    out = np.full(coerced.shape, np.nan, dtype=float)

    # 对每个等级向量化采样并填充
    for grade in valid_grades:
        idx = (~na_mask) & (coerced == grade)
        cnt = int(np.sum(idx))
        if cnt > 0:
            lower, upper = get_interval(int(grade))
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma
            samples = truncnorm.rvs(a, b, size=cnt)
            out[idx] = mu + samples * sigma

    # 还原为原始形状
    return out.reshape(orig_shape)


def generate_from_2d_levels_averaged(
    grade_matrix, n_samples=100, mu=0.0, sigma=1.0, random_state=None
):
    """
    多次调用 generate_from_2d_levels 并取平均值, 并返回平均值和标准差

    Args:
        grade_matrix: 2D NumPy 数组或 pandas DataFrame，元素为 {-2, -1, 0, 1, 2} 或 NA
        n_samples: 样本数量
        mu: 正态分布均值 (默认 0.0)
        sigma: 正态分布标准差 (默认 1.0)
        random_state: 随机种子

    Returns:
        tuple[np.ndarray, np.ndarray]: 平均值和标准差
    """
    if random_state is not None:
        np.random.seed(random_state)

    samples = []
    for _ in range(n_samples):
        sample = rand_generate_from_std_levels(grade_matrix, mu=mu, sigma=sigma)
        samples.append(sample)

    return np.mean(samples, axis=0), np.std(samples, axis=0)
