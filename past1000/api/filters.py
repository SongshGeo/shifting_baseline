#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""用来针对滑动窗口的滤波方法
"""
from typing import Callable, TypeAlias

import numpy as np
import pandas as pd
from scipy import stats

from past1000.api.mc import standardize_data

ArrayLike: TypeAlias = np.ndarray | pd.Series


def natural_filter(
    data: pd.Series,
    window: int,
    center: bool = False,
    min_periods: int = 3,
    filter_func: Callable | None = None,
    normalize: bool = True,
    **kwargs,
) -> pd.Series:
    """自然滤波"""

    # 定义应用于滚动窗口的函数
    def rolling_filter(series: pd.Series):
        # 如果有预处理则应用滤波方法
        if filter_func is not None:
            series = pd.Series(
                data=filter_func(series, **kwargs), index=series.index, name=series.name
            )
        # 然后再标准化
        normalized_data, _ = standardize_data(series)
        if normalize:
            return normalized_data.iloc[-1]
        else:
            return series.iloc[-1]

    # 应用滚动窗口，对自然数据进行滤波
    return data.rolling(
        window=window,
        center=center,
        min_periods=min_periods,
    ).apply(rolling_filter)


def final_percentile(data: ArrayLike) -> np.ndarray:
    """序列值在当前序列中的百分位数。
    这种滤波方法说明自然数据在当前序列中的"极端"情况。

    Parameters:
        data: 用来检测的数据。

    Return:
        序列最后一个值在当前序列中的百分位数
    """
    return (
        stats.percentileofscore(
            data,
            data,
            kind="rank",
        )
        / 100
    )[-1]


# def exponential_memory_decay(
#     data: ArrayLike,
#     decay_rate: float = 0.05,
#     normalization: bool = False,
#     opt_target: ArrayLike | None = None,
#     **opt_kwargs,
# ) -> np.ndarray:
#     """序列按照指数衰减计算权重，对数据进行加权。
#     w(t) = e^{-\alpha t}
#     """
#     if opt_target is not None:
#         decay_rate, _ = optimize_exponential_decay(
#             data,
#             opt_target,
#             **opt_kwargs,
#         )
#     if decay_rate <= 0:
#         raise ZeroDivisionError("decay_rate must be positive.")
#     # 创建时间序列（从近到远）
#     t = np.arange(len(data))[::-1]
#     # 创建权重
#     weights = np.exp(-decay_rate * t)
#     # 归一化
#     if normalization:
#         weights = weights / weights.sum()
#     # 返回加权计算结果
#     return data * weights


# def two_stages_decaying(
#     data: ArrayLike,
#     p: float,
#     r: float,
#     q: float,
#     normalization: bool = False,
#     opt_target: ArrayLike | None = None,
#     **opt_kwargs,
# ) -> np.ndarray:
#     r"""Two-stage exponential decay function.

#     The function implements the following formula:
#     $$ S(t)=\frac{N}{p+r-q}\left[(p-q) e^{-(p+r) t}+r e^{-q t}\right] $$

#     Args:
#         data: Input data array
#         p: First decay rate parameter
#         r: Second decay rate parameter
#         q: Third decay rate parameter
#         normalization: Whether to normalize the weights

#     Returns:
#         Weighted data array

#     Raises:
#         ValueError: If p + r - q is zero or negative
#     """
#     if opt_target is not None:
#         p, r, q, _ = optimize_two_stages_decay(
#             data,
#             opt_target,
#             **opt_kwargs,
#         )
#     if p + r - q <= 0:
#         raise ValueError("p + r - q must be positive")

#     # Create time series (from recent to past)
#     t = np.arange(len(data))[::-1]

#     # Calculate weights using the formula
#     weights = (p - q) * np.exp(-(p + r) * t) + r * np.exp(-q * t)
#     weights = weights / (p + r - q)

#     # Normalize if requested
#     if normalization:
#         weights = weights / weights.sum()

#     # Return weighted data
#     return data * weights


# def optimize_two_stages_decay(
#     data: ArrayLike,
#     target: ArrayLike,
#     p_range: tuple[float, float] = (0.01, 1.0),
#     r_range: tuple[float, float] = (0.01, 1.0),
#     q_range: tuple[float, float] = (0.01, 1.0),
#     method: str = "differential_evolution",
#     verbose: bool = False,
# ) -> tuple[float, float, float, float]:
#     """Optimize parameters for two_stages_decaying function to maximize correlation.

#     Args:
#         data: Input data array or pandas Series
#         target: Target data array or pandas Series
#         p_range: Range for parameter p (min, max)
#         r_range: Range for parameter r (min, max)
#         q_range: Range for parameter q (min, max)
#         method: Optimization method ('differential_evolution' or 'brute')
#         verbose: Whether to print debugging information

#     Returns:
#         Tuple of (best_p, best_r, best_q, best_correlation)
#     """
#     from scipy import optimize, stats

#     # 数据预处理和索引对齐
#     if isinstance(data, pd.Series) and isinstance(target, pd.Series):
#         # 对齐索引
#         common_index = data.index.intersection(target.index)
#         if len(common_index) == 0:
#             raise ValueError("No common index between data and target")
#         data = data.loc[common_index]
#         target = target.loc[common_index]
#         if verbose:
#             print(f"Aligned data length: {len(common_index)}")
#     else:
#         # 如果不是Series，转换为numpy数组
#         data = np.asarray(data)
#         target = np.asarray(target)
#         if len(data) != len(target):
#             raise ValueError("Input data and target must have the same length")

#     def objective(params):
#         p, r, q = params
#         try:
#             # 检查参数是否在有效范围内
#             if p + r - q <= 0:
#                 if verbose:
#                     print(f"Invalid parameters: p={p:.3f}, r={r:.3f}, q={q:.3f}")
#                 return float("inf")

#             # 计算加权数据
#             weighted_data = two_stages_decaying(data, p, r, q)

#             # 检查结果是否有效
#             if np.any(np.isinf(weighted_data)):
#                 if verbose:
#                     print(
#                         f"Invalid weighted data with parameters: p={p:.3f}, r={r:.3f}, q={q:.3f}"
#                     )
#                 return float("inf")

#             # 处理NaN值
#             mask = ~(np.isnan(weighted_data) | np.isnan(target))
#             if np.sum(mask) < 3:  # 至少需要3个点来计算相关系数
#                 if verbose:
#                     print("Not enough valid points after removing NaN values")
#                 return float("inf")

#             # 计算相关系数
#             correlation = stats.pearsonr(weighted_data[mask], target[mask])[0]

#             if verbose:
#                 print(
#                     f"Parameters: p={p:.3f}, r={r:.3f}, q={q:.3f}, correlation={correlation:.3f}"
#                 )
#                 print(f"Valid points: {np.sum(mask)}/{len(mask)}")

#             return -correlation

#         except Exception as e:
#             if verbose:
#                 print(
#                     f"Error with parameters p={p:.3f}, r={r:.3f}, q={q:.3f}: {str(e)}"
#                 )
#             return float("inf")

#     if method == "differential_evolution":
#         bounds = [p_range, r_range, q_range]
#         result = optimize.differential_evolution(
#             objective,
#             bounds=bounds,
#             maxiter=100,
#             popsize=15,
#             tol=1e-3,
#             mutation=(0.5, 1.0),
#             recombination=0.7,
#             disp=verbose,
#         )

#         if not result.success:
#             if verbose:
#                 print(f"Optimization failed: {result.message}")
#             return (0.1, 0.1, 0.1, 0.0)  # 返回默认值

#         best_params = result.x
#         best_correlation = -result.fun

#     elif method == "brute":
#         # 创建参数网格
#         p_values = np.linspace(p_range[0], p_range[1], 10)
#         r_values = np.linspace(r_range[0], r_range[1], 10)
#         q_values = np.linspace(q_range[0], q_range[1], 10)

#         best_correlation = -float("inf")
#         best_params = None

#         for p in p_values:
#             for r in r_values:
#                 for q in q_values:
#                     try:
#                         correlation = -objective([p, r, q])
#                         if correlation > best_correlation and not np.isinf(correlation):
#                             best_correlation = correlation
#                             best_params = [p, r, q]
#                     except Exception as e:
#                         if verbose:
#                             print(
#                                 f"Error with parameters p={p:.3f}, r={r:.3f}, q={q:.3f}: {str(e)}"
#                             )
#                         continue

#         if best_params is None:
#             if verbose:
#                 print("No valid parameters found")
#             return (0.1, 0.1, 0.1, 0.0)  # 返回默认值

#     else:
#         raise ValueError("Method must be either 'differential_evolution' or 'brute'")

#     return tuple(best_params) + (best_correlation,)


# def optimize_exponential_decay(
#     data: ArrayLike,
#     target: ArrayLike,
#     decay_range: tuple[float, float] = (0.01, 1.0),
#     method: str = "differential_evolution",
#     verbose: bool = False,
# ) -> tuple[float, float]:
#     """Optimize decay_rate parameter for exponential_memory_decay function.

#     Args:
#         data: Input data array or pandas Series
#         target: Target data array or pandas Series
#         decay_range: Range for decay_rate parameter (min, max)
#         method: Optimization method ('differential_evolution' or 'brute')
#         verbose: Whether to print debugging information

#     Returns:
#         Tuple of (best_decay_rate, best_correlation)
#     """
#     from scipy import optimize, stats

#     # 数据预处理和索引对齐
#     if isinstance(data, pd.Series) and isinstance(target, pd.Series):
#         # 对齐索引
#         common_index = data.index.intersection(target.index)
#         if len(common_index) == 0:
#             raise ValueError("No common index between data and target")
#         data = data.loc[common_index]
#         target = target.loc[common_index]
#         if verbose:
#             print(f"Aligned data length: {len(common_index)}")
#     else:
#         # 如果不是Series，转换为numpy数组
#         data = np.asarray(data)
#         target = np.asarray(target)
#         if len(data) != len(target):
#             raise ValueError("Input data and target must have the same length")

#     def objective(decay_rate):
#         try:
#             # 计算加权数据
#             weighted_data = exponential_memory_decay(data, decay_rate=decay_rate)

#             # 处理NaN值
#             mask = ~(np.isnan(weighted_data) | np.isnan(target))
#             if np.sum(mask) < 3:  # 至少需要3个点来计算相关系数
#                 if verbose:
#                     print("Not enough valid points after removing NaN values")
#                 return float("inf")

#             # 计算相关系数
#             correlation = stats.pearsonr(weighted_data[mask], target[mask])[0]

#             if verbose:
#                 print(f"Decay rate: {decay_rate:.3f}, correlation: {correlation:.3f}")
#                 print(f"Valid points: {np.sum(mask)}/{len(mask)}")

#             return -correlation

#         except Exception as e:
#             if verbose:
#                 print(f"Error with decay_rate={decay_rate:.3f}: {str(e)}")
#             return float("inf")

#     if method == "differential_evolution":
#         bounds = [decay_range]
#         result = optimize.differential_evolution(
#             objective,
#             bounds=bounds,
#             maxiter=100,
#             popsize=15,
#             tol=1e-3,
#             mutation=(0.5, 1.0),
#             recombination=0.7,
#             disp=verbose,
#         )

#         if not result.success:
#             if verbose:
#                 print(f"Optimization failed: {result.message}")
#             return (0.1, 0.0)  # 返回默认值

#         best_decay_rate = result.x[0]
#         best_correlation = -result.fun

#     elif method == "brute":
#         # 创建参数网格
#         decay_rates = np.linspace(decay_range[0], decay_range[1], 50)

#         best_correlation = -float("inf")
#         best_decay_rate = None

#         for decay_rate in decay_rates:
#             try:
#                 correlation = -objective(decay_rate)
#                 if correlation > best_correlation and not np.isinf(correlation):
#                     best_correlation = correlation
#                     best_decay_rate = decay_rate
#             except Exception as e:
#                 if verbose:
#                     print(f"Error with decay_rate={decay_rate:.3f}: {str(e)}")
#                 continue

#         if best_decay_rate is None:
#             if verbose:
#                 print("No valid parameters found")
#             return (0.1, 0.0)  # 返回默认值

#     else:
#         raise ValueError("Method must be either 'differential_evolution' or 'brute'")

#     return (best_decay_rate, best_correlation)
