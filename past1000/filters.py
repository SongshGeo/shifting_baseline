#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd

from past1000.constants import LEVELS, THRESHOLDS


def identify_extreme_events(trace, combined, total_prob=0.1, index=None):
    """基于后验分布识别极端事件，支持任意双尾概率

    Args:
        trace (az.InferenceData): PyMC后验采样结果，包含 true_values
        combined (pd.DataFrame): 后验统计量（mean, sd, hdi_3%, hdi_97%）
        total_prob (float): 双尾概率（如0.1表示上下各5%）
        index (pd.Index): 年份索引

    Returns:
        pd.DataFrame: 极端事件（年份、事件类型、均值、阈值）
    """
    tail_prob = total_prob / 2  # 单尾概率
    posterior_samples = trace.posterior[
        "true_drought"
    ].values  # shape: (chains, draws, years)
    posterior_samples = posterior_samples.reshape(-1, len(index))  # 合并 chains 和 draws
    extreme_events = []

    for i, year in enumerate(index):
        samples = posterior_samples[:, i]
        lower_bound = np.percentile(samples, tail_prob * 100)  # 下尾分位点
        upper_bound = np.percentile(samples, (1 - tail_prob) * 100)  # 上尾分位点
        mean_val = combined.loc[year, "mean"]
        if mean_val >= upper_bound:
            extreme_events.append((year, "Extreme Wet", mean_val, upper_bound))
        elif mean_val <= lower_bound:
            extreme_events.append((year, "Extreme Dry", mean_val, lower_bound))

    return (
        pd.DataFrame(
            extreme_events,
            columns=["year", "event_type", "mean", "threshold"],
        ).set_index("year")
        if extreme_events
        else pd.DataFrame(
            columns=["year", "event_type", "mean", "threshold"]
        ).set_index("year")
    )


# 方法2: 考虑不确定性的极端值检测
def find_extremes_with_uncertainty(
    trace,
    var_name="true_values",
    lower_threshold=None,
    upper_threshold=None,
    percentile=False,
):
    """
    考虑完整后验分布的极端值检测

    Parameters:
    -----------
    trace: PyMC trace对象
    var_name: 变量名
    percentile: 极端值的百分比阈值（单尾）
    both_tails: 是否考虑双尾（True）还是只考虑单尾最大值（False）

    Returns:
    --------
    extreme_probabilities: 每个时间点是极端值的概率
    lower_probs: 每个时间点在下尾的概率
    upper_probs: 每个时间点在上尾的概率
    """
    if lower_threshold is None and upper_threshold is None:
        raise ValueError("Either lower_threshold or upper_threshold must be provided.")

    # 获取完整的后验样本
    posterior_samples = trace.posterior[
        var_name
    ].values  # shape: (chains, draws, time_points)

    # 重塑为 (total_samples, time_points)
    samples_reshaped = posterior_samples.reshape(-1, posterior_samples.shape[-1])

    # 计算全局分位数阈值
    all_samples = samples_reshaped.flatten()
    if percentile and lower_threshold is not None:
        lower_threshold = np.percentile(all_samples, lower_threshold)
    if percentile and upper_threshold is not None:
        upper_threshold = np.percentile(all_samples, upper_threshold)

    # 对每个时间点，计算极端概率
    n_timepoints = samples_reshaped.shape[1]
    lower_probs = np.zeros(n_timepoints)
    upper_probs = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        # 当前时间点的所有后验样本
        current_samples = samples_reshaped[:, t]

        # 计算该时间点样本落在下尾和上尾的概率
        if lower_threshold is not None:
            lower_probs[t] = np.mean(current_samples <= lower_threshold)
        if upper_threshold is not None:
            upper_probs[t] = np.mean(current_samples >= upper_threshold)

    if lower_threshold is not None:
        print(f"下尾阈值: {lower_threshold:.4f}")
    elif upper_threshold is not None:
        print(f"上尾阈值: {upper_threshold:.4f}")
    return lower_probs, upper_probs


# 简化的使用函数
def detect_extremes_simple(
    trace,
    var_name="true_values",
    lower_threshold=None,
    upper_threshold=None,
    percentile=False,
    confidence_threshold=0.95,
) -> dict:
    """
    简化的极端值检测函数

    Parameters:
    -----------
    extreme_percentile: 极端值定义（如10表示最小/最大的10%）
    confidence_threshold: 认可概率阈值（如0.95表示95%置信度）
    both_tails: True=双尾检测，False=只检测上尾（最大值）

    Returns:
    --------
    dict: 包含检测结果的字典
    """
    lower_probs, upper_probs = find_extremes_with_uncertainty(
        trace, var_name, lower_threshold, upper_threshold, percentile
    )
    extreme_probs = np.maximum(lower_probs, upper_probs)

    extreme_mask = extreme_probs >= confidence_threshold
    extreme_indices = np.where(extreme_mask)[0]

    results_dict = {
        "extreme_indices": extreme_indices,
        "extreme_probabilities": extreme_probs[extreme_mask]
        if len(extreme_indices) > 0
        else [],
        "all_probabilities": extreme_probs,
        "lower_probabilities": lower_probs,
        "upper_probabilities": upper_probs,
        "settings": {
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "percentile": percentile,
            "confidence_threshold": confidence_threshold,
        },
    }
    return results_dict


def split_by_trace(
    trace,
    split_at=None,
    var_name="true_values",
    confidence_threshold=0.95,
    labels=None,
    percentile=True,
) -> pd.Series:
    """
    根据后验分布分割数据
    """
    if split_at is None:
        split_at = [10, 40, 60, 90, 100]
    if labels is None:
        labels = LEVELS
    if len(labels) != len(split_at):
        raise ValueError("labels must be one more than extreme_percentiles.")
    results = np.repeat(np.nan, trace.posterior[var_name].shape[2])
    last_idx = np.array([])
    for i, threshold in enumerate(split_at):
        result = detect_extremes_simple(
            trace=trace,
            lower_threshold=threshold,
            confidence_threshold=confidence_threshold,
            var_name=var_name,
            percentile=percentile,
        )
        # 更新结果，不在 last_idx 但属于当前极端值的点
        idx = np.setdiff1d(result["extreme_indices"], last_idx)
        results[idx] = labels[i]
        last_idx = result["extreme_indices"]
    return results


def calc_std_deviation(series: pd.Series | np.ndarray) -> float:
    """Calculate the standard deviation of the last value relative to the past window years.

    Args:
        series: A pandas Series with time index

    Returns:
        float: The number of standard deviations the last value is from the mean

    Raises:
        ValueError: If series is empty, has only one value, or window is larger than series length
    """
    if len(series) <= 1:
        raise ValueError("Series must have more than one value")

    # Get the last value
    last_value = series.iloc[-1] if isinstance(series, pd.Series) else series[-1]

    # Calculate mean and std of the window
    window_mean = series.mean()
    window_std = series.std()

    # Handle case where std is 0 (constant values)
    if window_std == 0:
        return 0

    # Calculate number of standard deviations
    return (last_value - window_mean) / window_std


def classify(
    series: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    levels: list[int] | None = None,
) -> pd.Series:
    """Classify values in the series based on standard deviation thresholds.

    Args:
        series: Input series or array to classify
        thresholds: List of thresholds for classification. Default is [-1.17, -0.33, 0.33, 1.17], which is the standard deviation thresholds.
        levels: List of level values to assign. Default is [-2, -1, 0, 1, 2]
            Must be one element longer than thresholds

    Returns:
        pd.Series: Series with classification labels.
            For default values:
            -2: < -1.17 (Lowest)
            -1: -1.17 to -0.33 (Low)
            0: -0.33 to 0.33 (Normal)
            1: 0.33 to 1.17 (High)
            2: > 1.17 (Highest)

    Raises:
        ValueError: If levels length is not thresholds length + 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    if levels is None:
        levels = LEVELS
    if len(levels) != len(thresholds) + 1:
        raise ValueError("Levels must be one element longer than thresholds")

    # Classify based on thresholds
    classification = np.full_like(series, levels[0], dtype=int)
    for i, threshold in enumerate(thresholds, 1):
        classification[series > threshold] = levels[i]

    return pd.Series(
        classification, index=series.index if isinstance(series, pd.Series) else None
    )
