#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd
import scipy.stats as stats


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
        thresholds: List of thresholds for classification. Default is [-1.17, -0.33, 0.33, 1.17]
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
        thresholds = [-1.17, -0.33, 0.33, 1.17]
    if levels is None:
        levels = [-2, -1, 0, 1, 2]
    if len(levels) != len(thresholds) + 1:
        raise ValueError("Levels must be one element longer than thresholds")

    # Classify based on thresholds
    classification = np.full_like(series, levels[0], dtype=int)
    for i, threshold in enumerate(thresholds, 1):
        classification[series > threshold] = levels[i]

    return pd.Series(
        classification, index=series.index if isinstance(series, pd.Series) else None
    )


def calc_percentile(series: pd.Series | np.ndarray) -> float:
    """Calculate the percentile rank of the last value in the series.

    Args:
        series: A pandas Series or numpy array

    Returns:
        float: Percentile rank (0-100) of the last value
    """
    if len(series) <= 1:
        raise ValueError("Series must have more than one value")

    last_value = series.iloc[-1] if isinstance(series, pd.Series) else series[-1]
    return stats.percentileofscore(series, last_value, kind="strict") - 50


def calc_relative_position(series: pd.Series | np.ndarray) -> float:
    """Calculate the relative position of the last value in the series.

    Args:
        series: A pandas Series or numpy array

    Returns:
        float: Relative position (0-1) of the last value
    """
    if len(series) <= 1:
        raise ValueError("Series must have more than one value")

    last_value = series.iloc[-1] if isinstance(series, pd.Series) else series[-1]
    sorted_series = np.sort(series)
    return np.searchsorted(sorted_series, last_value) / len(series)


def calc_extreme_ratio(series: pd.Series | np.ndarray) -> float:
    """Calculate how extreme the last value is compared to historical extremes.

    Args:
        series: A pandas Series or numpy array

    Returns:
        float: Ratio indicating how extreme the last value is (-1 to 1)
    """
    if len(series) <= 1:
        raise ValueError("Series must have more than one value")

    last_value = series.iloc[-1] if isinstance(series, pd.Series) else series[-1]
    if last_value < 0:
        max_val = series.min()
    else:
        max_val = series.max()

    if max_val == 0:
        return 0

    return (last_value / (max_val - series.mean())) * 2
