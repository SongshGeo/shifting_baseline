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


def classify_single_value(
    value: float,
    thresholds: list[float] | None = None,
    levels: list[int] | None = None,
) -> int:
    r"""Classify a single value based on standard deviation thresholds.

    This function performs threshold-based classification of a single numeric value using
    a series of ordered thresholds. The classification logic follows a step-wise comparison
    where the value is assigned to progressively higher levels as it exceeds each threshold.
    The mathematical relationship can be expressed as: for thresholds $t_1 < t_2 < ... < t_n$
    and levels $l_0, l_1, ..., l_n$, a value $x$ is classified as level $l_i$ if
    $t_{i-1} < x \\leq t_i$ (where $t_0 = -\\infty$ and $t_{n+1} = +\\infty$).

    The default thresholds correspond to standardized drought severity classifications:
    - Level -2 ($$x \leq -1.17$$): Severe drought conditions
    - Level -1 ($$-1.17 < x \leq -0.33$$): Moderate drought conditions
    - Level 0 ($$-0.33 < x \leq 0.33$$): Normal conditions
    - Level 1 ($$0.33 < x \leq 1.17$$): Moderate wet conditions
    - Level 2 ($$x > 1.17$$): Severe wet conditions

    Args:
        value: The numeric value to classify. Must be a finite number.
        thresholds: List of threshold values for classification boundaries. Must be
            in strictly ascending order. Default is [-1.17, -0.33, 0.33, 1.17],
            which corresponds to standardized drought index thresholds.
        levels: List of classification levels to assign. Must contain exactly
            len(thresholds) + 1 elements. Default is [-2, -1, 0, 1, 2].

    Returns:
        int: The classification level for the input value. Returns the lowest
            level for values below the first threshold, and the highest level
            for values above the last threshold.

    Raises:
        ValueError: If levels length is not thresholds length + 1, if thresholds
            are not in strictly ascending order, or if input value is NaN or infinite.
        TypeError: If value is not a numeric type.

    Examples:
        >>> classify_single_value(-1.5)
        -2
        >>> classify_single_value(0.0)
        0
        >>> classify_single_value(1.5)
        2
        >>> classify_single_value(0.5, thresholds=[0.0, 1.0], levels=[0, 1, 2])
        1
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    if levels is None:
        levels = LEVELS

    # Input validation
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"Value must be numeric, got {type(value)}")

    if np.isnan(value):
        raise ValueError("Cannot classify NaN values")

    if np.isinf(value):
        raise ValueError("Cannot classify infinite values")

    if len(levels) != len(thresholds) + 1:
        raise ValueError(
            f"Levels must be one element longer than thresholds. "
            f"Got {len(levels)} levels and {len(thresholds)} thresholds"
        )

    # Check if thresholds are in ascending order
    if not all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
        raise ValueError("Thresholds must be in strictly ascending order")

    # Classify based on thresholds
    classification = levels[0]  # Start with the lowest level
    for i, threshold in enumerate(thresholds, 1):
        if value > threshold:
            classification = levels[i]
        else:
            break

    return classification


def classify_series(
    series: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    levels: list[int] | None = None,
    handle_na: str = "raise",
) -> pd.Series:
    """Classify values in a series based on standard deviation thresholds.

    This function applies threshold-based classification to each element in a pandas Series
    or numpy array. The classification is performed element-wise using the same logic as
    classify_single_value, providing consistent behavior across individual values and
    batch operations. The function maintains the original index structure for pandas Series
    and creates a new index for numpy arrays.

    The classification process handles various data quality issues including missing values,
    infinite values, and mixed data types. The mathematical foundation remains the same
    as the single-value classification, with additional considerations for vectorized
    operations and error handling strategies.

    Args:
        series: Input data to classify. Can be either a pandas Series or numpy array
            containing numeric values.
        thresholds: List of threshold values for classification boundaries. Must be
            in strictly ascending order. Default is [-1.17, -0.33, 0.33, 1.17].
        levels: List of classification levels to assign. Must contain exactly
            len(thresholds) + 1 elements. Default is [-2, -1, 0, 1, 2].
        handle_na: Strategy for handling NaN values. Options are:
            - "raise": Raise ValueError when NaN values are encountered
            - "skip": Skip NaN values and return NaN in the result
            - "fill": Fill NaN values with the lowest classification level

    Returns:
        pd.Series: Series with classification labels and preserved index structure.
            The dtype will be 'Int64' to handle potential NaN values properly.

    Raises:
        ValueError: If levels length is not thresholds length + 1, if thresholds
            are not in strictly ascending order, if series is empty, or if NaN values
            are encountered with handle_na="raise".
        TypeError: If series contains non-numeric data types.

    Examples:
        >>> data = pd.Series([-1.5, -0.5, 0.0, 0.5, 1.5])
        >>> classify_series(data)
        0   -2
        1   -1
        2    0
        3    1
        4    2
        dtype: Int64

        >>> data_with_na = pd.Series([-1.5, np.nan, 0.5])
        >>> classify_series(data_with_na, handle_na="skip")
        0   -2
        1   <NA>
        2    1
        dtype: Int64
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    if levels is None:
        levels = LEVELS

    # Input validation
    if len(series) == 0:
        raise ValueError("Cannot classify empty series")

    if len(levels) != len(thresholds) + 1:
        raise ValueError(
            f"Levels must be one element longer than thresholds. "
            f"Got {len(levels)} levels and {len(thresholds)} thresholds"
        )

    # Check if thresholds are in ascending order
    if not all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
        raise ValueError("Thresholds must be in strictly ascending order")

    # Convert to pandas Series if numpy array
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    # Handle non-numeric data
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Series must contain numeric data")

    # Handle NaN values
    nan_mask = series.isna()
    if nan_mask.any():
        if handle_na == "raise":
            raise ValueError(
                "Cannot classify NaN values. Use handle_na='skip' or 'fill'"
            )
        elif handle_na == "skip":
            # Continue with NaN values, they will be preserved
            pass
        elif handle_na == "fill":
            series = series.fillna(levels[0])  # Fill with lowest level
        else:
            raise ValueError("handle_na must be 'raise', 'skip', or 'fill'")

    # Handle infinite values
    inf_mask = np.isinf(series)
    if inf_mask.any():
        raise ValueError("Cannot classify infinite values")

    # Classify based on thresholds using vectorized operations
    classification = np.full(len(series), levels[0], dtype=int)

    # Apply thresholds in order
    for i, threshold in enumerate(thresholds, 1):
        classification[series > threshold] = levels[i]

    # Handle NaN values in the result
    if nan_mask.any() and handle_na == "skip":
        classification = classification.astype(object)
        classification[nan_mask] = pd.NA

    return pd.Series(
        classification,
        index=series.index,
        dtype="Int64" if (nan_mask.any() and handle_na == "skip") else int,
    )


# Backward compatibility alias
def classify(
    series: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    levels: list[int] | None = None,
    handle_na: str = "raise",
) -> pd.Series:
    """Classify values in the series based on standard deviation thresholds.

    This function is maintained for backward compatibility. For new code, consider
    using classify_series() for better parameter control and error handling options.

    Args:
        series: Input series or array to classify
        thresholds: List of thresholds for classification. Default is [-1.17, -0.33, 0.33, 1.17]
        levels: List of level values to assign. Default is [-2, -1, 0, 1, 2]

    Returns:
        pd.Series: Series with classification labels.

    Raises:
        ValueError: If levels length is not thresholds length + 1
    """
    return classify_series(series, thresholds, levels, handle_na)


def sigmoid_adjustment_probability(
    climate_diff: float,
    x0: float = 0.05,
    k: float = 27.72,
) -> float:
    """Calculate adjustment probability using sigmoid function based on climate difference.

    This function implements a sigmoid probability model that captures the psychological
    tendency to adjust expectations when climate conditions change. The probability
    increases with larger climate differences, following a smooth S-shaped curve.

    The probability is calculated using the sigmoid function:
    $$P_{adjust} = \\frac{1}{1 + e^{-k(|\\Delta climate| - x_0)}}$$

    Args:
        climate_diff: Climate difference value (climate_now - climate_then).
        x0: Offset threshold for sigmoid calculation. Climate differences below
            this threshold have low adjustment probability. Default is 0.05.
        k: Steepness parameter for sigmoid function. Higher values create
            sharper probability transitions. Default is 27.72.

    Returns:
        float: Adjustment probability between 0 and 1.

    Raises:
        TypeError: If any parameter is not numeric.

    Examples:
        >>> # Small climate difference
        >>> sigmoid_adjustment_probability(0.01)
        0.2481...

        >>> # Large climate difference
        >>> sigmoid_adjustment_probability(0.5)
        0.9999...

        >>> # No climate difference
        >>> sigmoid_adjustment_probability(0.0)
        0.2689...
    """
    if not isinstance(climate_diff, (int, float)):
        raise TypeError("climate_diff must be numeric")

    if not all(isinstance(x, (int, float)) for x in [x0, k]):
        raise TypeError("Parameters x0 and k must be numeric")

    abs_diff = abs(climate_diff)
    return 1 / (1 + np.exp(-k * (abs_diff - x0)))


def adjust_judgment_by_climate_direction(
    init_judgment: int,
    climate_now: float,
    climate_then: float,
    min_level: int = -2,
    max_level: int = 2,
) -> int:
    """Adjust judgment level based on climate change direction.

    This function implements deterministic judgment adjustment logic based on the
    direction of climate change. It assumes that the decision to adjust has already
    been made (e.g., through probability calculation and random sampling) and only
    determines the direction and magnitude of adjustment.

    The adjustment logic follows climate change direction:
    - If climate warmed/became wetter (climate_now > climate_then): increase level
    - If climate cooled/became drier (climate_now < climate_then): decrease level
    - If no climate change (climate_now == climate_then): no adjustment
    - All adjustments are constrained within [min_level, max_level] bounds

    Args:
        init_judgment: Initial judgment level to adjust.
            Must be within [min_level, max_level] range.
        climate_now: Current climate value.
        climate_then: Past climate value for comparison.
        min_level: Minimum allowable judgment level. Default is -2.
        max_level: Maximum allowable judgment level. Default is 2.

    Returns:
        int: Adjusted judgment level, constrained within [min_level, max_level].

    Raises:
        ValueError: If init_judgment is outside [min_level, max_level] range.
        TypeError: If any parameter is not of the expected type.

    Examples:
        >>> # Climate warmed - increase judgment
        >>> adjust_judgment_by_climate_direction(1, 0.5, 0.0)
        2

        >>> # Climate cooled - decrease judgment
        >>> adjust_judgment_by_climate_direction(1, 0.0, 0.5)
        0

        >>> # No climate change - no adjustment
        >>> adjust_judgment_by_climate_direction(1, 0.5, 0.5)
        1

        >>> # At boundary - constrained adjustment
        >>> adjust_judgment_by_climate_direction(2, 1.0, 0.0)  # Cannot exceed max
        2

        >>> adjust_judgment_by_climate_direction(-2, 0.0, 1.0)  # Cannot go below min
        -2
    """
    # Input validation
    if not isinstance(init_judgment, int):
        raise TypeError("init_judgment must be an integer")

    if init_judgment < min_level or init_judgment > max_level:
        raise ValueError(
            f"init_judgment must be within [{min_level}, {max_level}] range, "
            f"got {init_judgment}"
        )

    if not all(isinstance(x, (int, float)) for x in [climate_now, climate_then]):
        raise TypeError("Climate values must be numeric")

    # Calculate climate difference
    climate_diff = climate_now - climate_then

    # Determine adjustment direction based on climate change
    if climate_diff > 0:
        # Climate became warmer/wetter - increase judgment level
        return min(init_judgment + 1, max_level)
    elif climate_diff < 0:
        # Climate became cooler/drier - decrease judgment level
        return max(init_judgment - 1, min_level)
    else:
        # No climate change - no adjustment
        return init_judgment
