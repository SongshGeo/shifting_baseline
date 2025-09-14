#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd

from shifting_baseline.constants import LEVELS, THRESHOLDS


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
