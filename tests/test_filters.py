#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from past1000.filters import calc_std_deviation, classify


class TestStdDeviation:
    """Test cases for calculating standard deviation of time series data."""

    def test_normal_case(self, series):
        """Test normal case with a series of 1000 years data."""
        result = calc_std_deviation(series)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_edge_cases(self):
        """Test edge cases with different series lengths."""
        # Test with empty series
        empty_series = pd.Series([], index=[])
        with pytest.raises(ValueError):
            calc_std_deviation(empty_series)

        # Test with single value
        single_series = pd.Series([1], index=[1000])
        with pytest.raises(ValueError):
            calc_std_deviation(single_series)

    def test_numpy_array(self):
        """Test with numpy array input."""
        arr = np.array([1, 2, 3, 4, 5])
        result = calc_std_deviation(arr)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_constant_series(self):
        """Test with a series of constant values."""
        constant_series = pd.Series([1] * 1000, index=np.arange(850, 1850))
        result = calc_std_deviation(constant_series)
        assert result == 0  # Standard deviation of constant values should be 0

    def test_extreme_values(self):
        """Test with series containing extreme values."""
        extreme_series = pd.Series([-100, 100] * 500, index=np.arange(850, 1850))
        result = calc_std_deviation(extreme_series)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)


class TestClassification:
    """Test cases for classification based on standard deviation thresholds."""

    @pytest.mark.parametrize(
        "data, results",
        [
            ([-2, -1, -0.5, 0, 0.2, 1, 2], [-2, -1, -1, 0, 0, 1, 2]),
        ],
    )
    def test_normal_classification(self, data, results):
        """Test classification with normal distribution data."""
        # action
        result = classify(pd.Series(data))
        # assert
        assert_series_equal(result, pd.Series(results))

    def test_invalid_levels(self):
        """Test that invalid level lengths raise ValueError."""
        data = [-2, -1, 0, 1, 2]
        series = pd.Series(data)
        thresholds = [-1.5, -0.5, 0.5, 1.5]
        invalid_levels = [1, 2, 3]  # Too short

        with pytest.raises(ValueError):
            classify(series, thresholds=thresholds, levels=invalid_levels)

    def test_numpy_array(self):
        """Test classification with numpy array input."""
        arr = np.array([-2, -1, -0.5, 0, 0.2, 1, 2])
        result = classify(arr)
        assert_series_equal(result, pd.Series([-2, -1, -1, 0, 0, 1, 2]))
