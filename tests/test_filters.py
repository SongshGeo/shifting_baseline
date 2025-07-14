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

from past1000.filters import (
    adjust_judgment_by_climate_direction,
    calc_std_deviation,
    classify,
    classify_series,
    classify_single_value,
    sigmoid_adjustment_probability,
)


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


class TestClassifySingleValue:
    """Test cases for classify_single_value function"""

    def test_basic_classification(self):
        """Test basic classification with default thresholds"""
        # Test each classification level
        assert classify_single_value(-2.0) == -2  # Below first threshold
        assert classify_single_value(-1.0) == -1  # Between first and second threshold
        assert classify_single_value(0.0) == 0  # Between second and third threshold
        assert classify_single_value(1.0) == 1  # Between third and fourth threshold
        assert classify_single_value(2.0) == 2  # Above last threshold

    def test_boundary_values(self):
        """Test classification at exact threshold boundaries"""
        # Test values exactly at thresholds
        assert classify_single_value(-1.17) == -2  # Exactly at first threshold
        assert classify_single_value(-0.33) == -1  # Exactly at second threshold
        assert classify_single_value(0.33) == 0  # Exactly at third threshold
        assert classify_single_value(1.17) == 1  # Exactly at fourth threshold

        # Test values just above thresholds
        assert classify_single_value(-1.16) == -1  # Just above first threshold
        assert classify_single_value(-0.32) == 0  # Just above second threshold
        assert classify_single_value(0.34) == 1  # Just above third threshold
        assert classify_single_value(1.18) == 2  # Just above fourth threshold

    def test_custom_thresholds_and_levels(self):
        """Test classification with custom thresholds and levels"""
        custom_thresholds = [0.0, 1.0, 2.0]
        custom_levels = [10, 20, 30, 40]

        assert classify_single_value(-1.0, custom_thresholds, custom_levels) == 10
        assert classify_single_value(0.5, custom_thresholds, custom_levels) == 20
        assert classify_single_value(1.5, custom_thresholds, custom_levels) == 30
        assert classify_single_value(3.0, custom_thresholds, custom_levels) == 40

    def test_input_validation(self):
        """Test input validation and error handling"""
        # Test NaN values
        with pytest.raises(ValueError, match="Cannot classify NaN values"):
            classify_single_value(np.nan)

        # Test infinite values
        with pytest.raises(ValueError, match="Cannot classify infinite values"):
            classify_single_value(np.inf)

        with pytest.raises(ValueError, match="Cannot classify infinite values"):
            classify_single_value(-np.inf)

        # Test non-numeric types
        with pytest.raises(TypeError, match="Value must be numeric"):
            classify_single_value("not_a_number")

        # Test mismatched thresholds and levels
        with pytest.raises(
            ValueError, match="Levels must be one element longer than thresholds"
        ):
            classify_single_value(1.0, [0.0, 1.0], [1, 2])  # levels too short

        with pytest.raises(
            ValueError, match="Levels must be one element longer than thresholds"
        ):
            classify_single_value(1.0, [0.0, 1.0], [1, 2, 3, 4])  # levels too long

    def test_threshold_order_validation(self):
        """Test validation of threshold ordering"""
        # Test unsorted thresholds
        with pytest.raises(
            ValueError, match="Thresholds must be in strictly ascending order"
        ):
            classify_single_value(1.0, [1.0, 0.0, 2.0], [1, 2, 3, 4])

        # Test duplicate thresholds
        with pytest.raises(
            ValueError, match="Thresholds must be in strictly ascending order"
        ):
            classify_single_value(1.0, [0.0, 1.0, 1.0], [1, 2, 3, 4])

    def test_numpy_numeric_types(self):
        """Test compatibility with numpy numeric types"""
        # Test various numpy types
        assert classify_single_value(np.int32(-2)) == -2
        assert classify_single_value(np.float64(1.5)) == 2
        assert classify_single_value(np.float32(0.0)) == 0


class TestClassifySeries:
    """Test cases for classify_series function"""

    def test_basic_series_classification(self):
        """Test basic series classification"""
        data = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = classify_series(data)
        expected = pd.Series([-2, -1, 0, 1, 2], dtype=int)
        pd.testing.assert_series_equal(result, expected)

    def test_numpy_array_input(self):
        """Test classification with numpy array input"""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = classify_series(data)
        expected = pd.Series([-2, -1, 0, 1, 2], dtype=int)
        pd.testing.assert_series_equal(result, expected)

    def test_index_preservation(self):
        """Test that original index is preserved"""
        data = pd.Series([-2.0, 0.0, 2.0], index=["a", "b", "c"])
        result = classify_series(data)
        expected = pd.Series([-2, 0, 2], index=["a", "b", "c"], dtype=int)
        pd.testing.assert_series_equal(result, expected)

    def test_nan_handling_strategies(self):
        """Test different strategies for handling NaN values"""
        data = pd.Series([-1.5, np.nan, 0.5, np.nan, 1.5])

        # Test 'raise' strategy (default)
        with pytest.raises(ValueError, match="Cannot classify NaN values"):
            classify_series(data, handle_na="raise")

        # Test 'skip' strategy
        result_skip = classify_series(data, handle_na="skip")
        expected_skip = pd.Series([-2, pd.NA, 1, pd.NA, 2], dtype="Int64")
        pd.testing.assert_series_equal(result_skip, expected_skip)

        # Test 'fill' strategy
        result_fill = classify_series(data, handle_na="fill")
        expected_fill = pd.Series([-2, -2, 1, -2, 2], dtype=int)
        pd.testing.assert_series_equal(result_fill, expected_fill)

    def test_empty_series_handling(self):
        """Test handling of empty series"""
        with pytest.raises(ValueError, match="Cannot classify empty series"):
            classify_series(pd.Series([]))

    def test_infinite_values_handling(self):
        """Test handling of infinite values"""
        data = pd.Series([-1.0, np.inf, 1.0])
        with pytest.raises(ValueError, match="Cannot classify infinite values"):
            classify_series(data)

        data = pd.Series([-1.0, -np.inf, 1.0])
        with pytest.raises(ValueError, match="Cannot classify infinite values"):
            classify_series(data)

    def test_non_numeric_data_handling(self):
        """Test handling of non-numeric data"""
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="Series must contain numeric data"):
            classify_series(data)

    def test_custom_parameters(self):
        """Test classification with custom thresholds and levels"""
        data = pd.Series([-1.0, 0.5, 1.5, 3.0])
        custom_thresholds = [0.0, 1.0, 2.0]
        custom_levels = [10, 20, 30, 40]

        result = classify_series(data, custom_thresholds, custom_levels)
        expected = pd.Series([10, 20, 30, 40], dtype=int)
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_handle_na_parameter(self):
        """Test validation of handle_na parameter"""
        data = pd.Series([-1.0, np.nan, 1.0])
        with pytest.raises(
            ValueError, match="handle_na must be 'raise', 'skip', or 'fill'"
        ):
            classify_series(data, handle_na="invalid")


class TestClassifyBackwardCompatibility:
    """Test backward compatibility of the original classify function"""

    def test_backward_compatibility(self):
        """Test that the original classify function still works"""
        data = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = classify(data)
        expected = pd.Series([-2, -1, 0, 1, 2], dtype=int)
        pd.testing.assert_series_equal(result, expected)

    def test_nan_handling_in_backward_compatibility(self):
        """Test that NaN values raise error in backward compatibility mode"""
        data = pd.Series([-1.0, np.nan, 1.0])
        with pytest.raises(ValueError, match="Cannot classify NaN values"):
            classify(data)


@pytest.mark.parametrize(
    "value,expected",
    [
        (-3.0, -2),
        (-1.5, -2),
        (-1.17, -2),
        (-1.16, -1),
        (-0.5, -1),
        (-0.33, -1),
        (-0.32, 0),
        (0.0, 0),
        (0.32, 0),
        (0.33, 0),
        (0.34, 1),
        (1.0, 1),
        (1.17, 1),
        (1.18, 2),
        (3.0, 2),
    ],
)
def test_classify_single_value_parametrized(value, expected):
    """Parametrized test for classify_single_value with various inputs"""
    assert classify_single_value(value) == expected


@pytest.mark.parametrize(
    "thresholds,levels,test_value,expected",
    [
        # Simple binary classification
        ([0.0], [0, 1], -1.0, 0),
        ([0.0], [0, 1], 1.0, 1),
        # Three-level classification
        ([-1.0, 1.0], [1, 2, 3], -2.0, 1),
        ([-1.0, 1.0], [1, 2, 3], 0.0, 2),
        ([-1.0, 1.0], [1, 2, 3], 2.0, 3),
        # Custom levels with non-sequential values
        ([0.0, 1.0], [10, 50, 100], 0.5, 50),
        ([0.0, 1.0], [10, 50, 100], 1.5, 100),
    ],
)
def test_custom_classification_parametrized(thresholds, levels, test_value, expected):
    """Parametrized test for custom threshold and level configurations"""
    assert classify_single_value(test_value, thresholds, levels) == expected


class TestSigmoidAdjustmentProbability:
    """Test cases for sigmoid_adjustment_probability function"""

    def test_basic_probability_calculation(self):
        """Test basic probability calculation with different climate differences"""
        # Small difference should give low probability
        prob_small = sigmoid_adjustment_probability(0.01)
        assert 0 <= prob_small <= 1
        assert prob_small < 0.5

        # Large difference should give high probability
        prob_large = sigmoid_adjustment_probability(0.5)
        assert prob_large > 0.9

        # Zero difference should give some baseline probability
        prob_zero = sigmoid_adjustment_probability(0.0)
        assert 0 <= prob_zero <= 1

    def test_sigmoid_parameters(self):
        """Test different sigmoid function parameters"""
        climate_diff = 0.1

        # Test different x0 values
        prob_low_x0 = sigmoid_adjustment_probability(climate_diff, x0=0.01)
        prob_high_x0 = sigmoid_adjustment_probability(climate_diff, x0=0.2)
        assert prob_low_x0 > prob_high_x0  # Lower x0 should give higher probability

        # Test different k values
        prob_low_k = sigmoid_adjustment_probability(climate_diff, k=1.0)
        prob_high_k = sigmoid_adjustment_probability(climate_diff, k=100.0)
        # Both should be valid probabilities
        assert 0 <= prob_low_k <= 1
        assert 0 <= prob_high_k <= 1

    def test_negative_climate_differences(self):
        """Test that function handles negative climate differences correctly"""
        # Function should use absolute value, so results should be the same
        prob_pos = sigmoid_adjustment_probability(0.2)
        prob_neg = sigmoid_adjustment_probability(-0.2)
        assert prob_pos == prob_neg

    def test_input_validation(self):
        """Test input validation"""
        # Test invalid climate_diff type
        with pytest.raises(TypeError, match="climate_diff must be numeric"):
            sigmoid_adjustment_probability("not_numeric")

        # Test invalid parameter types
        with pytest.raises(TypeError, match="Parameters x0 and k must be numeric"):
            sigmoid_adjustment_probability(0.1, x0="not_numeric")

        with pytest.raises(TypeError, match="Parameters x0 and k must be numeric"):
            sigmoid_adjustment_probability(0.1, k="not_numeric")


class TestAdjustJudgmentByClimateDirection:
    """Test cases for adjust_judgment_by_climate_direction function"""

    def test_basic_adjustment_logic(self):
        """Test basic adjustment logic based on climate direction"""
        # Climate warmed - should increase judgment
        result = adjust_judgment_by_climate_direction(1, 0.5, 0.0)
        assert result == 2

        # Climate cooled - should decrease judgment
        result = adjust_judgment_by_climate_direction(1, 0.0, 0.5)
        assert result == 0

        # No climate change - no adjustment
        result = adjust_judgment_by_climate_direction(1, 0.5, 0.5)
        assert result == 1

    def test_boundary_constraints(self):
        """Test that adjustments respect min/max level constraints"""
        # At maximum level, cannot increase further
        result = adjust_judgment_by_climate_direction(2, 1.0, 0.0)
        assert result == 2

        # At minimum level, cannot decrease further
        result = adjust_judgment_by_climate_direction(-2, 0.0, 1.0)
        assert result == -2

        # Test custom boundaries
        result = adjust_judgment_by_climate_direction(
            4, 1.0, 0.0, min_level=-5, max_level=5
        )
        assert result == 5

        result = adjust_judgment_by_climate_direction(
            -4, 0.0, 1.0, min_level=-5, max_level=5
        )
        assert result == -5

    def test_input_validation(self):
        """Test input validation"""
        # Test invalid init_judgment type
        with pytest.raises(TypeError, match="init_judgment must be an integer"):
            adjust_judgment_by_climate_direction(1.5, 0.5, 0.0)

        # Test init_judgment out of range
        with pytest.raises(ValueError, match="init_judgment must be within"):
            adjust_judgment_by_climate_direction(5, 0.5, 0.0)

        # Test invalid climate values
        with pytest.raises(TypeError, match="Climate values must be numeric"):
            adjust_judgment_by_climate_direction(1, "not_numeric", 0.0)

        with pytest.raises(TypeError, match="Climate values must be numeric"):
            adjust_judgment_by_climate_direction(1, 0.5, "not_numeric")

    def test_edge_cases(self):
        """Test edge cases and extreme values"""
        # Test with very large climate differences
        result = adjust_judgment_by_climate_direction(1, 1000.0, 0.0)
        assert result == 2

        result = adjust_judgment_by_climate_direction(1, 0.0, 1000.0)
        assert result == 0

        # Test with negative climate values
        result = adjust_judgment_by_climate_direction(1, -0.5, -1.0)
        assert result == 2  # -0.5 > -1.0, so climate warmed

        result = adjust_judgment_by_climate_direction(1, -1.0, -0.5)
        assert result == 0  # -1.0 < -0.5, so climate cooled


@pytest.mark.parametrize(
    "climate_diff,expected_range",
    [
        (0.0, (0.2, 0.3)),  # Around x0 threshold
        (0.01, (0.2, 0.3)),  # Small difference
        (0.1, (0.7, 1.0)),  # Medium difference
        (0.5, (0.99, 1.0)),  # Large difference
        (-0.1, (0.7, 1.0)),  # Negative medium difference (should be same as positive)
    ],
)
def test_sigmoid_probability_parametrized(climate_diff, expected_range):
    """Parametrized test for sigmoid_adjustment_probability"""
    prob = sigmoid_adjustment_probability(climate_diff)
    assert expected_range[0] <= prob <= expected_range[1]


@pytest.mark.parametrize(
    "init_judgment,climate_now,climate_then,expected",
    [
        # Basic warming scenarios
        (1, 1.0, 0.0, 2),
        (0, 0.5, 0.0, 1),
        (-1, 0.1, 0.0, 0),
        # Basic cooling scenarios
        (1, 0.0, 1.0, 0),
        (0, 0.0, 0.5, -1),
        (2, 0.0, 0.1, 1),
        # No change scenarios
        (1, 0.5, 0.5, 1),
        (0, 0.0, 0.0, 0),
        (-1, -0.5, -0.5, -1),
        # Boundary scenarios
        (2, 1.0, 0.0, 2),  # Cannot exceed max
        (-2, 0.0, 1.0, -2),  # Cannot go below min
    ],
)
def test_adjust_judgment_parametrized(
    init_judgment, climate_now, climate_then, expected
):
    """Parametrized test for adjust_judgment_by_climate_direction"""
    result = adjust_judgment_by_climate_direction(
        init_judgment, climate_now, climate_then
    )
    assert result == expected
