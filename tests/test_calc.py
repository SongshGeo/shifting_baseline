#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Test cases for calc.py utility functions."""

import numpy as np
import pytest

from past1000.utils.calc import find_top_max_indices


class TestFindTopMaxIndices1D:
    """Test find_top_max_indices function with 1D arrays."""

    @pytest.fixture
    def simple_array_1d(self):
        """Simple 1D array for basic testing."""
        return np.array([1, 5, 3, 9, 2, 8, 4, 7, 6])

    @pytest.fixture
    def array_with_duplicates_1d(self):
        """1D array with duplicate values."""
        return np.array([1, 5, 9, 9, 2, 8, 9, 7, 6])

    @pytest.fixture
    def array_with_nans_1d(self):
        """1D array containing NaN values."""
        return np.array([1, 5, np.nan, 9, 2, np.nan, 4, 7, 6])

    def test_basic_functionality_default_ratio(self, simple_array_1d):
        """
        Test basic functionality with default 10% ratio on 1D array.

        Tests that the function correctly identifies the top 10% largest values
        in a simple 1D array without NaN values.
        """
        result = find_top_max_indices(simple_array_1d)
        # Array: [1, 5, 3, 9, 2, 8, 4, 7, 6], top 10% = 1 element (9)
        # Index of value 9 is 3
        assert len(result) == 1
        assert result[0] == 3
        assert simple_array_1d[result[0]] == 9

    @pytest.mark.parametrize(
        "ratio,expected_count",
        [
            (0.1, 1),  # 10% of 9 elements = 0.9 -> max(1, int(0.9)) = 1
            (0.2, 1),  # 20% of 9 elements = 1.8 -> max(1, int(1.8)) = 1
            (0.3, 2),  # 30% of 9 elements = 2.7 -> max(1, int(2.7)) = 2
            (0.5, 4),  # 50% of 9 elements = 4.5 -> max(1, int(4.5)) = 4
            (1.0, 9),  # 100% of 9 elements = 9.0 -> max(1, int(9.0)) = 9
        ],
    )
    def test_different_ratios(self, simple_array_1d, ratio, expected_count):
        """
        Test function with different ratio parameters.

        Tests that the function correctly calculates the number of elements
        to return based on different ratio values.
        """
        result = find_top_max_indices(simple_array_1d, ratio=ratio)
        assert len(result) == expected_count

        # Verify that returned indices point to the largest values
        values = simple_array_1d[result]
        sorted_all_values = np.sort(simple_array_1d)[::-1]  # Descending order
        expected_values = sorted_all_values[:expected_count]

        # Sort both arrays for comparison (in case of ties)
        assert np.array_equal(np.sort(values)[::-1], np.sort(expected_values)[::-1])

    def test_with_duplicates(self, array_with_duplicates_1d):
        """
        Test function behavior with duplicate maximum values.

        Tests that when there are duplicate maximum values, the function
        returns valid indices for some of them.
        """
        result = find_top_max_indices(array_with_duplicates_1d, ratio=0.3)
        # Array: [1, 5, 9, 9, 2, 8, 9, 7, 6], top 30% = 2-3 elements
        values = array_with_duplicates_1d[result]

        # All returned values should be among the largest values
        sorted_values = np.sort(array_with_duplicates_1d)[::-1]

        assert len(result) >= 2
        assert all(val in sorted_values[:3] for val in values)  # Top 3 unique positions

    def test_with_nans(self, array_with_nans_1d):
        """
        Test function behavior with NaN values in 1D array.

        Tests that NaN values are properly ignored and the function only
        considers valid (non-NaN) values when finding top indices.
        """
        result = find_top_max_indices(array_with_nans_1d, ratio=0.2)
        # Array: [1, 5, nan, 9, 2, nan, 4, 7, 6], valid: [1, 5, 9, 2, 4, 7, 6]
        # 20% of 7 valid elements = 1.4 -> 1 element

        assert len(result) == 1
        # Should not return indices of NaN values (2, 5)
        assert result[0] not in [2, 5]
        # Should return index of maximum valid value (9 at index 3)
        assert result[0] == 3


class TestFindTopMaxIndicesMultiD:
    """Test find_top_max_indices function with multi-dimensional arrays."""

    @pytest.fixture
    def simple_array_2d(self):
        """Simple 2D array for testing."""
        return np.array([[1, 8, 3], [9, 2, 7], [4, 6, 5]])

    @pytest.fixture
    def array_3d(self):
        """3D array for testing."""
        return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    @pytest.fixture
    def array_2d_with_nans(self):
        """2D array containing NaN values."""
        return np.array([[1, np.nan, 3], [9, 2, 7], [np.nan, 6, 5]])

    def test_2d_array_basic(self, simple_array_2d):
        """
        Test basic functionality with 2D array.

        Tests that the function correctly returns tuple of arrays for
        row and column indices of top values in 2D array.
        """
        result = find_top_max_indices(simple_array_2d, ratio=0.2)
        # 9 elements, 20% = 1.8 -> 1 element (maximum value 9 at position [1,0])

        assert isinstance(result, tuple)
        assert len(result) == 2  # (row_indices, col_indices)
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0] == 1  # Row index
        assert result[1][0] == 0  # Column index
        assert simple_array_2d[result[0][0], result[1][0]] == 9

    def test_2d_array_multiple_indices(self, simple_array_2d):
        """
        Test 2D array returning multiple indices.

        Tests that the function correctly returns multiple coordinate pairs
        when ratio results in more than one element.
        """
        result = find_top_max_indices(simple_array_2d, ratio=0.3)
        # 9 elements, 30% = 2.7 -> 2 elements

        row_indices, col_indices = result
        assert len(row_indices) == 2
        assert len(col_indices) == 2

        # Verify the returned positions contain the largest values
        values = simple_array_2d[row_indices, col_indices]
        flat_array = simple_array_2d.flatten()
        sorted_values = np.sort(flat_array)[::-1]
        expected_values = sorted_values[:2]

        assert np.array_equal(np.sort(values)[::-1], np.sort(expected_values)[::-1])

    def test_3d_array(self, array_3d):
        """
        Test function with 3D array.

        Tests that the function correctly handles 3D arrays and returns
        a tuple with three arrays for the three dimensions.
        """
        result = find_top_max_indices(array_3d, ratio=0.25)
        # 12 elements, 25% = 3 elements

        assert isinstance(result, tuple)
        assert len(result) == 3  # (dim0, dim1, dim2)
        assert all(len(idx_array) == 3 for idx_array in result)

        # Verify the returned positions contain the largest values
        values = array_3d[result]
        expected_values = [12, 11, 10]  # Top 3 values
        assert np.array_equal(np.sort(values)[::-1], expected_values)

    def test_2d_array_with_nans(self, array_2d_with_nans):
        """
        Test 2D array with NaN values.

        Tests that NaN values are properly ignored in multi-dimensional arrays
        and only valid values are considered for finding top indices.
        """
        result = find_top_max_indices(array_2d_with_nans, ratio=0.3)
        # Valid elements: [1, 3, 9, 2, 7, 6, 5] = 7 elements
        # 30% of 7 = 2.1 -> 2 elements

        row_indices, col_indices = result
        assert len(row_indices) == 2
        assert len(col_indices) == 2

        # Verify no NaN positions are returned
        for i, j in zip(row_indices, col_indices):
            assert not np.isnan(array_2d_with_nans[i, j])

        # Verify returned values are among the largest valid values
        values = array_2d_with_nans[row_indices, col_indices]
        valid_values = array_2d_with_nans[~np.isnan(array_2d_with_nans)]
        sorted_valid = np.sort(valid_values)[::-1]
        expected_values = sorted_valid[:2]

        assert np.array_equal(np.sort(values)[::-1], np.sort(expected_values)[::-1])


class TestFindTopMaxIndicesEdgeCases:
    """Test edge cases and error conditions for find_top_max_indices function."""

    def test_empty_array(self):
        """
        Test function with empty array.

        Tests that the function handles empty arrays gracefully by
        returning appropriate empty results.
        """
        empty_1d = np.array([])
        result = find_top_max_indices(empty_1d)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

        empty_2d = np.array([]).reshape(0, 0)
        result = find_top_max_indices(empty_2d)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(len(idx_array) == 0 for idx_array in result)

    def test_all_nan_array(self):
        """
        Test function with array containing only NaN values.

        Tests that when all values are NaN, the function returns
        empty arrays as there are no valid values to process.
        """
        all_nan_1d = np.array([np.nan, np.nan, np.nan])
        result = find_top_max_indices(all_nan_1d)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

        all_nan_2d = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        result = find_top_max_indices(all_nan_2d)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(len(idx_array) == 0 for idx_array in result)

    def test_single_element_array(self):
        """
        Test function with single-element arrays.

        Tests that single-element arrays are handled correctly,
        returning the single element regardless of ratio.
        """
        single_1d = np.array([5])
        result = find_top_max_indices(single_1d, ratio=0.1)
        assert len(result) == 1
        assert result[0] == 0

        single_2d = np.array([[5]])
        result = find_top_max_indices(single_2d, ratio=0.5)
        row_indices, col_indices = result
        assert len(row_indices) == 1
        assert len(col_indices) == 1
        assert row_indices[0] == 0
        assert col_indices[0] == 0

    def test_all_equal_values(self):
        """
        Test function with array where all values are equal.

        Tests behavior when all valid values are identical - the function
        should still return the requested number of indices.
        """
        equal_values = np.array([5, 5, 5, 5, 5])
        result = find_top_max_indices(equal_values, ratio=0.4)
        # 40% of 5 elements = 2 elements
        assert len(result) == 2
        # All returned values should be 5
        assert all(equal_values[idx] == 5 for idx in result)

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 0.5, 1.0])
    def test_extreme_ratio_values(self, ratio):
        """
        Test function with extreme ratio values.

        Tests behavior with ratio values at boundaries (0, negative, 0.5, 1.0).
        The function should handle these gracefully.
        """
        test_array = np.array([1, 2, 3, 4, 5])

        if ratio <= 0:
            # For zero or negative ratios, should return at least 1 element
            result = find_top_max_indices(test_array, ratio=ratio)
            assert len(result) >= 1
        elif ratio == 0.5:
            # 50% of 5 elements = 2.5 -> 2 elements
            result = find_top_max_indices(test_array, ratio=ratio)
            assert len(result) == 2
        else:  # ratio == 1.0
            # 100% of 5 elements = 5 elements
            result = find_top_max_indices(test_array, ratio=ratio)
            assert len(result) == 5

    def test_very_small_array_with_large_ratio(self):
        """
        Test small array with large ratio.

        Tests that when ratio would result in more elements than available,
        the function returns all available elements.
        """
        small_array = np.array([1, 2])
        result = find_top_max_indices(small_array, ratio=0.8)
        # 80% of 2 elements = 1.6 -> 1 element, but function ensures at least 1
        # With such a high ratio on small array, might return both
        assert len(result) >= 1
        assert len(result) <= 2

    def test_mixed_positive_negative_values(self):
        """
        Test function with mixed positive and negative values.

        Tests that the function correctly identifies the largest values
        even when the array contains both positive and negative numbers.
        """
        mixed_array = np.array([-5, 3, -1, 7, -3, 2, 0, -2])
        result = find_top_max_indices(mixed_array, ratio=0.25)
        # 25% of 8 elements = 2 elements

        assert len(result) == 2
        values = mixed_array[result]
        # Should return indices of 7 and 3 (largest values)
        expected_values = [7, 3]
        assert np.array_equal(np.sort(values)[::-1], expected_values)

    def test_large_array_performance(self):
        """
        Test function performance with large array.

        Tests that the function can handle reasonably large arrays
        efficiently and returns correct results.
        """
        # Create a large array with known maximum values
        np.random.seed(42)  # Set seed for reproducible results
        large_array = np.random.rand(10000)
        # Set some known maximum values at specific positions
        large_array[100] = 0.999
        large_array[200] = 0.998
        large_array[300] = 0.997

        result = find_top_max_indices(large_array, ratio=0.001)
        # 0.1% of 10000 = 10 elements
        assert len(result) == 10

        # Verify that our known maximum values are included
        values = large_array[result]
        # Check that the values we set are among the top values
        assert 0.999 in values or np.any(values >= 0.999)
        assert 0.998 in values or np.any(values >= 0.998)
        assert 0.997 in values or np.any(values >= 0.997)
