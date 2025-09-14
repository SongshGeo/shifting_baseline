#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 generate_from_2d_levels 函数
"""

import numpy as np
import pytest

from shifting_baseline.utils.calc import rand_generate_from_std_levels


class TestGenerateFrom2DLevels:
    """测试 generate_from_2d_levels 函数的各种情况"""

    def test_basic_functionality(self):
        """测试基本功能"""
        # 创建测试数据
        grade_matrix = np.array([[1, 2, 0], [-1, -2, 1], [0, 1, -1]])

        result = rand_generate_from_std_levels(grade_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状
        assert result.shape == grade_matrix.shape
        assert result.dtype == float

        # 检查所有值都是数值（没有 NaN，除了输入中的 NA）
        assert not np.any(np.isnan(result))

    def test_with_nan_values(self):
        """测试包含 NaN 值的情况"""
        grade_matrix = np.array([[1, np.nan, 0], [-1, -2, np.nan], [np.nan, 1, -1]])

        result = rand_generate_from_std_levels(grade_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查 NaN 位置保持不变
        nan_positions = np.isnan(grade_matrix)
        assert np.all(np.isnan(result[nan_positions]))

        # 检查非 NaN 位置有数值
        non_nan_positions = ~nan_positions
        assert not np.any(np.isnan(result[non_nan_positions]))

    def test_string_nan_values(self):
        """测试字符串类型的 NaN 值"""
        # 创建包含字符串 NaN 的数组
        grade_matrix = np.array(
            [["1", "nan", "0"], ["-1", "-2", "NaN"], ["", "1", "-1"]]
        )

        result = rand_generate_from_std_levels(grade_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状
        assert result.shape == grade_matrix.shape
        assert result.dtype == float

        # 检查字符串 NaN 被正确转换为数值 NaN
        expected_nan_positions = np.array(
            [[False, True, False], [False, False, True], [True, False, False]]
        )
        assert np.all(np.isnan(result[expected_nan_positions]))
        assert not np.any(np.isnan(result[~expected_nan_positions]))

    def test_mixed_data_types(self):
        """测试混合数据类型"""
        grade_matrix = np.array(
            [[1, "nan", 0.0], [-1, -2, None], [np.nan, 1, -1]], dtype=object
        )

        result = rand_generate_from_std_levels(grade_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == grade_matrix.shape
        assert result.dtype == float

    def test_pandas_dataframe_input(self):
        """测试直接使用 pandas DataFrame 输入"""
        import pandas as pd

        # 创建包含各种 NA 类型的 DataFrame
        df = pd.DataFrame(
            {
                "col1": [1, 2, pd.NA, 0, -1],
                "col2": ["1", "nan", "0", "-1", "2"],
                "col3": [1.0, 2.0, pd.NA, 0.0, -1.0],
                "col4": [np.nan, "NaN", "", None, 1],
            }
        )

        # 直接使用 DataFrame 输入
        result = rand_generate_from_std_levels(df, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == df.shape
        assert result.dtype == float

        # 检查 NA 位置被正确转换为 NaN
        expected_nan_positions = np.array(
            [
                [False, False, False, True],  # col4 的 np.nan
                [False, True, False, True],  # col2 的 'nan', col4 的 'NaN'
                [True, False, True, True],  # col1 的 pd.NA, col3 的 pd.NA, col4 的 ''
                [False, False, False, True],  # col4 的 None
                [False, False, False, False],  # 无 NA
            ]
        )
        assert np.all(np.isnan(result[expected_nan_positions]))
        assert not np.any(np.isnan(result[~expected_nan_positions]))

    def test_pandas_na_values_backward_compatibility(self):
        """测试使用 .values 属性的向后兼容性"""
        import pandas as pd

        # 创建包含 pandas NA 的 DataFrame
        df = pd.DataFrame(
            {
                "col1": [1, 2, pd.NA, 0, -1],
                "col2": ["1", "nan", "0", "-1", "2"],
                "col3": [1.0, 2.0, pd.NA, 0.0, -1.0],
            }
        )

        values = df.values
        result = rand_generate_from_std_levels(values, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == values.shape
        assert result.dtype == float

        # 检查 pandas NA 位置被正确转换为 NaN
        expected_nan_positions = np.array(
            [
                [False, False, False],
                [False, True, False],
                [True, False, True],
                [False, False, False],
                [False, False, False],
            ]
        )
        assert np.all(np.isnan(result[expected_nan_positions]))
        assert not np.any(np.isnan(result[~expected_nan_positions]))

    def test_complex_mixed_types(self):
        """测试复杂的混合类型"""
        import pandas as pd

        # 创建包含所有可能类型的 DataFrame
        df = pd.DataFrame(
            {
                "col1": [1, 2, pd.NA, 0, -1],
                "col2": ["1", "nan", "0", "-1", "2"],
                "col3": [1.0, 2.0, pd.NA, 0.0, -1.0],
                "col4": [np.nan, "NaN", "", None, 1],
            }
        )

        values = df.values
        result = rand_generate_from_std_levels(values, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == values.shape
        assert result.dtype == float

    def test_invalid_grades(self):
        """测试无效等级值"""
        grade_matrix = np.array([[1, 3, 0], [-1, -2, 1], [0, 1, -1]])  # 3 是无效等级

        with pytest.raises(ValueError, match="非 NA 元素包含无效等级"):
            rand_generate_from_std_levels(grade_matrix)

    def test_invalid_input_type(self):
        """测试无效输入类型"""
        # 非数组输入
        with pytest.raises(
            ValueError, match="输入必须是 numpy.ndarray、pandas.Series 或 pandas.DataFrame"
        ):
            rand_generate_from_std_levels([[1, 2], [3, 4]])

        # 1D 数组
        with pytest.raises(ValueError, match="输入必须是 2D 数组"):
            rand_generate_from_std_levels(np.array([1, 2, 3]))

    def test_custom_mu_sigma(self):
        """测试自定义均值和标准差"""
        grade_matrix = np.array([[1, 0, -1], [2, -2, 1], [0, 1, -1]])

        mu, sigma = 5.0, 2.0
        result = rand_generate_from_std_levels(grade_matrix, mu=mu, sigma=sigma)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查所有值都是数值
        assert not np.any(np.isnan(result))

    def test_empty_matrix(self):
        """测试空矩阵"""
        grade_matrix = np.array([[]])

        result = rand_generate_from_std_levels(grade_matrix)

        # 检查输出形状
        assert result.shape == grade_matrix.shape
        assert result.dtype == float

    def test_all_nan_matrix(self):
        """测试全 NaN 矩阵"""
        grade_matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        result = rand_generate_from_std_levels(grade_matrix)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查所有值都是 NaN
        assert np.all(np.isnan(result))

    def test_single_grade(self):
        """测试单一等级"""
        grade_matrix = np.array([[1, 1, 1], [1, 1, 1]])

        result = rand_generate_from_std_levels(grade_matrix)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查所有值都是数值
        assert not np.any(np.isnan(result))

    @pytest.mark.parametrize("grade", [-2, -1, 0, 1, 2])
    def test_individual_grades(self, grade):
        """测试每个有效等级"""
        grade_matrix = np.array([[grade]])

        result = rand_generate_from_std_levels(grade_matrix)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查输出是数值
        assert not np.isnan(result[0, 0])
        assert isinstance(result[0, 0], (int, float))

    def test_large_matrix(self):
        """测试大矩阵"""
        # 创建 100x100 的随机等级矩阵
        np.random.seed(42)
        grades = np.array([-2, -1, 0, 1, 2])
        grade_matrix = np.random.choice(grades, size=(100, 100)).astype(float)

        # 随机添加一些 NaN
        nan_indices = np.random.choice(10000, size=1000, replace=False)
        grade_matrix.flat[nan_indices] = np.nan

        result = rand_generate_from_std_levels(grade_matrix)

        # 检查输出形状
        assert result.shape == grade_matrix.shape

        # 检查 NaN 位置保持一致
        nan_positions = np.isnan(grade_matrix)
        assert np.all(np.isnan(result[nan_positions]))
        assert not np.any(np.isnan(result[~nan_positions]))
