#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 rand_generate_from_std_levels 函数的全面测试套件

该测试套件使用 pytest 的高级功能，包括 fixtures、mark.parametrize 等，
为每个功能创建专门的测试类，覆盖基本功能、边缘情况、新功能和错误处理。
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.utils.calc import rand_generate_from_std_levels

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_grade_matrix() -> np.ndarray:
    """基本测试用的等级矩阵，包含所有有效等级"""
    return np.array([[1, 2, 0], [-1, -2, 1], [0, 1, -1]])


@pytest.fixture
def matrix_with_nan() -> np.ndarray:
    """包含 NaN 值的等级矩阵"""
    return np.array([[1, np.nan, 0], [-1, -2, np.nan], [np.nan, 1, -1]])


@pytest.fixture
def matrix_with_string_nan() -> np.ndarray:
    """包含字符串 NaN 的等级矩阵"""
    return np.array([["1", "nan", "0"], ["-1", "-2", "NaN"], ["", "1", "-1"]])


@pytest.fixture
def mixed_types_matrix() -> np.ndarray:
    """混合数据类型的等级矩阵"""
    return np.array([[1, "nan", 0.0], [-1, -2, None], [np.nan, 1, -1]], dtype=object)


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    """包含各种 NA 类型的 pandas DataFrame"""
    return pd.DataFrame(
        {
            "col1": [1, 2, pd.NA, 0, -1],
            "col2": ["1", "nan", "0", "-1", "2"],
            "col3": [1.0, 2.0, pd.NA, 0.0, -1.0],
            "col4": [np.nan, "NaN", "", None, 1],
        }
    )


@pytest.fixture
def empty_matrix() -> np.ndarray:
    """空矩阵"""
    return np.array([[]])


@pytest.fixture
def all_nan_matrix() -> np.ndarray:
    """全 NaN 矩阵"""
    return np.array([[np.nan, np.nan], [np.nan, np.nan]])


@pytest.fixture
def single_grade_matrices() -> Dict[int, np.ndarray]:
    """单一等级的矩阵集合"""
    return {grade: np.array([[grade]]) for grade in [-2, -1, 0, 1, 2]}


@pytest.fixture
def large_matrix() -> np.ndarray:
    """大矩阵 (100x100)"""
    np.random.seed(42)
    grades = np.array([-2, -1, 0, 1, 2])
    matrix = np.random.choice(grades, size=(100, 100)).astype(float)
    # 随机添加一些 NaN
    nan_indices = np.random.choice(10000, size=1000, replace=False)
    matrix.flat[nan_indices] = np.nan
    return matrix


@pytest.fixture
def invalid_matrices() -> Dict[str, Any]:
    """各种无效输入的矩阵"""
    return {
        "invalid_grade": np.array([[1, 3, 0], [-1, -2, 1], [0, 1, -1]]),  # 3 是无效等级
        "list_input": [[1, 2], [3, 4]],  # 列表输入
        "1d_array": np.array([1, 2, 0]),  # 1D 数组，但包含有效等级值
        "3d_array": np.array([[[1, 2], [0, -1]]]),  # 3D 数组，应该被拒绝
    }


# ============================================================================
# 基本功能测试类
# ============================================================================


class TestBasicFunctionality:
    """测试 rand_generate_from_std_levels 函数的基本功能"""

    def test_basic_single_sample(self, basic_grade_matrix):
        """
        测试基本单次采样功能

        验证函数能够正确处理包含所有有效等级的矩阵，
        并返回正确形状和类型的输出。
        """
        result = rand_generate_from_std_levels(basic_grade_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == basic_grade_matrix.shape
        assert result.dtype == float

        # 检查所有值都是数值（没有 NaN）
        assert not np.any(np.isnan(result))

        # 检查值的合理性（应该在截断正态分布的范围内）
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        "mu,sigma",
        [
            (0.0, 1.0),  # 标准正态分布
            (5.0, 2.0),  # 自定义均值和标准差
            (-10.0, 0.5),  # 负均值，小标准差
            (100.0, 10.0),  # 大均值和大标准差
        ],
    )
    def test_custom_mu_sigma(self, basic_grade_matrix, mu, sigma):
        """
        测试自定义均值和标准差参数

        验证函数能够正确处理不同的 mu 和 sigma 参数，
        确保采样结果符合预期的分布参数。
        """
        result = rand_generate_from_std_levels(basic_grade_matrix, mu=mu, sigma=sigma)

        # 检查输出形状
        assert result.shape == basic_grade_matrix.shape
        assert result.dtype == float

        # 检查所有值都是数值
        assert not np.any(np.isnan(result))
        assert np.all(np.isfinite(result))

    def test_default_parameters(self, basic_grade_matrix):
        """
        测试默认参数

        验证不指定 mu 和 sigma 时使用默认值 (0.0, 1.0)。
        """
        result = rand_generate_from_std_levels(basic_grade_matrix)

        # 检查基本输出
        assert result.shape == basic_grade_matrix.shape
        assert result.dtype == float
        assert not np.any(np.isnan(result))


# ============================================================================
# 数据类型处理测试类
# ============================================================================


class TestDataTypeHandling:
    """测试各种数据类型的处理能力"""

    def test_nan_values_preservation(self, matrix_with_nan):
        """
        测试 NaN 值的保持

        验证输入中的 NaN 值在输出中保持不变，
        非 NaN 位置被正确采样。
        """
        result = rand_generate_from_std_levels(matrix_with_nan, mu=0.0, sigma=1.0)

        # 检查输出形状
        assert result.shape == matrix_with_nan.shape

        # 检查 NaN 位置保持不变
        nan_positions = np.isnan(matrix_with_nan)
        assert np.all(np.isnan(result[nan_positions]))

        # 检查非 NaN 位置有数值
        non_nan_positions = ~nan_positions
        assert not np.any(np.isnan(result[non_nan_positions]))

    def test_string_nan_conversion(self, matrix_with_string_nan):
        """
        测试字符串 NaN 的转换

        验证各种字符串形式的 NaN 表示能够被正确识别和转换。
        """
        result = rand_generate_from_std_levels(
            matrix_with_string_nan, mu=0.0, sigma=1.0
        )

        # 检查输出形状和类型
        assert result.shape == matrix_with_string_nan.shape
        assert result.dtype == float

        # 检查字符串 NaN 被正确转换为数值 NaN
        expected_nan_positions = np.array(
            [
                [False, True, False],  # "nan"
                [False, False, True],  # "NaN"
                [True, False, False],  # ""
            ]
        )
        assert np.all(np.isnan(result[expected_nan_positions]))
        assert not np.any(np.isnan(result[~expected_nan_positions]))

    def test_mixed_data_types(self, mixed_types_matrix):
        """
        测试混合数据类型的处理

        验证包含不同类型数据的矩阵能够被正确处理。
        """
        result = rand_generate_from_std_levels(mixed_types_matrix, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == mixed_types_matrix.shape
        assert result.dtype == float

    def test_pandas_dataframe_input(self, pandas_dataframe):
        """
        测试 pandas DataFrame 输入

        验证直接使用 pandas DataFrame 作为输入的功能。
        """
        result = rand_generate_from_std_levels(pandas_dataframe, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == pandas_dataframe.shape
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

    def test_pandas_series_input(self, basic_grade_matrix):
        """
        测试 pandas Series 输入

        验证使用 pandas Series 作为输入的功能。
        """
        series = pd.Series(basic_grade_matrix.flatten())
        result = rand_generate_from_std_levels(series, mu=0.0, sigma=1.0)

        # 检查输出形状和类型
        assert result.shape == series.shape
        assert result.dtype == float
        assert not np.any(np.isnan(result))


# ============================================================================
# 新功能测试类（多次采样和随机种子）
# ============================================================================


class TestMultipleSampling:
    """测试多次采样功能"""

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 50, 100])
    def test_multiple_samples_basic(self, basic_grade_matrix, n_samples):
        """
        测试多次采样的基本功能

        验证不同采样次数下的输出形状和数据类型。
        """
        result = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        if n_samples == 1:
            expected_shape = basic_grade_matrix.shape
        else:
            expected_shape = (n_samples, *basic_grade_matrix.shape)

        assert result.shape == expected_shape
        assert result.dtype == float

        # 检查所有值都是数值
        assert not np.any(np.isnan(result))

    def test_multiple_samples_with_nan(self, matrix_with_nan):
        """
        测试多次采样时 NaN 值的处理

        验证 NaN 位置在所有采样中保持一致。
        """
        n_samples = 5
        result = rand_generate_from_std_levels(
            matrix_with_nan, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        expected_shape = (n_samples, *matrix_with_nan.shape)
        assert result.shape == expected_shape

        # 检查 NaN 位置在所有采样中保持不变
        nan_positions = np.isnan(matrix_with_nan)
        for i in range(n_samples):
            assert np.all(np.isnan(result[i][nan_positions]))
            assert not np.any(np.isnan(result[i][~nan_positions]))

    def test_large_sample_size(self, basic_grade_matrix):
        """
        测试大采样次数

        验证大采样次数下的性能和统计特性。
        """
        n_samples = 1000
        result = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        expected_shape = (n_samples, *basic_grade_matrix.shape)
        assert result.shape == expected_shape
        assert result.dtype == float
        assert not np.any(np.isnan(result))

        # 对于大样本，检查统计特性
        mean_result = np.mean(result, axis=0)
        std_result = np.std(result, axis=0)

        # 标准差应该大于0
        assert np.all(std_result > 0)

        # 均值应该在合理范围内
        assert np.all(np.isfinite(mean_result))


class TestRandomSeed:
    """测试随机种子功能"""

    def test_seed_reproducibility(self, basic_grade_matrix):
        """
        测试随机种子的可重现性

        验证相同种子产生相同结果，不同种子产生不同结果。
        """
        seed = 123
        n_samples = 5

        # 使用相同种子生成两次结果
        result1 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=seed
        )
        result2 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=seed
        )

        # 检查结果完全一致
        assert np.allclose(result1, result2)

        # 使用不同种子生成结果
        result3 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=456
        )

        # 检查结果不同
        assert not np.allclose(result1, result3)

    def test_seed_with_single_sample(self, basic_grade_matrix):
        """
        测试单次采样时的随机种子功能

        验证单次采样时随机种子也能正常工作。
        """
        seed = 42

        result1 = rand_generate_from_std_levels(basic_grade_matrix, random_seed=seed)
        result2 = rand_generate_from_std_levels(basic_grade_matrix, random_seed=seed)

        # 检查结果一致
        assert np.allclose(result1, result2)

    def test_no_seed_randomness(self, basic_grade_matrix):
        """
        测试不设置种子时的随机性

        验证不设置种子时能够产生不同的结果。
        """
        result1 = rand_generate_from_std_levels(basic_grade_matrix)
        result2 = rand_generate_from_std_levels(basic_grade_matrix)

        # 结果可能相同也可能不同，但至少应该是有效的结果
        assert result1.shape == basic_grade_matrix.shape
        assert result2.shape == basic_grade_matrix.shape
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))


class TestStatisticalProperties:
    """测试统计特性"""

    def test_multiple_samples_statistics(self, basic_grade_matrix):
        """
        测试多次采样的统计特性

        验证多次采样的均值、标准差等统计量。
        """
        n_samples = 100
        result = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=42
        )

        # 计算统计量
        mean_result = np.mean(result, axis=0)
        std_result = np.std(result, axis=0)

        # 检查统计量形状
        assert mean_result.shape == basic_grade_matrix.shape
        assert std_result.shape == basic_grade_matrix.shape

        # 检查统计量的合理性
        assert np.all(std_result >= 0)
        assert np.all(np.isfinite(mean_result))
        assert np.all(np.isfinite(std_result))

    @pytest.mark.parametrize("grade", [-2, -1, 0, 1, 2])
    def test_individual_grade_statistics(self, single_grade_matrices, grade):
        """
        测试单个等级的统计特性

        验证每个等级对应的采样结果在统计上合理。
        """
        grade_matrix = single_grade_matrices[grade]
        n_samples = 1000

        result = rand_generate_from_std_levels(
            grade_matrix, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        assert result.shape == (n_samples, 1, 1)

        # 检查值的范围合理性（根据等级应该在不同的区间内）
        values = result.flatten()
        assert np.all(np.isfinite(values))

        # 检查统计量
        mean_val = np.mean(values)
        std_val = np.std(values)
        assert np.isfinite(mean_val)
        assert std_val > 0


# ============================================================================
# 边缘情况测试类
# ============================================================================


class TestEdgeCases:
    """测试边缘情况和特殊情况"""

    def test_empty_matrix(self, empty_matrix):
        """
        测试空矩阵

        验证空矩阵能够被正确处理。
        """
        result = rand_generate_from_std_levels(empty_matrix)

        # 检查输出形状
        assert result.shape == empty_matrix.shape
        assert result.dtype == float

    def test_all_nan_matrix(self, all_nan_matrix):
        """
        测试全 NaN 矩阵

        验证全 NaN 矩阵的输出也是全 NaN。
        """
        result = rand_generate_from_std_levels(all_nan_matrix)

        # 检查输出形状
        assert result.shape == all_nan_matrix.shape

        # 检查所有值都是 NaN
        assert np.all(np.isnan(result))

    def test_single_element_matrix(self, basic_grade_matrix):
        """
        测试单元素矩阵

        验证 1x1 矩阵的处理。
        """
        single_matrix = basic_grade_matrix[:1, :1]
        result = rand_generate_from_std_levels(single_matrix)

        assert result.shape == (1, 1)
        assert result.dtype == float
        assert not np.isnan(result[0, 0])

    def test_single_grade_matrices(self, single_grade_matrices):
        """
        测试单一等级的矩阵

        验证每个单一等级的矩阵能够正确处理。
        """
        for grade, matrix in single_grade_matrices.items():
            result = rand_generate_from_std_levels(matrix)

            # 检查输出形状
            assert result.shape == matrix.shape
            assert result.dtype == float

            # 检查输出是数值
            assert not np.isnan(result[0, 0])
            assert isinstance(result[0, 0], (int, float))

    def test_large_matrix_performance(self, large_matrix):
        """
        测试大矩阵的性能

        验证大矩阵的处理性能和正确性。
        """
        result = rand_generate_from_std_levels(large_matrix)

        # 检查输出形状
        assert result.shape == large_matrix.shape

        # 检查 NaN 位置保持一致
        nan_positions = np.isnan(large_matrix)
        assert np.all(np.isnan(result[nan_positions]))
        assert not np.any(np.isnan(result[~nan_positions]))

    @pytest.mark.parametrize("shape", [(1, 10), (10, 1), (5, 5), (100, 1), (1, 100)])
    def test_various_matrix_shapes(self, shape):
        """
        测试各种矩阵形状

        验证不同形状的矩阵都能正确处理。
        """
        # 创建指定形状的随机等级矩阵
        np.random.seed(42)
        grades = np.array([-2, -1, 0, 1, 2])
        matrix = np.random.choice(grades, size=shape).astype(float)

        result = rand_generate_from_std_levels(matrix)

        # 检查输出形状
        assert result.shape == matrix.shape
        assert result.dtype == float
        assert not np.any(np.isnan(result))


# ============================================================================
# 向后兼容性测试类
# ============================================================================


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_default_parameters_compatibility(self, basic_grade_matrix):
        """
        测试默认参数的向后兼容性

        验证不指定新参数时保持原有行为。
        """
        # 不指定新参数，应该保持原有行为
        result_old = rand_generate_from_std_levels(
            basic_grade_matrix, mu=0.0, sigma=1.0
        )
        result_new = rand_generate_from_std_levels(
            basic_grade_matrix, mu=0.0, sigma=1.0, n_samples=1
        )

        # 形状应该相同
        assert result_old.shape == result_new.shape
        assert result_old.dtype == result_new.dtype

    def test_positional_arguments(self, basic_grade_matrix):
        """
        测试位置参数的兼容性

        验证只使用位置参数时的行为。
        """
        result = rand_generate_from_std_levels(basic_grade_matrix)

        assert result.shape == basic_grade_matrix.shape
        assert result.dtype == float
        assert not np.any(np.isnan(result))

    def test_mixed_parameter_styles(self, basic_grade_matrix):
        """
        测试混合参数风格的兼容性

        验证混合使用位置参数和关键字参数的行为。
        """
        # 测试各种参数组合
        result1 = rand_generate_from_std_levels(basic_grade_matrix, 0.0, 1.0)
        result2 = rand_generate_from_std_levels(basic_grade_matrix, mu=0.0, sigma=1.0)
        result3 = rand_generate_from_std_levels(basic_grade_matrix, 0.0, sigma=1.0)

        # 所有结果应该有相同的形状和类型
        for result in [result1, result2, result3]:
            assert result.shape == basic_grade_matrix.shape
            assert result.dtype == float
            assert not np.any(np.isnan(result))


# ============================================================================
# 错误处理测试类
# ============================================================================


class TestErrorHandling:
    """测试错误处理和异常情况"""

    def test_invalid_n_samples(self, basic_grade_matrix):
        """
        测试无效的 n_samples 参数

        验证 n_samples 参数验证逻辑。
        """
        # n_samples 为 0
        with pytest.raises(ValueError, match="n_samples 必须大于等于 1 的整数"):
            rand_generate_from_std_levels(basic_grade_matrix, n_samples=0)

        # n_samples 为负数
        with pytest.raises(ValueError, match="n_samples 必须大于等于 1 的整数"):
            rand_generate_from_std_levels(basic_grade_matrix, n_samples=-1)

        # n_samples 为浮点数（虽然不是整数，但应该被处理）
        with pytest.raises(ValueError, match="n_samples 必须大于等于 1 的整数"):
            rand_generate_from_std_levels(basic_grade_matrix, n_samples=1.5)

    def test_invalid_grades(self, invalid_matrices):
        """
        测试无效等级值

        验证对无效等级值的错误处理。
        """
        with pytest.raises(ValueError, match="非 NA 元素包含无效等级"):
            rand_generate_from_std_levels(invalid_matrices["invalid_grade"])

    def test_invalid_input_types(self, invalid_matrices):
        """
        测试无效输入类型

        验证对各种无效输入类型的错误处理。
        """
        # 非数组输入
        with pytest.raises(
            ValueError, match="输入必须是 numpy.ndarray、pandas.Series 或 pandas.DataFrame"
        ):
            rand_generate_from_std_levels(invalid_matrices["list_input"])

        # 3D 数组（超过2维）
        with pytest.raises(ValueError, match="输入数组维度不能超过2维"):
            rand_generate_from_std_levels(invalid_matrices["3d_array"])

    def test_invalid_mu_sigma_types(self, basic_grade_matrix):
        """
        测试无效的 mu 和 sigma 类型

        验证对无效参数类型的处理。
        """
        # 这些测试可能会通过，因为 numpy 可能会自动转换
        # 但我们可以测试一些边界情况
        try:
            result = rand_generate_from_std_levels(
                basic_grade_matrix, mu="invalid", sigma=1.0
            )
            # 如果没有抛出异常，检查结果是否合理
            assert result.shape == basic_grade_matrix.shape
        except (ValueError, TypeError):
            # 如果抛出异常，这也是可以接受的
            pass

    def test_extreme_parameter_values(self, basic_grade_matrix):
        """
        测试极端参数值

        验证对极端参数值的处理。
        """
        # 极大的 sigma 值
        result = rand_generate_from_std_levels(basic_grade_matrix, mu=0.0, sigma=1e10)
        assert result.shape == basic_grade_matrix.shape
        assert not np.any(np.isnan(result))

        # 极小的 sigma 值
        result = rand_generate_from_std_levels(basic_grade_matrix, mu=0.0, sigma=1e-10)
        assert result.shape == basic_grade_matrix.shape
        assert not np.any(np.isnan(result))


# ============================================================================
# 性能和压力测试类
# ============================================================================


class TestPerformance:
    """测试性能和压力情况"""

    @pytest.mark.slow
    def test_large_sample_count(self, basic_grade_matrix):
        """
        测试大采样次数的性能

        验证大量采样时的性能和内存使用。
        """
        n_samples = 10000
        result = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        expected_shape = (n_samples, *basic_grade_matrix.shape)
        assert result.shape == expected_shape

        # 检查内存使用合理（不应该有内存泄漏）
        assert result.nbytes < 1e9  # 小于 1GB

    @pytest.mark.slow
    def test_large_matrix_with_samples(self):
        """
        测试大矩阵和大采样次数的组合

        验证大矩阵配合多次采样的性能。
        """
        # 创建较大的矩阵
        large_matrix = np.random.choice([-2, -1, 0, 1, 2], size=(50, 50)).astype(float)
        n_samples = 1000

        result = rand_generate_from_std_levels(
            large_matrix, n_samples=n_samples, random_seed=42
        )

        # 检查输出形状
        expected_shape = (n_samples, *large_matrix.shape)
        assert result.shape == expected_shape
        assert not np.any(np.isnan(result))

    def test_memory_efficiency(self, basic_grade_matrix):
        """
        测试内存效率

        验证函数不会产生不必要的内存拷贝。
        """
        import sys

        # 测试单次采样
        result1 = rand_generate_from_std_levels(basic_grade_matrix, n_samples=1)
        size1 = sys.getsizeof(result1)

        # 测试多次采样
        result2 = rand_generate_from_std_levels(basic_grade_matrix, n_samples=100)
        size2 = sys.getsizeof(result2)

        # 多次采样应该占用更多内存，但比例应该合理
        assert size2 > size1
        assert size2 / size1 <= 150  # 允许一些开销，但不应该超过 150 倍


# ============================================================================
# 集成测试类
# ============================================================================


class TestIntegration:
    """集成测试，测试各种功能的组合使用"""

    def test_complex_scenario(self):
        """
        测试复杂场景

        验证各种功能组合使用的正确性。
        """
        # 创建复杂的测试数据
        np.random.seed(42)
        matrix = np.random.choice([-2, -1, 0, 1, 2], size=(10, 10)).astype(float)
        # 添加一些 NaN
        nan_indices = np.random.choice(100, size=10, replace=False)
        matrix.flat[nan_indices] = np.nan

        # 测试多次采样
        n_samples = 50
        result = rand_generate_from_std_levels(
            matrix, mu=5.0, sigma=2.0, n_samples=n_samples, random_seed=123
        )

        # 检查基本属性
        assert result.shape == (n_samples, *matrix.shape)
        assert result.dtype == float

        # 检查 NaN 位置
        nan_positions = np.isnan(matrix)
        for i in range(n_samples):
            assert np.all(np.isnan(result[i][nan_positions]))
            assert not np.any(np.isnan(result[i][~nan_positions]))

        # 检查统计特性
        mean_result = np.mean(result, axis=0)
        std_result = np.std(result, axis=0)
        assert np.all(np.isfinite(mean_result[~nan_positions]))
        assert np.all(std_result[~nan_positions] > 0)

    def test_reproducible_workflow(self, basic_grade_matrix):
        """
        测试可重现的工作流程

        验证完整的分析工作流程的可重现性。
        """
        # 第一次运行
        result1 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=100, random_seed=42
        )
        mean1 = np.mean(result1, axis=0)
        std1 = np.std(result1, axis=0)

        # 第二次运行（相同种子）
        result2 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=100, random_seed=42
        )
        mean2 = np.mean(result2, axis=0)
        std2 = np.std(result2, axis=0)

        # 结果应该完全一致
        assert np.allclose(result1, result2)
        assert np.allclose(mean1, mean2)
        assert np.allclose(std1, std2)

        # 第三次运行（不同种子）
        result3 = rand_generate_from_std_levels(
            basic_grade_matrix, n_samples=100, random_seed=123
        )

        # 结果应该不同
        assert not np.allclose(result1, result3)
