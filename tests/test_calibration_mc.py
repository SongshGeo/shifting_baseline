#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""calibration.MismatchReport 本轮改动的测试：

- 蒙特卡洛显著性检验改用 default_rng + random_seed → 可复现；
- get_mean_diff 改为所有非 NaN 单元的整体均值（每单元等权），而非列均值的均值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.constants import LEVELS
from shifting_baseline.filters import classify


@pytest.fixture
def report_with_errors() -> MismatchReport:
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.RangeIndex(1000, 1000 + n)
    val = pd.Series(rng.normal(0, 1, n), index=idx)
    true = classify(val)
    pred = true.copy()
    err = rng.choice(n, 40, replace=False)
    pred.iloc[err] = rng.choice([-2, -1, 0, 1, 2], 40)
    return MismatchReport(pred=pred, true=true, value_series=val)


class TestMonteCarloReproducibility:
    def test_same_seed_same_pvalues(self, report_with_errors):
        r1 = report_with_errors
        r1.analyze_error_patterns(mc_runs=30, random_seed=42)
        # 同样的数据 + 同样的 seed → 同样的 p 值矩阵
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.RangeIndex(1000, 1000 + n)
        val = pd.Series(rng.normal(0, 1, n), index=idx)
        true = classify(val)
        pred = true.copy()
        err = rng.choice(n, 40, replace=False)
        pred.iloc[err] = rng.choice([-2, -1, 0, 1, 2], 40)
        r2 = MismatchReport(pred=pred, true=true, value_series=val)
        r2.analyze_error_patterns(mc_runs=30, random_seed=42)
        np.testing.assert_allclose(
            r1.p_value_matrix.values, r2.p_value_matrix.values, equal_nan=True
        )

    def test_no_global_state_pollution(self, report_with_errors):
        # 跑 MC 不应改变全局 np.random 状态
        np.random.seed(123)
        before = np.random.random()
        np.random.seed(123)
        report_with_errors.analyze_error_patterns(mc_runs=20, random_seed=7)
        after = np.random.random()
        assert before == after  # 全局状态未被 MC 干扰

    def test_pvalue_matrix_shape_and_range(self, report_with_errors):
        report_with_errors.analyze_error_patterns(mc_runs=30, random_seed=1)
        pm = report_with_errors.p_value_matrix
        assert pm.shape == (len(LEVELS), len(LEVELS))
        vals = pm.values[~np.isnan(pm.values)]
        assert ((vals >= 0) & (vals <= 1)).all()


class TestGetMeanDiffOverallMean:
    def _report(self, matrix) -> MismatchReport:
        # 构造最小 report 并直接塞入 diff_matrix
        s = pd.Series([0, 1, -1, 2, -2], index=range(5))
        rep = MismatchReport(pred=s, true=s)
        rep.diff_matrix = matrix
        return rep

    def test_all_is_overall_mean_not_mean_of_means(self):
        # 列均值的均值 = mean(3, 6) = 4.5；整体均值 = (2+4+6)/3 = 4.0
        m = pd.DataFrame(
            [[2.0, np.nan], [4.0, 6.0]], index=LEVELS[:2], columns=LEVELS[:2]
        )
        rep = self._report(m)
        assert rep.get_mean_diff("all") == pytest.approx(4.0)

    def test_positive_and_negative(self):
        m = pd.DataFrame(
            [[2.0, -4.0], [-6.0, 8.0]], index=LEVELS[:2], columns=LEVELS[:2]
        )
        rep = self._report(m)
        assert rep.get_mean_diff("positive") == pytest.approx((2 + 8) / 2)
        assert rep.get_mean_diff("negative") == pytest.approx((4 + 6) / 2)
        assert rep.get_mean_diff("all") == pytest.approx((2 + 4 + 6 + 8) / 4)

    def test_empty_selection_returns_nan(self):
        m = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], index=LEVELS[:2], columns=LEVELS[:2])
        rep = self._report(m)
        assert np.isnan(rep.get_mean_diff("negative"))  # 没有负值

    def test_invalid_direction_raises(self):
        m = pd.DataFrame([[1.0]], index=LEVELS[:1], columns=LEVELS[:1])
        rep = self._report(m)
        with pytest.raises(ValueError, match="Invalid direction"):
            rep.get_mean_diff("sideways")
