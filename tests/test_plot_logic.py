#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""utils/plot.py 中纯数据逻辑（非绘图）的测试。

绘图本身不单测；这里覆盖可测的纯函数：
- _filter_improvements —— headline 图改进值过滤（含历史 bug 回归）
- _compute_y_value —— mismatch 柱状图各 y 指标的数学
- _prepare_mismatch_data —— 失配配对生成
- _compute_bar_positions —— 组内柱位置
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.constants import LEVELS
from shifting_baseline.utils.plot import (
    _compute_bar_positions,
    _compute_y_value,
    _filter_improvements,
    _prepare_mismatch_data,
)


class TestFilterImprovements:
    """回归 plot_correlation_windows 曾经的"多 if 把越界值追加两次"bug。"""

    DATA = [
        np.array([0.1]),
        np.array([-0.2]),  # 越界 → NaN
        np.array([0.3]),
        np.array([1.5]),  # 越界 → NaN
        np.array([0.4]),
    ]

    def test_length_strictly_preserved(self):
        out = _filter_improvements(self.DATA)
        assert len(out) == len(self.DATA)  # 每个输入恰好一个输出

    def test_out_of_range_to_nan_in_place(self):
        out = _filter_improvements(self.DATA)
        np.testing.assert_array_equal(np.isnan(out), [False, True, False, True, False])
        assert out[0] == pytest.approx(0.1)
        assert out[2] == pytest.approx(0.3)
        assert out[4] == pytest.approx(0.4)

    def test_boundaries_0_and_1_kept(self):
        out = _filter_improvements([np.array([0.0]), np.array([1.0])])
        assert not np.isnan(out).any()

    def test_uses_window_mean(self):
        out = _filter_improvements([np.array([0.2, 0.4])])  # mean 0.3
        assert out[0] == pytest.approx(0.3)


class TestComputeYValue:
    def test_count(self):
        assert _compute_y_value(7, 0.5, "count", 0.5, False, 100, 1, 1) == 7

    def test_diff(self):
        assert _compute_y_value(7, 0.5, "diff", 0.5, False, 100, 1, 1) == 0.5

    def test_abs_diff(self):
        assert _compute_y_value(7, -0.5, "abs_diff", 0.5, False, 100, 1, 1) == 0.5

    def test_proportion_default(self):
        assert _compute_y_value(25, 0.5, "proportion", 0.5, False, 100, 1, 1) == 0.25

    def test_weighted(self):
        # w*prop + (1-w)*norm_abs_diff = 0.5*0.25 + 0.5*(0.5/1.0)
        y = _compute_y_value(25, 0.5, "weighted", 0.5, False, 100, 1.0, 1)
        assert y == pytest.approx(0.5 * 0.25 + 0.5 * 0.5)

    def test_weighted_signed_keeps_sign(self):
        y = _compute_y_value(25, -0.5, "weighted", 0.5, True, 100, 1.0, 1)
        assert y < 0

    def test_weighted_max_abs_diff_zero_guarded(self):
        # max_abs_diff==0 时 norm_abs_diff 应取 0，不除零
        y = _compute_y_value(25, 0.0, "weighted", 0.5, False, 100, 0.0, 1)
        assert y == pytest.approx(0.5 * 0.25)

    def test_contribution(self):
        assert _compute_y_value(25, 0.5, "contribution", 0.5, False, 100, 1, 1) == (
            pytest.approx(0.125)
        )

    def test_norm_contribution(self):
        # (count*diff)/contrib_den = (4*0.5)/8
        assert _compute_y_value(4, 0.5, "norm_contribution", 0.5, False, 100, 1, 8) == (
            pytest.approx(0.25)
        )


class TestPrepareMismatchData:
    @staticmethod
    def _mats():
        idx = pd.Index(LEVELS, name="true")
        cols = pd.Index(LEVELS, name="pred")
        diff = pd.DataFrame(0.0, index=idx, columns=cols)
        count = pd.DataFrame(1.0, index=idx, columns=cols)
        pval = pd.DataFrame(0.5, index=idx, columns=cols)
        return diff, count, pval

    def test_excludes_diagonal_20_pairs(self):
        long = _prepare_mismatch_data(*self._mats())
        assert len(long) == 20  # 5x5 去掉 5 个对角
        assert (long["true"] != long["pred"]).all()

    def test_offset_and_label(self):
        long = _prepare_mismatch_data(*self._mats())
        row = long[(long["true"] == 2) & (long["pred"] == -2)].iloc[0]
        assert row["offset"] == 4
        assert row["abs_offset"] == 4
        assert row["label"] == "2--2"


class TestComputeBarPositions:
    def test_centers_and_counts(self):
        long = pd.DataFrame({"group": [1, 1, 1, 2]})
        centers, offsets = _compute_bar_positions(long, groups=[1, 2])
        assert centers == {1: 0, 2: 1}
        assert len(offsets[1]) == 3
        assert len(offsets[2]) == 1

    def test_positions_within_group_width(self):
        long = pd.DataFrame({"group": [1, 1, 1]})
        centers, offsets = _compute_bar_positions(long, groups=[1])
        for x in offsets[1]:
            assert centers[1] - 0.4 <= x <= centers[1] + 0.4

    def test_empty_group_skipped(self):
        long = pd.DataFrame({"group": [1]})
        centers, offsets = _compute_bar_positions(long, groups=[1, 2])
        assert 2 not in offsets  # group 2 无成员
