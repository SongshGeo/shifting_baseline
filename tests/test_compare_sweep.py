#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""compare.py 中 sweep / 过滤相关函数的测试。

补此前未覆盖的：sweep_slices、get_filtered_corr、compare_corr_2d，以及
sweep_max_corr_year 的 off-by-2 回归（max_corr_year 必须是窗口大小，非索引）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.compare import (
    compare_corr_2d,
    get_filtered_corr,
    sweep_max_corr_year,
    sweep_slices,
)
from shifting_baseline.filters import calc_std_deviation


class TestSweepSlices:
    def test_basic_windows(self):
        slices, mid, labels = sweep_slices(
            start_year=1000, window_size=200, step_size=100, end_year=1500
        )
        # 1000-1200, 1100-1300, 1200-1400, 1300-1500
        assert len(slices) == 4
        assert slices[0] == slice(1000, 1200)
        assert slices[-1] == slice(1300, 1500)
        assert mid == [1100, 1200, 1300, 1400]
        assert labels[0] == "1000-1200"

    def test_stops_before_exceeding_end(self):
        slices, _, _ = sweep_slices(1000, 200, 100, 1450)
        # 最后一个窗口结束不得超过 end_year
        assert all(s.stop <= 1450 for s in slices)

    def test_mid_year_is_window_center(self):
        slices, mid, _ = sweep_slices(1000, 100, 100, 1300)
        for s, m in zip(slices, mid):
            assert m == int((s.start + s.stop) / 2)


class TestGetFilteredCorr:
    def test_filters_by_sample_and_significance(self):
        rs = np.array([0.5, 0.6, 0.7])
        ps = np.array([0.001, 0.5, 0.005])  # 中间不显著
        ns = np.array([100, 100, 2])  # 末位样本不足
        windows = np.array([10, 10, 10])
        out = get_filtered_corr(
            rs, ps, ns, windows, sample_threshold=2, p_threshold=0.01
        )
        # 只有第 0 个同时满足 ns/window>2 且 p<0.01
        assert out[0] == pytest.approx(0.5)
        assert np.isnan(out[1])  # 不显著
        assert np.isnan(out[2])  # 样本不足 (2/10=0.2 < 2)

    def test_all_pass(self):
        rs = np.array([0.4, 0.5])
        out = get_filtered_corr(
            rs,
            ps=np.array([0.001, 0.001]),
            ns=np.array([100, 100]),
            windows=np.array([10, 10]),
            sample_threshold=2,
            p_threshold=0.01,
        )
        np.testing.assert_allclose(out, rs)


@pytest.fixture
def correlated_pair():
    rng = np.random.default_rng(0)
    n = 300
    idx = np.arange(1000, 1000 + n)
    base = rng.normal(0, 1, n).cumsum()  # 带结构，便于滤波后仍相关
    d1 = pd.Series(base + rng.normal(0, 0.3, n), index=idx)
    d2 = pd.Series(base + rng.normal(0, 0.3, n), index=idx)
    return d1, d2


class TestCompareCorr2d:
    def test_shapes_match_input(self, correlated_pair):
        d1, d2 = correlated_pair
        windows = np.array([20, 30, 40])
        min_periods = np.array([5, 5, 5])
        rs, ps, ns = compare_corr_2d(
            d1,
            d2,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            filter_side="right",
        )
        assert rs.shape == windows.shape

    def test_rejects_mismatched_shapes(self, correlated_pair):
        d1, d2 = correlated_pair
        with pytest.raises(ValueError, match="形状不一致"):
            compare_corr_2d(
                d1,
                d2,
                windows=np.array([20, 30]),
                min_periods=np.array([5]),
            )


class TestSweepMaxCorrYearOffBy2:
    """回归 off-by-2：max_corr_year 必须是窗口大小（windows 的值），不是索引。"""

    def test_returns_window_sizes_not_indices(self, correlated_pair):
        d1, d2 = correlated_pair
        slices = [slice(1000, 1150), slice(1100, 1250)]
        # 关键：windows 不从 0/1 开始，使"窗口大小"与"索引(0..3)"可区分
        windows = np.array([20, 30, 40, 50])
        min_periods = np.array([5, 5, 5, 5])
        mcy, mc, rb, pv = sweep_max_corr_year(
            d1,
            d2,
            slices,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            filter_side="right",
            corr_method="pearson",
        )
        assert len(mcy) == len(slices)
        for arr in mcy:
            vals = set(np.atleast_1d(arr).tolist())
            # 必须全是窗口大小；绝不能是索引 {0,1,2,3}
            assert vals.issubset({20, 30, 40, 50}), f"疑似 off-by-2 回退: {vals}"
            assert not vals & {0, 1, 3}  # 索引特征值

    def test_output_lengths_align_with_slices(self, correlated_pair):
        d1, d2 = correlated_pair
        slices = [slice(1000, 1150), slice(1100, 1250), slice(1150, 1299)]
        windows = np.array([20, 30, 40])
        min_periods = np.array([5, 5, 5])
        mcy, mc, rb, pv = sweep_max_corr_year(
            d1,
            d2,
            slices,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            filter_side="right",
        )
        assert len(mcy) == len(mc) == len(rb) == len(pv) == len(slices)
