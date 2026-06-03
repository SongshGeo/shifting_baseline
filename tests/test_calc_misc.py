#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""utils/calc.py 中此前未覆盖的辅助函数测试：

low_pass_filter（论文 ~30 年低通滤波本体）、get_significance_stars、
fill_star_matrix、get_interval。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.constants import EXTREME_STD_CAP, LEVELS, THRESHOLDS
from shifting_baseline.utils.calc import (
    fill_star_matrix,
    get_interval,
    get_significance_stars,
    low_pass_filter,
)


class TestLowPassFilter:
    def test_rolling_mean_smooths(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 200), index=range(1000, 1200))
        out = low_pass_filter(s, window_size=30)
        assert isinstance(out, pd.Series)
        assert len(out) == len(s)
        assert out.index.equals(s.index)
        assert out.var() < s.var()  # 高频被滤掉，方差下降

    def test_constant_input_preserved(self):
        s = pd.Series([5.0] * 100, index=range(100))
        out = low_pass_filter(s, window_size=10)
        assert np.allclose(out.dropna(), 5.0)

    def test_numpy_input_accepted(self):
        out = low_pass_filter(np.arange(100.0), window_size=10)
        assert isinstance(out, pd.Series)
        assert len(out) == 100

    def test_center_true_on_linear_ramp(self):
        # 居中滚动均值在线性斜坡的内部点上等于中心值
        s = pd.Series(np.arange(100.0))
        out = low_pass_filter(s, window_size=11, center=True)
        assert out.iloc[50] == pytest.approx(50.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown filtering method"):
            low_pass_filter(pd.Series(np.arange(10.0)), method="wavelet")


class TestSignificanceStars:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.0, "***"),
            (0.005, "***"),
            (0.009, "***"),
            (0.01, "**"),  # 边界：0.01 不 < 0.01 → ** 档
            (0.03, "**"),
            (0.049, "**"),
            (0.05, "*"),  # 边界：0.05 不 < 0.05 → * 档
            (0.08, "*"),
            (0.099, "*"),
            (0.1, ""),
            (0.5, ""),
            (1.0, ""),
        ],
    )
    def test_tiers(self, p, expected):
        assert get_significance_stars(p) == expected

    def test_nan_returns_empty(self):
        assert get_significance_stars(np.nan) == ""

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="between 0 and 1"):
            get_significance_stars(bad)


class TestFillStarMatrix:
    def test_formats_value_and_stars(self):
        values = pd.DataFrame(
            [[0.5, np.nan], [1.234, -0.7]], index=["a", "b"], columns=["x", "y"]
        )
        pvals = pd.DataFrame(
            [[0.01, 0.5], [0.08, 0.2]], index=["a", "b"], columns=["x", "y"]
        )
        out = fill_star_matrix(pvals, values)
        assert out.loc["a", "x"] == "0.50**"  # p=0.01 → **
        assert out.loc["a", "y"] == ""  # value NaN → 空
        assert out.loc["b", "x"] == "1.23*"  # p=0.08 → *
        assert out.loc["b", "y"] == "-0.70"  # p=0.2 → 无星


class TestGetInterval:
    def test_values_match_original(self):
        assert get_interval(-2) == (-2.0, -1.17)
        assert get_interval(-1) == (-1.17, -0.33)
        assert get_interval(0) == (-0.33, 0.33)
        assert get_interval(1) == (0.33, 1.17)
        assert get_interval(2) == (1.17, 2.0)

    def test_derived_from_thresholds(self):
        # 内部切点与 constants.THRESHOLDS 同源
        assert get_interval(-1)[1] == THRESHOLDS[1]
        assert get_interval(1)[0] == THRESHOLDS[2]

    def test_extremes_capped(self):
        assert get_interval(-2)[0] == -EXTREME_STD_CAP
        assert get_interval(2)[1] == EXTREME_STD_CAP

    def test_intervals_contiguous(self):
        # 相邻区间首尾相接，连续覆盖 [-cap, cap]
        ivs = [get_interval(lvl) for lvl in LEVELS]
        for (_, upper), (lower_next, _) in zip(ivs, ivs[1:]):
            assert upper == lower_next

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="无效等级"):
            get_interval(3)
