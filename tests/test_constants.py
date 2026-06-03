#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""底层常量自洽性测试。

constants.py 是整条流水线的根基（filters / data / calibration / abm / plot
都依赖它）。这里只做表驱动的不变量断言，守住各组常量之间的对齐关系，
防止有人改了一处魔数却让另一处静默失同步。
"""

from __future__ import annotations

from shifting_baseline import constants as c


class TestThresholdLevelAlignment:
    """切点 / 等级 / 概率 / 标签四者长度与对齐关系。"""

    def test_four_thresholds_make_five_levels(self):
        # 4 个切点把数轴切成 5 个区间，正好对应 5 个等级
        assert len(c.THRESHOLDS) == 4
        assert len(c.LEVELS) == 5
        assert len(c.LEVELS) == len(c.THRESHOLDS) + 1

    def test_labels_align_with_levels(self):
        assert len(c.TICK_LABELS) == len(c.LEVELS)
        assert len(c.VERBOSE_LABELS) == len(c.LEVELS)

    def test_levels_are_sorted_and_symmetric(self):
        assert c.LEVELS == sorted(c.LEVELS)
        # 等级关于 0 对称（-2..2）
        assert c.LEVELS == [-x for x in reversed(c.LEVELS)]

    def test_thresholds_strictly_increasing(self):
        assert c.THRESHOLDS == sorted(c.THRESHOLDS)
        assert len(set(c.THRESHOLDS)) == len(c.THRESHOLDS)  # 无重复，无相邻相等

    def test_thresholds_symmetric_about_zero(self):
        assert c.THRESHOLDS == [-x for x in reversed(c.THRESHOLDS)]

    def test_colors_match_threshold_count(self):
        # 绘图里 zip(THRESHOLDS, COLORS)，两者必须等长
        assert len(c.COLORS) == len(c.THRESHOLDS)


class TestLevelsProb:
    def test_length_matches_levels(self):
        assert len(c.LEVELS_PROB) == len(c.LEVELS)

    def test_sums_to_one(self):
        assert abs(sum(c.LEVELS_PROB) - 1.0) < 1e-9

    def test_all_positive(self):
        assert all(p > 0 for p in c.LEVELS_PROB)

    def test_symmetric(self):
        assert c.LEVELS_PROB == list(reversed(c.LEVELS_PROB))


class TestGradeMapping:
    """历史档案 5 级 ↔ 代表性 z 值。"""

    def test_grade_values_length(self):
        assert len(c.GRADE_VALUES) == 5
        assert len(c.GRADE_VALUES) == len(c.STD_THRESHOLDS)

    def test_std_thresholds_derived_from_thresholds(self):
        # STD_THRESHOLDS 必须等于「4 个分类切点 + 中性档 0」，
        # 这是 constants.py 里两套阈值不失同步的关键不变量。
        expected = [*c.THRESHOLDS[:2], 0.0, *c.THRESHOLDS[2:]]
        assert c.STD_THRESHOLDS == expected

    def test_grade_to_z_mapping(self):
        # data.py 里 rescale_to_std 依赖这个映射顺序
        mapping = dict(zip(c.GRADE_VALUES, c.STD_THRESHOLDS))
        assert mapping == {5: -1.17, 4: -0.33, 3: 0.0, 2: 0.33, 1: 1.17}

    def test_map_keys_are_levels(self):
        assert sorted(c.MAP.keys()) == sorted(c.LEVELS)

    def test_map_symmetric_about_zero(self):
        for lvl in c.LEVELS:
            assert c.MAP[lvl] == -c.MAP[-lvl]
        assert c.MAP[0] == 0


class TestTimeStages:
    def test_stages_bins_strictly_increasing(self):
        assert c.STAGES_BINS == sorted(c.STAGES_BINS)
        assert len(set(c.STAGES_BINS)) == len(c.STAGES_BINS)

    def test_stages_bins_composition(self):
        assert c.STAGES_BINS == [c.START, c.STAGE1, c.STAGE2, c.END, c.FINAL]

    def test_stage_ordering(self):
        assert c.START < c.STAGE1 < c.STAGE2 < c.END < c.FINAL
