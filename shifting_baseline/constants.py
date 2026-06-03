#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""常数
"""

from __future__ import annotations

# ── 旱涝分类：切点 / 等级 / 先验概率 ───────────────────────────────
# 4 个 z-score 切点把数轴分成 5 个旱涝等级（classify 的默认切点）
THRESHOLDS: list[float] = [-1.17, -0.33, 0.33, 1.17]  # 阈值 - 4
LEVELS: list[int] = [-2, -1, 0, 1, 2]  # 等级值（对应 5 个区间）

# 这里设置的先验概率和 《历史旱涝地图集》的先验概率一致
LEVELS_PROB: list[float] = [0.1, 0.25, 0.30, 0.25, 0.1]  # 各等级先验概率（和=1）

# ── 历史档案 5 级 → 标准差 ─────────────────────────────────────────
GRADE_VALUES: list[int] = [5, 4, 3, 2, 1]  # 原始等级值（旱→涝）
# 5 级档案值的代表性 z 值（中性档=0，其余直接取分类切点）。
# 从 THRESHOLDS 派生，确保两套阈值不会失同步。
STD_THRESHOLDS: list[float] = [*THRESHOLDS[:2], 0.0, *THRESHOLDS[2:]]  # 标准差阈值 - 5

COLORS = ["#EF7722", "#FAA533", "#BBDCE5", "#0BA6DF"]
TICK_LABELS: list[str] = ["SD", "MD", "N", "MW", "SW"]
VERBOSE_LABELS: list[str] = [
    "Very dry",
    "Moderate dry",
    "Normal",
    "Moderate wet",
    "Very wet",
]

# 等级映射为标准差值
MAP = {
    -2: -1.5,
    -1: -0.5,
    0: 0,
    1: 0.5,
    2: 1.5,
}

# 历史记录数据的时间段
START = 1000
STAGE1 = 1469
STAGE2 = 1659
END = 1900
FINAL = 2000

STAGES_BINS: list[int] = [START, STAGE1, STAGE2, END, FINAL]
