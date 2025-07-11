#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""常数
"""

from __future__ import annotations

from typing import Literal, TypeAlias

GRADE_VALUES: list[int] = [5, 4, 3, 2, 1]  # 原始等级值
STD_THRESHOLDS: list[float] = [-1.17, -0.33, 0, 0.33, 1.17]  # 标准差阈值 - 5
THRESHOLDS: list[float] = [-1.17, -0.33, 0.33, 1.17]  # 阈值 - 4
LEVELS: list[int] = [-2, -1, 0, 1, 2]  # 等级值
LEVELS_PROB: list[float] = [0.1, 0.25, 0.30, 0.25, 0.1]  # 等级值

TICK_LABELS: list[str] = ["SD", "MD", "N", "MW", "SW"]
VERBOSE_LABELS: list[str] = [
    "Very dry",
    "Moderate dry",
    "Normal",
    "Moderate wet",
    "Very wet",
]

Region: TypeAlias = Literal[
    "华北地区",
    "西北内陆区",
    "西南地区",
    "东北地区",
    "华南地区",
    "长江中下游地区",
    "青藏高原区",
]

# 历史记录数据的时间段
START = 1000
STAGE1 = 1469
STAGE2 = 1659
END = 1870
STAGES_BINS: list[int] = [START, STAGE1, STAGE2, END, 2021]
LABELS = [
    f"{START}-{STAGE1}",
    f"{STAGE1}-{STAGE2}",
    f"{STAGE2}-{END}",
    f"{END}-2021",
]
# 保留率
RETENTION_RATES = [25.17, 66.46, 80, 80]

MAX_AGE: int = 40
