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
STD_THRESHOLDS: list[float] = [-1.17, -0.33, 0, 0.33, 1.17]  # 标准差阈值


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
END = 1949
STAGES_BINS: list[int] = [START, STAGE1, STAGE2, END, 2021]
LABELS = [
    f"{START}-{STAGE1}",
    f"{STAGE1}-{STAGE2}",
    f"{STAGE2}-{END}",
    f"{END}-2021",
]
STAGE_LABELS = [
    f"Stage 1: {START}-{STAGE1}\nRetention Rate 25.17%",
    f"Stage 2: {STAGE1}-{STAGE2}\nRetention Rate 66.46%",
    f"Stage 3: {STAGE2}-{END}\nRetention Rate 80%",
    f"Stage 4: {END}-2020\nRetention Rate 80%",
]
