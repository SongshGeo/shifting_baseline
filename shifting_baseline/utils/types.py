#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from typing import Literal

    CorrFunc = Literal["pearson", "kendall", "spearman"]
    FilterSide = Literal["both", "left", "right"]
    HistoricalAggregateType = Literal["mean", "median", "mode"]

    Region: TypeAlias = Literal[
        "华北地区",  # North China
        "西北内陆区",  # Northwest inland area
        "西南地区",  # Southwest China
        "东北地区",  # Northeast China
        "华南地区",  # South China
        "长江中下游地区",  # Middle and Lower Yangtze River Valley
        "青藏高原区",  # Tibet Plateau
    ]

    # stage 可为 1-4 期编号、年份 slice，或 "all" 等字符串
    Stages: TypeAlias = int | slice | str | Literal[1, 2, 3, 4]
    ToStdMethod: TypeAlias = Literal["mapping", "sampling"]

    # climate forcing, climate process, subannual aggregation
    SubannualAggregation: TypeAlias = Literal["mean", "sum", "last"]
    ClimateProcess: TypeAlias = Literal["iid", "ar1", "trend_plus_noise", "ar1_trend"]
