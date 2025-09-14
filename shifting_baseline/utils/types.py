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
    DistributionType = Literal["pearson", "gamma"]
    HistoricalAggregateType = Literal["mean", "median", "mode"]

    Region: TypeAlias = Literal[
        "华北地区",
        "西北内陆区",
        "西南地区",
        "东北地区",
        "华南地区",
        "长江中下游地区",
        "青藏高原区",
    ]

    Stages: TypeAlias = int | slice | str | Literal[1, 2, 3, 4]
    ToStdMethod: TypeAlias = Literal["mapping", "sampling"]
