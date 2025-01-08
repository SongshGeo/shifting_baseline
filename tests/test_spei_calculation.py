#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
测试 SPEI 计算
"""

from functools import partial

import pytest
import xarray as xr

from past1000.ci.spei import calc_single_spei


@pytest.mark.parametrize(
    "dist, scale",
    [
        ("gamma", 1),
        ("pearson", 1),
        ("gamma", 3),
        ("pearson", 3),
        ("gamma", 6),
        ("pearson", 6),
        ("gamma", 12),
        ("pearson", 12),
    ],
)
def test_spei_calculation(pr, pet, expected_spei_dict, dist, scale):
    """测试 SPEI 计算"""
    # arrange
    expected_spei = expected_spei_dict[(dist, scale)]
    func = partial(calc_single_spei, distribution=dist, scale=scale)
    # act
    spei = xr.apply_ufunc(
        func,
        pr,
        pet,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
    # assert
    xr.testing.assert_allclose(spei, expected_spei, rtol=1e-1)
