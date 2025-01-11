#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from pathlib import Path

import pytest
import xarray as xr
from hydra import compose, initialize

DATA_DIR = Path("tests/data")


@pytest.fixture(name="pr")
def pr_fixture():
    """降水数据"""
    return xr.open_dataarray(DATA_DIR / "input" / "pr.nc")


@pytest.fixture(name="pet")
def pet_fixture():
    """潜在蒸散发数据"""
    return xr.open_dataarray(DATA_DIR / "input" / "pet.nc")


@pytest.fixture(name="expected_spei_dict")
def expected_spei_dict_fixture():
    """预期 SPEI 数据

    返回一个字典，键为 (dist, scale)，值为预期 SPEI 数据。
    dist 为分布类型，可选 "gamma" 或 "pearson"。
    scale 为尺度，可选 1, 3, 6, 12。
    """
    expected_spei_dict = {}
    for dist in ["gamma", "pearson"]:
        for scale in [1, 3, 6, 12]:
            data_path = DATA_DIR / "expected" / f"spei_{dist}_{scale:02d}.nc"
            expected_spei = xr.open_dataarray(data_path)
            expected_spei_dict[(dist, scale)] = expected_spei
    return expected_spei_dict


@pytest.fixture(name="cfg")
def fixture_cfg():
    """配置文件"""
    with initialize(config_path="config"):
        cfg = compose(config_name="test_config.yaml")
    return cfg
