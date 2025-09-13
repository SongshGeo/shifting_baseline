#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="test_config.yaml")
    return cfg


@pytest.fixture(name="series")
def fixture_series():
    """用来测试的系列数据包含1000个数据，每年有一个随机的[-4, 4]的数，但符合正态分布，0 是正常值"""
    data = np.random.normal(0, 1, size=1000).clip(-4, 4)
    return pd.Series(data, index=np.arange(850, 1850))


@pytest.fixture(scope="session", name="python_bin")
def fixture_python_bin() -> str:
    """Python executable for subprocess runs."""
    return sys.executable


@pytest.fixture(name="repo_root")
def fixture_repo_root() -> Path:
    """Repository root path."""
    return Path(__file__).resolve().parents[1]


def _make_fake_excel(
    region_names: list[str], years: list[int], cities: list[str]
) -> dict[str, pd.DataFrame]:
    """Build in-memory Excel-like data for multiple regions."""
    rng = np.random.default_rng(123)
    sheets: dict[str, pd.DataFrame] = {}
    for region in region_names:
        data = rng.integers(low=0, high=6, size=(len(years), len(cities)))
        df = pd.DataFrame(data, index=years, columns=cities)
        df.index.name = "year"
        sheets[region] = df
    return sheets


@pytest.fixture(name="excel_file")
def fixture_excel_file(tmp_path: Path) -> Path:
    """Create a temporary multi-sheet Excel with zeros as missing values."""
    years = list(range(1000, 1011))
    cities = ["北京", "天津", "保定"]
    regions = ["华北地区", "华南地区"]
    sheets = _make_fake_excel(regions, years, cities)
    sheets["华北地区"].loc[1003, :] = 0
    sheets["华北地区"].loc[1007, "天津"] = 0
    xlsx = tmp_path / "hist.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        for region, df in sheets.items():
            df.to_excel(writer, sheet_name=region)
    return xlsx


@pytest.fixture(name="shp_file")
def fixture_shp_file(tmp_path: Path) -> Path:
    """Dummy path for shapefile; reading will be monkeypatched."""
    return tmp_path / "dummy.shp"


@pytest.fixture(name="test_df")
def fixture_test_df():
    """
    测试数据，value 是自然记录，expect 是历史记录，classified 是分类结果


    value	classified	expect	exact	last	diff
    0	0.11	0	0	True	NaN	NaN
    1	0.34	1	1	True	NaN	NaN
    2	1.10	2	1	False	NaN	NaN
    3	0.10	-1	0	False	NaN	NaN
    4	0.32	2	2	True	1.10	-0.78
    5	0.33	0	2	False	0.11	0.22
    6	3.00	2	2	True	0.32	2.68
    7	0.11	0	0	True	0.33	-0.22
    """
    return pd.DataFrame(
        {
            "value": [0.11, 0.34, 1.10, 0.10, 0.32, 0.33, 3.0, 0.11],
            "classified": [0, 1, 2, -1, 2, 0, 2, 0],
            "expect": [0, 1, 1, 0, 2, 2, 2, 0],
        }
    )
