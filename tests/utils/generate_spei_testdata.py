#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""这个脚本用于生成SPEI测试数据

数据基于 Yaping Wang 提供的降水、潜在蒸散发数据，以及 SPEI 计算结果。
选取 2015-2020 年黄河流域 105E-106E, 36N-37N 范围内的数据，生成测试数据。
"""

from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from hydra import main
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from past1000.api.io import check_data_dir
from past1000.ci.spei import DistributionType

# 选取黄河流域 105E-106E, 36N-37N 范围内的数据
YEARS = None
LON = slice(105, 106)
LAT = slice(36, 37)
SCALES: list[str] = ["01", "03", "06", "12"]
DISTS: list[DistributionType] = ["gamma", "pearson"]


def generate_spei_testdata(path_in: Path, pattern: str, path_out: Path) -> None:
    """生成SPEI测试数据

    参数：
        folder: 数据文件夹，所有数据文件的父目录
        pattern: 数据文件名模式，用于匹配数据文件，支持 dist 和 freq 两个变量，分别是干旱指数的分布类型和时间尺度。
    """
    for scale, dist in product(SCALES, DISTS):
        expected = xr.open_dataarray(path_in / pattern.format(dist=dist, freq=scale))
        expected.data = np.flipud(expected.data)

        selected = expected.sel(lon=LON, lat=LAT)
        selected.to_netcdf(path_out / f"spei_{dist}_{scale}.nc")


def generate_input_testdata(
    path_in: Path,
    input_name: str,
    path_out: Path,
    output_name: str,
) -> None:
    """生成输入数据"""
    data = xr.open_dataarray(path_in / input_name)
    data.data = np.flipud(data.data)
    data.sel(lon=LON, lat=LAT).to_netcdf(path_out / output_name)


@main(config_path="../../config", config_name="config", version_base=None)
def generate_test_data(cfg: Optional[OmegaConf] = None):
    """生成测试数据"""
    if cfg is None:
        raise ValueError("Configuration is not provided.")
    # 原始数据所在目录
    path_in = check_data_dir(cfg.ds.spei.folder)
    # 输出数据所在目录
    path_out = Path(get_original_cwd()) / "tests/data"
    # 生成SPEI测试数据
    generate_spei_testdata(path_in, cfg.ds.spei.pattern, path_out / "expected")
    # 生成输入数据
    generate_input_testdata(
        path_in=path_in,
        input_name=cfg.ds.spei.pet,
        path_out=path_out / "input",
        output_name="pet.nc",
    )
    generate_input_testdata(
        path_in=path_in,
        input_name=cfg.ds.spei.pr,
        path_out=path_out / "input",
        output_name="pr.nc",
    )


if __name__ == "__main__":
    generate_test_data()
