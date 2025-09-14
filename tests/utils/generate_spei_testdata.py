#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""SPEI 测试数据生成脚本

本脚本用于从完整的 SPEI 数据集中提取测试数据样本。主要功能包括：

1. 提取特定时空范围的数据：
   - 时间范围：2000-2020年
   - 空间范围：黄河流域 (105°E-106°E, 36°N-37°N)

2. 生成两类测试数据：
   - 输入数据：降水(pr)和潜在蒸散发(pet)
   - 预期输出：不同参数组合下的SPEI计算结果
     * 分布类型：gamma和pearson
     * 时间尺度：1、3、6、12个月

使用方法：
    python generate_spei_testdata.py

数据来源：
    基于 Yaping Wang 提供的降水、潜在蒸散发数据，以及 SPEI 计算结果。

输出位置：
    - tests/data/input/: 输入数据
    - tests/data/expected/: 预期输出数据
"""

from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from hydra import main
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from shifting_baseline.api.io import check_data_dir
from shifting_baseline.ci.spei import DistributionType

# 配置参数
START_YEAR = 2000
END_YEAR = 2020
YEARS = slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
LON = slice(105, 106)
LAT = slice(36, 37)
SCALES: list[str] = ["01", "03", "06", "12"]  # SPEI计算的时间尺度
DISTS: list[DistributionType] = ["gamma", "pearson"]  # SPEI计算的分布类型


def generate_spei_testdata(folder: Path, pattern: str):
    """生成SPEI测试数据

    从完整的SPEI数据集中提取测试样本，并保存为netCDF格式。

    Args:
        folder: 源数据文件夹路径
        pattern: 源数据文件名模式，支持{dist}和{freq}两个变量替换
            - dist: 分布类型 (gamma/pearson)
            - freq: 时间尺度 (01/03/06/12)

    输出:
        在 tests/data/expected/ 下生成多个netCDF文件，命名格式为：
        spei_{dist}_{scale}.nc
    """
    cwd = Path(get_original_cwd())
    data_dir = cwd / "data"
    (data_dir / "expected").mkdir(parents=True, exist_ok=True)

    for scale, dist in product(SCALES, DISTS):
        expected = xr.open_dataarray(folder / pattern.format(dist=dist, freq=scale))
        expected.data = np.flipud(expected.data)
        selected = expected.sel(time=YEARS, lon=LON, lat=LAT)
        selected.to_netcdf(data_dir / "expected" / f"spei_{dist}_{scale}.nc")


def generate_input_testdata(folder: Path, input_name: str, output_name: str):
    """生成输入测试数据

    从完整的输入数据集中提取测试样本。

    Args:
        folder: 源数据文件夹路径
        input_name: 输入文件名
        output_name: 输出文件名

    输出:
        在 tests/data/input/ 下生成netCDF文件
    """
    cwd = Path(get_original_cwd())
    data_dir = cwd / "data"
    (data_dir / "input").mkdir(parents=True, exist_ok=True)

    data = xr.open_dataarray(folder / input_name)
    data.data = np.flipud(data.data)
    data.to_netcdf(data_dir / "input" / output_name)


@main(config_path="../../config", config_name="config")
def generate_test_data(cfg: Optional[OmegaConf] = None):
    """主函数：生成所有测试数据

    使用hydra管理配置，从config文件中读取数据路径等信息。

    Args:
        cfg: hydra配置对象，包含数据路径等信息
    """
    if cfg is None:
        raise FileExistsError("配置文件不存在")
    folder = Path(cfg.ds.spei.folder)
    check_data_dir(folder)

    # 生成SPEI预期输出数据
    generate_spei_testdata(folder, cfg.ds.spei.pattern)

    # 生成输入数据（降水和潜在蒸散发）
    generate_input_testdata(folder, cfg.ds.spei.pet, f"pet_{START_YEAR}-{END_YEAR}.nc")
    generate_input_testdata(folder, cfg.ds.spei.pr, f"pr_{START_YEAR}-{END_YEAR}.nc")


if __name__ == "__main__":
    generate_test_data()
