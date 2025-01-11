#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""测试能从配置文件中获取输入输出路径。
"""
import pytest
from omegaconf import DictConfig

from past1000.utils.config import format_by_config


@pytest.mark.parametrize(
    "string, expected",
    [
        ("${ds.folder}", "~/Downloads"),
        ("${ds.root}/data", "~/Documents/data"),
    ],
)
def test_format_config_paths(cfg: DictConfig, string: str, expected: str):
    """测试能从配置文件中获取输入输出路径。"""
    formatted_config = format_by_config(cfg, string)
    assert formatted_config == expected


def test_format_config_recursive(cfg: DictConfig):
    """测试能递归处理配置文件中的变量引用。"""
    assert cfg.test2.a == "~/Documents/data"
