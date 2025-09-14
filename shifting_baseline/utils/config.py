#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import re
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def format_by_config(config: DictConfig, string: Optional[str] = None) -> str:
    """使用配置值格式化单个字符串

    Args:
        config: OmegaConf 配置对象
        string: 需要格式化的字符串，如 "{ds.root}/data"

    Returns:
        格式化后的字符串
    """
    if not isinstance(config, DictConfig):
        raise ValueError("config 必须是 OmegaConf 配置对象")

    if string is None:
        OmegaConf.resolve(config)
        return config

    pattern = r"\$\{([^}]+)\}"
    paths = re.findall(pattern, string)

    result = string
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is None:
            raise KeyError(f"配置中未找到路径: {path}")
        result = result.replace(f"${{{path}}}", str(value))

    return result


def get_output_dir() -> Path:
    """获取输出目录"""
    return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
