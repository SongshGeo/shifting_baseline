#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import re
from pathlib import Path
from typing import List, Optional, TypeAlias

import click
from loguru import logger

from past1000.api.log import setup_logger

PathLike: TypeAlias = str | Path

setup_logger()


def check_data_dir(
    path: Optional[PathLike] = None,
    create: bool = False,
) -> Path:
    """检查数据目录，如果不存在则创建。

    Args:
        path: 要检查的目录。
        create: 如果目录不存在，是否创建。

    Returns:
        检查的目录。
    """
    # 如果 path 为 None，则使用当前工作目录
    if path is None:
        path = Path.cwd()
    path = Path(path)
    # 检查路径是否为目录
    if not path.is_dir():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            msg = f"创建目录: {path}"
            logger.info(msg)
        else:
            msg = f"目录不存在: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
    logger.info(f"目录 {path} 检查完毕。")
    return path


def get_matching_files(
    directory: PathLike,
    pattern: str,
) -> List[str]:
    """
    获取指定目录下所有符合正则表达式的文件名列表。

    Args:
        directory: 要搜索的目录路径。
        pattern: 正则表达式模式。

    Returns:
        符合正则表达式的文件名列表。
    """
    directory = Path(directory)
    logger.info(f"搜索目录: {directory}")
    logger.debug(f"正则表达式: {pattern}")
    regex = re.compile(pattern)
    matching_files = [
        file.name
        for file in directory.iterdir()
        if file.is_file() and regex.match(file.name)
    ]
    logger.info(f"找到 {len(matching_files)} 个文件。")
    return matching_files


def create_cmip_regex(
    variable: Optional[str] = None,
    model: Optional[str] = None,
    frequency: Optional[str] = None,
    experiment: Optional[str] = None,
    ensemble: Optional[str] = None,
) -> str:
    """创建用于匹配CMIP文件名的正则表达式。

    Args:
        variable: 变量名。
        model: 模型名。
        frequency: 频率。
        experiment: 实验名。
        ensemble: 集合名。

    Returns:
        用于匹配CMIP文件名的正则表达式。
    """
    logger.debug(f"变量名: {variable}")
    logger.debug(f"模型名: {model}")
    logger.debug(f"频率: {frequency}")
    logger.debug(f"实验名: {experiment}")
    logger.debug(f"集合名: {ensemble}")
    pattern = [
        r"(?P<variable>{})".format(variable or r"[\w]+"),
        r"(?P<frequency>{})".format(frequency or r"[\w]+"),
        r"(?P<model>{})".format(re.escape(model) if model else r"[\w-]+"),
        r"(?P<experiment>{})".format(experiment or r"[\w]+"),
        r"(?P<ensemble>{})".format(ensemble or r"[\w]+"),
        r"[\d]{6}-[\d]{6}",
    ]
    return r"_".join(pattern) + r"\.nc$"


@click.command()
@click.argument("path", type=click.Path(exists=False), default=None)
@click.option("--create", "-c", is_flag=True, help="如果目录不存在，是否创建")
def cli(path: Optional[str] = None, create: bool = False):
    """检查数据目录，如果不存在则创建（当指定 --create 选项时）。

    PATH: 要检查的目录路径
    """
    check_data_dir(path, create)


if __name__ == "__main__":
    cli()
