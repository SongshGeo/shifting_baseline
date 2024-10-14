#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from pathlib import Path
from typing import Optional, TypeAlias

import click
from log import setup_logger
from loguru import logger

PathLike: TypeAlias = str | Path


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


@click.command()
@click.argument("path", type=click.Path(exists=False), default=None)
@click.option("--create", "-c", is_flag=True, help="如果目录不存在，是否创建")
def cli(path: Optional[str] = None, create: bool = False):
    """检查数据目录，如果不存在则创建（当指定 --create 选项时）。

    PATH: 要检查的目录路径
    """
    check_data_dir(path, create)


if __name__ == "__main__":
    setup_logger()
    cli()
