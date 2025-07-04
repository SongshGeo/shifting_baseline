#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(
    std_level: str = "WARNING",
    file_level: str = "DEBUG",
    file_name: str = "10days.{time:YYYY-MM-DD}.log",
) -> None:
    """设置日志记录器"""
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(
        sys.stderr,
        level=std_level,
        format=fmt,
    )
    logger.add(
        file_name,
        rotation="1 day",
        retention="1 week",
        level=file_level,
        format=fmt,
    )
    logger.info(f"新一次运行，日志输出等级: {std_level}")
    logger.info(f"日志文件名: {file_name}，记录等级: {file_level}")


if __name__ == "__main__":
    setup_logger(std_level="DEBUG", file_level="DEBUG")
    logger.info("Hello, World!")
