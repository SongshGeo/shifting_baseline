#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import sys

from loguru import logger


def setup_logger() -> None:
    """设置日志记录器"""
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=fmt,
    )
    logger.add(
        "10days.log",
        rotation="1 week",
        retention="10 days",
        level="DEBUG",
        format=fmt,
    )


if __name__ == "__main__":
    setup_logger()
    logger.info("Hello, World!")
