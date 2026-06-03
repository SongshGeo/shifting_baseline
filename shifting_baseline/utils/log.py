#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""shifting_baseline 的日志工具。

遵循 Python 标准 logging 的库约定，与 Hydra / Mesa / ABSESpy 保持一致：

- 库代码只用命名 logger（``"shifting_baseline.*"``），自身**不挂真实 handler**，
  只挂一个 ``NullHandler``；日志记录靠 ``propagate`` 上抛到 root。
- 运行期由 Hydra 的 ``hydra/job_logging``（见 ``config/hydra/job_logging/log.yaml``）
  统一配置 root 的 console + file handler——本模块不插手，因此不会重复输出，
  也不会和 Hydra 的 file handler 抢同一个日志文件。
- 笔记本等非 Hydra 环境，显式调用 :func:`setup_notebook_logging` 让 INFO 可见。
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

PACKAGE_LOGGER_NAME = "shifting_baseline"
_CONSOLE_FORMAT = "%(levelname)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# 库最佳实践：给包根 logger 挂 NullHandler，避免"无 handler"告警，
# 也不抢占 root 的输出（NullHandler 不影响 propagate）。
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """取包内命名 logger（缺省取包根 ``"shifting_baseline"``）。

    无任何副作用——不配置 handler、不设级别。运行期 root 由 Hydra 配置，
    记录经 ``propagate`` 输出。

    Args:
        name: logger 名，通常传 ``__name__``；缺省用包根 logger。

    Returns:
        对应的 :class:`logging.Logger`。
    """
    return logging.getLogger(name or PACKAGE_LOGGER_NAME)


def setup_logger_from_hydra(cfg: Optional[object] = None) -> logging.Logger:
    """Hydra 运行下 root logger 已由 ``hydra/job_logging`` 配好，本函数无需动作。

    包内记录会 ``propagate`` 到 root 的 console(WARNING) 与 file(DEBUG) handler。
    保留此函数仅为兼容 ``__main__`` 的现有调用、并使"日志已就绪"的语义显式。

    Args:
        cfg: Hydra 配置对象（未使用，仅保留以兼容调用签名）。

    Returns:
        包根 logger。
    """
    return get_logger()


def setup_notebook_logging(
    console_level: str = "INFO",
    file_path: Optional[str] = None,
    file_level: str = "DEBUG",
) -> logging.Logger:
    """为非 Hydra（笔记本/脚本）环境配置包 logger，使 ``log.info(...)`` 可见。

    幂等：重复调用只调整级别，不会重复叠加 handler。默认只配控制台；仅当传入
    ``file_path`` 时才额外写文件——**运行期的文件日志应交给 Hydra，不要在这里配**。
    控制台 handler 加上后会把包 logger 的 ``propagate`` 关掉，避免与（如 IPython
    可能配置的）root handler 重复输出。

    Args:
        console_level: 控制台级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）。
        file_path: 可选，日志文件路径；不传则不写文件。
        file_level: 文件级别（仅 ``file_path`` 给定时生效）。

    Returns:
        配置好的包 logger。
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    console = next(
        (
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ),
        None,
    )
    if console is None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        logger.addHandler(console)
        logger.propagate = False  # 笔记本里我们自己就是 handler，避免重复输出
    console.setLevel(getattr(logging, console_level.upper()))

    levels = [getattr(logging, console_level.upper())]
    if file_path is not None:
        exists = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(file_path)
            for h in logger.handlers
        )
        if not exists:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_handler.setFormatter(
                logging.Formatter(_FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
            )
            logger.addHandler(file_handler)
        levels.append(getattr(logging, file_level.upper()))

    logger.setLevel(min(levels))
    return logger


def adjust_log_level(
    console_level: Optional[str] = None, file_level: Optional[str] = None
) -> None:
    """笔记本里动态调整包 logger 的级别（兼容旧签名）。

    若尚未配置控制台 handler，会先经 :func:`setup_notebook_logging` 建立，再调整级别。
    ``file_level`` 用于把 logger 整体级别放低，以便（如已配置）文件 handler 记录更详细。

    注意：运行期（Hydra）下**不应**调用本函数——root 已由 Hydra 负责。

    Args:
        console_level: 新的控制台级别；缺省 ``"INFO"``。
        file_level: 新的文件级别（用于下调 logger 整体级别）。
    """
    logger = setup_notebook_logging(console_level or "INFO")
    if file_level:
        logger.setLevel(min(logger.level, getattr(logging, file_level.upper())))


if __name__ == "__main__":
    # 自测：模拟笔记本环境
    _logger = setup_notebook_logging(console_level="DEBUG")
    _logger.debug("Debug message")
    _logger.info("Info message")
    _logger.warning("Warning message")
    _logger.error("Error message")
    _logger.critical("Critical message")
