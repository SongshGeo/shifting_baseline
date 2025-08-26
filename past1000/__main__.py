#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
past1000 是一个用于对比历史集体记忆和气候重建资料的 Python 库。
"""

import logging

from hydra import main
from omegaconf import DictConfig

from past1000.process import batch_process_recon_data
from past1000.utils.config import format_by_config

__version__ = "0.1.0"
__all__ = [
    "_main",
    "batch_process_recon_data",
]


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None):
    """根据配置文件自动化运行。"""
    if cfg is None:
        raise ValueError("cfg 不能为空")
    cfg = format_by_config(cfg)
    log = logging.getLogger(__name__)
    log.debug("开始以“%s”模式运行。", cfg.how.name)
    log.info("测试")
    log.warning("测试")
    log.error("测试")
    log.critical("测试")


if __name__ == "__main__":
    _main()
