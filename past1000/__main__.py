#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
past1000 是一个用于处理气候模式数据，并进行气候重建的Python库。

如果直接运行此文件，则默认使用 config/config.yaml 作为配置文件。
如果需要使用其他配置文件，则可以通过命令行参数 --config-name 指定。

本文件用于对当前的工作流进行快速的批处理与输出。
"""

from hydra import main
from loguru import logger
from omegaconf import DictConfig

from past1000.api.log import setup_logger
from past1000.ci.process import batch_process_recon_data
from past1000.core.exp import batch_process_and_save_by_config
from past1000.utils.config import format_by_config

__version__ = "0.1.0"
__all__ = [
    "_main",
    "batch_process_and_save_by_config",
    "batch_process_recon_data",
]


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None):
    """根据配置文件自动化运行。"""
    if cfg is None:
        raise ValueError("cfg 不能为空")
    cfg = format_by_config(cfg)
    setup_logger(std_level=cfg.log.std)

    logger.info(f"开始以“{cfg.how.name}”模式运行。")
    pipeline = cfg.how.pipeline
    # 获取处理流程
    for step in pipeline:
        logger.info(f"开始处理 {step}.")
        # 在当前文件里，获取该步骤的函数
        try:
            func = globals()[step]
            func(cfg)
        except KeyError as e:
            logger.error(f"没有找到 {step} 函数: {e}")
        logger.info(f"步骤处理完成: {step}.")
    logger.info("日志保存在 config/log/past1000.log")


if __name__ == "__main__":
    _main()
