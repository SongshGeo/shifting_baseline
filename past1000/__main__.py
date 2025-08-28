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

from past1000.data import load_data
from past1000.mc import combine_reconstructions
from past1000.process import batch_process_recon_data
from past1000.utils.config import format_by_config, get_output_dir

__version__ = "0.1.0"
__all__ = [
    "_main",
    "batch_process_recon_data",
]


def _test_logging():
    """Test logging functionality with all levels."""
    log = logging.getLogger(__name__)
    log.debug("这是一条 DEBUG 日志消息")
    log.info("这是一条 INFO 日志消息")
    log.warning("这是一条 WARNING 日志消息")
    log.error("这是一条 ERROR 日志消息")
    log.critical("这是一条 CRITICAL 日志消息")
    log.info("日志测试完成，程序退出")


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None):
    """根据配置文件自动化运行。"""
    if cfg is None:
        raise ValueError("cfg 不能为空")
    cfg = format_by_config(cfg)

    # Check if in test mode
    if cfg.get("test_mode", False):
        _test_logging()
        return

    log = logging.getLogger(__name__)
    log.info("实验开始，配置文件请参看 %s", get_output_dir() / ".hydra/config.yaml")
    log.info("Step 1: 加载数据 ...")
    datasets, uncertainties, _ = load_data(cfg)
    log.info("Step 2: 比较每个树轮数据")
    # TODO 需要添加一个函数，用于比较每个树轮数据
    log.info("Step 3: 整合树轮数据")
    _, _ = combine_reconstructions(
        reconstructions=datasets,
        uncertainties=uncertainties,
        standardize=True,
    )
    log.info("Step 4: 历史数据时期对比")
    # history.detect_mismatch(
    #     true_data=combined,
    # )


if __name__ == "__main__":
    _main()
