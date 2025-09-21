#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
shifting_baseline 是一个用于对比历史集体记忆和气候重建资料的 Python 库。
"""

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from hydra import main
from omegaconf import DictConfig

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.compare import (
    experiment_corr_2d,
    sweep_max_corr_year,
    sweep_slices,
)
from shifting_baseline.constants import END, STAGE1
from shifting_baseline.data import load_data, load_validation_data
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.mc import combine_reconstructions
from shifting_baseline.process import batch_process_recon_data
from shifting_baseline.utils.config import format_by_config, get_output_dir
from shifting_baseline.utils.log import get_logger, setup_logger_from_hydra
from shifting_baseline.utils.plot import plot_correlation_windows

if TYPE_CHECKING:
    from geo_dskit.utils.path import PathLike

    from shifting_baseline.utils.types import Stages

__version__ = "0.1.0"
__all__ = [
    "_main",
    "batch_process_recon_data",
]


def _test_logging():
    """Test logging functionality with all levels."""
    log = get_logger(__name__)
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
    # Set up logging from Hydra configuration
    setup_logger_from_hydra(cfg)

    log = get_logger(__name__)
    out_dir = get_output_dir()
    log.info("实验开始，配置文件请参看 %s", out_dir / ".hydra/config.yaml")
    log.info("Step 1: 加载数据 ...")
    combined, uncertainties, history = load_data(cfg)
    _, regional_prec_z = load_validation_data(
        cfg.ds.out.precip_z,
        resolution=cfg.resolution,
        regional_csv=cfg.ds.out.regional_csv,
    )
    # log.info("Step 2: 整合树轮数据")
    log.info("Step 3: 比较每个树轮数据")
    # TODO 需要添加一个函数，用于比较每个树轮数据
    log.info("Step 4: 比较树轮数据和测试数据 z-score")
    tree_ring = combined["mean"]
    control_mismatch_report = MismatchReport(
        pred=classify(regional_prec_z),
        true=classify(tree_ring),
        value_series=tree_ring,
    )
    control_mismatch_report.analyze_error_patterns()
    control_mismatch_report.generate_report_figure(
        save_path=out_dir / "control_mismatch.png"
    )
    log.info("Step 5: 分时期对比历史数据和整合树轮数据")
    slice_now = history.get_time_slice("2-3")
    his, nat = history.aggregate(cfg.agg_method, inplace=True).merge_with(
        combined["mean"],
        time_range=slice_now,
        split=True,
    )
    mismatch_report = MismatchReport(
        pred=his,
        true=classify(nat),
        value_series=nat,
    )
    mismatch_report.analyze_error_patterns()
    log.debug(mismatch_report.get_statistics_summary(as_str=True))
    mismatch_report.generate_report_figure(save_path=out_dir / "mismatch_2-3.png")
    _, r_benchmark, ax = experiment_corr_2d(
        data1=his,
        data2=nat,
        time_slice=slice_now,
        corr_method=cfg.corr_method,
        filter_func=calc_std_deviation,
        filter_side=cfg.filter_side,
        n_diff_w=5,
        std_offset=0.1,
    )
    ax.set_title(
        f"{slice_now.start}-{slice_now.stop} AD. $Tau={r_benchmark:.3f}$",
        fontsize=9,
    )
    ax.figure.savefig(out_dir / "periodization.png")

    log.info("step 6: 最佳匹配窗口")
    # 生成所有可能的300年窗口
    slices, mid_year, slice_labels = sweep_slices(
        start_year=STAGE1,
        window_size=200,
        step_size=20,
        end_year=END,
    )

    data1, data2 = history.merge_with(combined["mean"], split=True)
    max_corr_year, max_corr, r_benchmark_list = sweep_max_corr_year(
        data1=data1,
        data2=data2,
        slices=slices,
        corr_method=cfg.corr_method,
        windows=np.arange(2, 100),
        min_periods=np.repeat(5, 98),
        filter_func=calc_std_deviation,
    )
    # 计算最大相关性改进值
    max_corr_improvment = np.zeros_like(max_corr)
    for i, (corr, r_benchmark) in enumerate(zip(max_corr, r_benchmark_list)):
        max_corr_improvment[i] = (corr - r_benchmark) / r_benchmark
    # 使用函数
    ax = plot_correlation_windows(
        max_corr_year,
        max_corr_improvment,
        mid_year,
        slice_labels,
    )
    ax.figure.savefig(out_dir / "correlation_windows.png")


if __name__ == "__main__":
    _main()
