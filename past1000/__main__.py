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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from hydra import main
from omegaconf import DictConfig

from past1000.calibration import MismatchReport
from past1000.compare import experiment_corr_2d, sweep_max_corr_year, sweep_slices
from past1000.constants import END, STAGE1
from past1000.data import HistoricalRecords, load_data
from past1000.filters import calc_std_deviation, classify
from past1000.mc import combine_reconstructions
from past1000.process import batch_process_recon_data
from past1000.utils.config import format_by_config, get_output_dir
from past1000.utils.plot import plot_correlation_windows

if TYPE_CHECKING:
    from geo_dskit.utils.path import PathLike

    from past1000.utils.types import Stages

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
    out_dir = get_output_dir()
    log.info("实验开始，配置文件请参看 %s", out_dir / ".hydra/config.yaml")
    log.info("Step 1: 加载数据 ...")
    datasets, uncertainties, _ = load_data(cfg)
    log.info("Step 2: 比较每个树轮数据")
    # TODO 需要添加一个函数，用于比较每个树轮数据
    log.info("Step 3: 整合树轮数据")
    combined, _ = combine_reconstructions(
        reconstructions=datasets,
        uncertainties=uncertainties,
        standardize=True,
    )
    log.info("Step 4: 加载历史数据")
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
    )
    log.info("Step 5: 分时期对比历史数据和整合树轮数据")
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
    axs = axs.flatten()
    stages: list[Stages] = [2, 3, "2-3", 4]
    for i, stage in enumerate(stages):
        slice_now = history.get_time_slice(stage)
        ax = axs[i]
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
        log.info("Stage %s 处理中...", stage)
        mismatch_report.analyze_error_patterns()
        log.debug(mismatch_report.get_statistics_summary(as_str=True))
        mismatch_report.generate_report_figure(
            save_path=out_dir / f"mismatch_{stage}.png"
        )
        _, r_benchmark, ax = experiment_corr_2d(
            data1=his,
            data2=nat,
            time_slice=slice_now,
            corr_method=cfg.corr_method,
            filter_func=calc_std_deviation,
            filter_side=cfg.filter_side,
            ax=ax,
            penalty=False,
            n_diff_w=5,
            std_offset=0.1,
        )
        ax.set_title(
            f"{slice_now.start}-{slice_now.stop} AD. $Tau={r_benchmark:.3f}$",
            fontsize=9,
        )
        ax.locator_params(axis="x", nbins=9)  # x轴最多9个主刻度
        ax.locator_params(axis="y", nbins=4)  # y轴最多9个主刻度
        ax.tick_params(axis="both", rotation=0)
        log.info("Stage %s 处理完成", stage)
    fig.savefig(out_dir / "periodization.png")

    log.info("step 6: 最佳匹配窗口")
    # 生成所有可能的300年窗口
    slices, mid_year, slice_labels = sweep_slices(
        start_year=STAGE1,
        window_size=200,
        step_size=20,
        end_year=END,
    )

    data1, data2 = history.merge_with(combined["mean"], split=True)
    max_corr_year, max_corr = sweep_max_corr_year(
        data1=data1,
        data2=data2,
        slices=slices,
        corr_method=cfg.corr_method,
        windows=np.arange(2, 100),
        min_periods=np.repeat(5, 98),
        filter_func=calc_std_deviation,
    )

    # 使用函数
    ax4 = plot_correlation_windows(
        max_corr_year,
        max_corr,
        mid_year,
        slice_labels,
    )
    ax4.figure.savefig(out_dir / "correlation_windows.png")

    ax4.axvspan(1636, 1720, color="gray", alpha=0.2, label="Dynasty Transition")
    ax4.legend()


if __name__ == "__main__":
    _main()
