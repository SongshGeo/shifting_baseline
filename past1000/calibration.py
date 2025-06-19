#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import pandas as pd
from hydra import main
from omegaconf import DictConfig

from past1000.api.mc import combine_reconstructions
from past1000.api.series import HistoricalRecords
from past1000.data import load_nat_data
from past1000.filters import classify
from past1000.utils.config import get_output_dir
from past1000.viz.plot import plot_confusion_matrix


@main(config_path="../config", config_name="config", version_base=None)
def calibrate(cfg: DictConfig | None = None) -> pd.DataFrame:
    """读取自然数据并用现代数据进行校准

    Args:
        datasets (pd.DataFrame): 数据
        uncertainties (pd.DataFrame): 不确定性
        method (str): 方法
        name (str): 名称

    Returns:
        pd.DataFrame: 校准后的数据
    """
    assert isinstance(cfg, DictConfig), "cfg must be an instance of DictConfig"
    out_path = get_output_dir()
    slice_ = slice(cfg.how.start_year, cfg.how.end_year)
    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        index_name="year",
        start_year=cfg.how.start_year,
    )
    combined, _ = combine_reconstructions(
        reconstructions=datasets,
        uncertainties=uncertainties,
        standardize=cfg.how.standardize,
    )
    stage4 = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
    ).stage4
    classification = classify(combined["mean"])
    y_pred = classification.loc[slice_]
    # todo 这里可以测试不同的历史数据处理方法
    y_true = stage4.mode(axis=1)[0].loc[slice_]
    ax = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        dropna=True,
    )
    # 使用 Hydra 的输出目录
    ax.figure.savefig(out_path / "calibration_matrix.png")
    return combined


if __name__ == "__main__":
    calibrate()
