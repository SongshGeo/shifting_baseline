#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import re
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files
from hydra import main
from omegaconf import DictConfig

from past1000.api.mc import combine_reconstructions, standardize_both
from past1000.api.series import HistoricalRecords
from past1000.filters import classify
from past1000.viz.plot import plot_confusion_matrix


def load_nat_data(
    folder: str,
    includes: list[str],
    index_name: str = "year",
    start_year: int = 1000,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载数据

    Args:
        folder (str): 数据文件夹
        includes (list[str]): 包含的字符串
        index_name (str): 索引名称
        start_year (int): 开始年份
    Returns:
        datasets (pd.DataFrame): 数据
        uncertainties (pd.DataFrame): 不确定性
    """
    datasets = []
    uncertainties = []
    # 匹配包含included中任意一个字符串的文件
    pattern = f"(?:{'|'.join(map(re.escape, includes))})"
    # 读取树轮重建数据
    paths = get_files(folder, iter_subdirs=True, wildcard="*.txt")
    paths = filter_files(paths, pattern)
    for path in paths:
        lino_1st = find_first_uncommented_line(path)
        sep = r"\t" if check_tab_sep(path) else r"\s+"
        df = pd.read_csv(path, sep=sep, skiprows=lino_1st - 1, index_col=0)
        df.index.name = index_name
        if standardize:
            ser, uncertainty = standardize_both(
                df.iloc[:, 0]
            )  # TODO：这里怎么没有输入uncertainty？
            uncertainty.name = path.stem
        else:
            ser = df.iloc[:, 0]
            uncertainty = pd.Series(
                np.ones(shape=ser.shape) * ser.std(), index=ser.index
            )
        ser.name = path.stem
        datasets.append(ser)
        uncertainties.append(uncertainty)
    datasets = pd.concat(datasets, axis=1).sort_index().loc[start_year:]
    uncertainties = pd.concat(uncertainties, axis=1).sort_index().loc[start_year:]
    return datasets, uncertainties


def mcmc_check(
    datasets: pd.DataFrame,
    uncertainties: pd.DataFrame,
    method: str = "mean",
    name: str = "mean",
) -> pd.DataFrame:
    """MCMC校准

    Args:
        datasets (pd.DataFrame): 数据
        uncertainties (pd.DataFrame): 不确定性
        method (str): 方法
        name (str): 名称

    Returns:
        pd.DataFrame: 校准后的数据
    """
    pass


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
    out_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    slice_ = slice(cfg.how.start_year, cfg.how.end_year)
    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        index_name="year",
        start_year=cfg.how.start_year,
    )
    combined, trace = combine_reconstructions(
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
