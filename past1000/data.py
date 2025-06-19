#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import re

import numpy as np
import pandas as pd
from geo_dskit.utils.io import check_tab_sep, find_first_uncommented_line
from geo_dskit.utils.path import filter_files, get_files
from scipy.signal import detrend

from past1000.api.mc import standardize_both
from past1000.api.series import HistoricalRecords


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
