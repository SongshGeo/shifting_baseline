#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig
from scipy.stats import kendalltau, norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm.auto import tqdm

from past1000.constants import LEVELS, LEVELS_PROB, TICK_LABELS
from past1000.data import HistoricalRecords, load_nat_data
from past1000.filters import classify
from past1000.mc import combine_reconstructions
from past1000.utils.config import get_output_dir
from past1000.utils.plot import (
    heatmap_with_annot,
    plot_confusion_matrix,
    plot_mismatch_matrix,
)

logger = logging.getLogger(__name__)


def generate_last_column_by_classified(
    df: pd.DataFrame,
    value_col: str = "value",
    expect_col: str = "expect",
    classify_col: str = "classified",
) -> pd.Series:
    # TODO: 这里需要核对一下，优化掉没用到的参数
    """
    生成 last 列：上一次 classified 等级与当前相同的 expect 的 value
    """
    df["last"] = np.nan
    # 按 classified 分组，对每组的 expect 的 value shift
    for c in df[classify_col].unique():
        mask = df[classify_col] == c
        df.loc[mask, "last"] = df.loc[mask, value_col].shift(1)
    return df["last"]


def analyze_misclassification_pivot(
    df: pd.DataFrame,
    expect_col: str = "expect",
    classified_col: str = "classified",
    diff_col: str = "diff",
    default_val: float = np.nan,
) -> pd.DataFrame:
    """
    使用 pandas pivot_table 分析被错判组合的平均 diff
    """
    # 只选择被错判的行
    misclassified = df[~df["exact"]].copy()

    if len(misclassified) == 0:
        print("没有找到错判的数据")
        return pd.DataFrame()

    # 使用 pivot_table 创建矩阵
    pivot_matrix = pd.pivot_table(
        misclassified,
        values=diff_col,
        index=expect_col,
        columns=classified_col,
        aggfunc="mean",  # 计算平均值
        fill_value=default_val,  # 先填充为 NaN
    )

    # 确保所有可能的组合都存在，并用默认值填充
    all_levels = sorted(df[expect_col].unique())
    all_columns = sorted(df[classified_col].unique())

    # 重新索引确保所有组合都存在
    pivot_matrix = pivot_matrix.reindex(
        index=all_levels, columns=all_columns, fill_value=default_val
    )

    return pivot_matrix


def check_estimation(
    data: pd.DataFrame,
    value_col: str = "value",
    expect_col: str = "expect",
    classify_col: str = "classified",
) -> pd.DataFrame:
    """找到 h_data 里在 classify 中被归类为同一个级别的值，对比相同索引"""
    data["exact"] = data[expect_col] == data[classify_col]
    data["last"] = generate_last_column_by_classified(
        data, value_col, expect_col, classify_col
    )
    data["diff"] = data[value_col] - data["last"]
    return data


def analyze_mismatch(
    natural_data: pd.Series,
    historical_data: pd.Series,
    cm_df: pd.DataFrame | None = None,
    plot: None | Literal["heatmap", "flowmap"] = None,
    ax: None | plt.Axes = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """数据校准分析

    Args:
        natural_data: pd.DataFrame
            自然数据
        historical_data: pd.DataFrame
            历史数据
        plot: None | Literal["heatmap", "flowmap"]
            绘图类型
        ax: None | plt.Axes
            绘图轴

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            actual_diff 矩阵、p 值矩阵和false 计数矩阵
    """
    if cm_df is None:
        cm_df = notna_confusion_matrix(
            y_true=natural_data,
            y_pred=historical_data,
            labels=TICK_LABELS,
        )
    df = pd.DataFrame(
        {
            "value": natural_data,
            "expect": historical_data,
            "classified": classify(natural_data),
        }
    ).dropna()
    checked_df = check_estimation(df)
    diff_m = analyze_misclassification_pivot(checked_df)

    # 对角线为 0，因为是判断正确的位置，其他为实际差异
    false_count_m = pd.DataFrame(
        np.where(np.eye(5), 0, cm_df.values),
        index=LEVELS,
        columns=LEVELS,
    )
    # 计算平均 diff 矩阵、标准差矩阵和 p 值矩阵
    _, _, p_value_m = run_mc_simulation(
        actual_diff_matrix=diff_m,
        n_runs=100,
        n_samples=len(checked_df),
    )
    if plot == "flowmap":
        plot_mismatch_matrix(
            actual_diff_aligned=diff_m,
            p_value_matrix=p_value_m,
            false_count_matrix=false_count_m,
            ax=ax,
        )
    elif plot == "heatmap":
        heatmap_with_annot(
            matrix=diff_m,
            p_value=p_value_m,
            ax=ax,
        )
    else:
        raise ValueError(f"Invalid plot type: {plot}")
    return diff_m, p_value_m, false_count_m


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
    # 获取切片
    slice_ = slice(cfg.how.start_year, cfg.how.end_year)
    # 1. 读取自然数据和不确定性
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
    )  # 用 Bayesian 方法合并自然数据和不确定性
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
    )  # 读取历史数据
    # 将历史数据转换为 series
    history.to_series(
        inplace=True,
        interpolate=None,
        name="historical_mean",
        how="mode",
    )
    # 处理缺失值
    natural, historical = dropna_series(
        combined["mean"].loc[slice_],
        history.data.loc[slice_],
    )
    cm_df = notna_confusion_matrix(
        y_true=natural,
        y_pred=historical,
        labels=TICK_LABELS,
    )
    # ======== 绘图 ========
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(5, 3),
        gridspec_kw={"width_ratios": [2, 1]},
        tight_layout=True,
    )
    # 绘制混淆矩阵热力图
    title = mismatch_stats(
        y_true=natural,
        y_pred=historical,
        as_str=True,
    )
    plot_confusion_matrix(cm_df=cm_df, title=title, ax=ax1)
    # 绘制不匹配情况前后对比图
    analyze_mismatch(
        natural_data=combined["mean"].loc[slice_],
        historical_data=history.data.loc[slice_],
        cm_df=cm_df,
        plot="flowmap",
        ax=ax2,
    )
    outpath = get_output_dir()
    fig.savefig(outpath / f"calibration_{slice_.start}-{slice_.stop}.png")
    return combined


def dropna_series(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    处理缺失值
    """
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        combined = pd.concat([y_true, y_pred], axis=1).dropna(axis=0)
        y_true = combined.iloc[:, 0]
        y_pred = combined.iloc[:, 1]
    # 转换为 numpy 数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # 处理缺失值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return y_true, y_pred


def notna_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    dropna: bool = True,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    获取混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred).T
    if labels is None:
        return cm
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df


def mismatch_stats(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    kappa_weights: str = "quadratic",
    as_str: bool = False,
) -> tuple[float, float, float] | str:
    """
    计算混淆矩阵的统计信息
    """
    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        weights=kappa_weights,
    )

    # 3. 计算 Kendall's Tau
    tau, p_value = kendalltau(y_true, y_pred)
    if as_str:
        string = f"Kappa: {kappa:.2f}, Kendall's Tau: {tau:.2f}"
        string += "**" if p_value < 0.05 else ""
        return string
    return kappa, tau, p_value


def run_mc_simulation(
    actual_diff_matrix: pd.DataFrame | np.ndarray,
    n_runs: int = 1000,
    n_samples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    运行蒙特卡洛模拟，返回平均 diff 矩阵和标准差矩阵。

    Parameters:
    -----------
    n_runs : int
        模拟运行的次数。
    n_samples : int
        每次模拟生成的样本数量。

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        返回 (平均diff矩阵, 标准差diff矩阵)。
    """
    all_diff_matrices = []

    for _ in tqdm(range(n_runs)):
        # 1. 生成随机数据
        random_data = np.random.normal(0, 1, n_samples)
        random_expect = np.random.choice(LEVELS, size=n_samples, p=LEVELS_PROB)
        random_df = pd.DataFrame(
            {
                "value": random_data,
                "expect": random_expect,
                "classified": classify(random_data),
            }
        )

        # 2. 计算并获取 diff 矩阵
        checked_df = check_estimation(random_df)
        diff_matrix = analyze_misclassification_pivot(checked_df)
        if not diff_matrix.empty:
            all_diff_matrices.append(diff_matrix)

    if not all_diff_matrices:
        print("Warning: No misclassifications found in any simulation run.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 3. 将所有矩阵堆叠并计算均值和标准差
    # 使用 concat 和 groupby 可以优雅地处理每次模拟中维度不一致的问题
    combined_matrices = pd.concat(all_diff_matrices)

    # --- 这里是修正的部分 ---
    # 只按 level=0 (即 'expect' 索引) 分组，并且不再需要 unstack()
    mean_matrix = combined_matrices.groupby(level=0).mean()
    std_matrix = combined_matrices.groupby(level=0).std()
    # --- 修正结束 ---

    # 为了保持一致的行列顺序
    all_levels = sorted(list(set(mean_matrix.index) | set(mean_matrix.columns)))
    mean_matrix = mean_matrix.reindex(index=all_levels, columns=all_levels)
    std_matrix = std_matrix.reindex(index=all_levels, columns=all_levels)

    # 将 Z-score 转换为 P 值 (双尾)
    z_score_matrix = (actual_diff_matrix - mean_matrix) / std_matrix
    p_value_matrix = z_score_matrix.apply(lambda z: 2 * (1 - norm.cdf(abs(z))))
    return mean_matrix, std_matrix, p_value_matrix


if __name__ == "__main__":
    calibrate()
