#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import kendalltau, norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm.auto import tqdm

from past1000.constants import LEVELS, LEVELS_PROB, TICK_LABELS
from past1000.data import HistoricalRecords, load_nat_data
from past1000.filters import classify
from past1000.utils.plot import (
    heatmap_with_annot,
    plot_confusion_matrix,
    plot_mismatch_matrix,
)

logger = logging.getLogger(__name__)


class MismatchReport:
    """等级数据之间的匹配情况报告

    提供完整的不匹配分析功能，包括：
    - 混淆矩阵计算
    - 分类准确性评估
    - 蒙特卡洛显著性检验
    - 统计指标计算
    - 错误模式分析
    - 可视化报告生成

    Args:
        pred: 预测/分类数据序列（等级值，如-2,-1,0,1,2）
        true: 真实/参考数据序列（等级值，如-2,-1,0,1,2）
        mc_runs: 蒙特卡洛模拟次数，默认1000
        labels: 等级标签，默认使用TICK_LABELS
        value_series: 原始连续值序列（如自然数据），用于错误模式分析

    Examples:
        基础用法：
        >>> report = MismatchReport(predicted_series, observed_series)
        >>> print(report.get_statistics_summary(as_str=True))
        >>> fig = report.generate_report_figure(save_path="report.png")

        带错误模式分析：
        >>> report = MismatchReport(pred_levels, true_levels, value_series=natural_data)
        >>> error_df = report.analyze_error_patterns()  # 分析错误模式
        >>> print(error_df[['value', 'expect', 'classified', 'exact', 'diff']])
    """

    def __repr__(self) -> str:
        return f"MismatchReport(n_samples={len(self.data)})"

    def __init__(
        self,
        pred: pd.Series,
        true: pd.Series,
        mc_runs: int = 1000,
        labels: list[str] | None = None,
        value_series: pd.Series | None = None,
    ):
        # 输入数据处理
        self.pred = pred
        self.true = true
        self.value_series = value_series  # 原始连续值序列，用于错误模式分析
        self.labels = labels if labels is not None else TICK_LABELS
        self.mc_runs = mc_runs
        self.diff_matrix = None  # 错误分析矩阵，只有在调用 analyze_error_patterns 后才创建

        # 清理数据：移除缺失值
        self._clean_data()
        self._compute_confusion_matrix()

    @property
    def n_samples(self) -> int:
        return len(self.data)

    def _clean_data(self) -> None:
        """清理输入数据，移除缺失值"""
        data = pd.concat([self.pred, self.true], axis=1).dropna()
        if len(data) == 0:
            raise ValueError("清理缺失值后没有有效数据")
        logger.info("有效样本数: %d", len(data))
        logger.debug("预测样本被丢弃了%d个", len(self.pred) - len(data))
        logger.debug("真实样本被丢弃了%d个", len(self.true) - len(data))
        self.pred_clean = data.iloc[:, 0]
        self.true_clean = data.iloc[:, 1]
        self.data = data
        self.n_categories = len(self.labels)

    def _compute_confusion_matrix(self):
        """计算混淆矩阵"""
        cm = confusion_matrix(self.true_clean, self.pred_clean, labels=LEVELS).T
        self.cm_df = pd.DataFrame(cm, index=self.labels, columns=self.labels)
        self.cm_df.index.name = "classified"
        self.cm_df.columns.name = "expect"

    def analyze_error_patterns(
        self, value_series: pd.Series | None = None
    ) -> pd.DataFrame:
        """分析分类错误的模式（公开方法）

        Args:
            value_series: 原始连续值序列。如果未提供，使用初始化时的 value_series

        Returns:
            pd.DataFrame: 包含错误分析的数据框

        Note:
            - value: 原始连续值（如自然数据）
            - expect: 预测等级 (pred_clean)
            - classified: 真实等级 (true_clean)
        """
        if value_series is None:
            value_series = self.value_series

        if value_series is None:
            logger.warning("未提供原始值序列，无法进行错误模式分析")
            return pd.DataFrame()

        # 清理数据：确保索引对齐
        df = pd.concat([value_series, self.data], axis=1).dropna()
        if len(df) == 0:
            logger.warning("清理后无有效数据用于错误模式分析")
            return pd.DataFrame()

        # 重命名列以匹配预期的结构
        df.columns = ["value", "expect", "classified"]
        # 检查分类准确性
        df["exact"] = df["expect"] == df["classified"]
        # 计算同类别内的差异
        df["last"] = self._generate_last_column(df)
        df["diff"] = df["value"] - df["last"]

        # 创建错误分析矩阵
        self.diff_matrix = self._create_misclassification_matrix(df)

        # 计算错误计数矩阵（对角线为0，其他为实际错误数）
        self.false_count_matrix = pd.DataFrame(
            np.where(np.eye(self.n_categories), 0, self.cm_df.values),
            index=LEVELS,
            columns=LEVELS,
        )

        self._run_significance_test()
        return df

    def _generate_last_column(self, df: pd.DataFrame) -> pd.Series:
        """生成上一次同类别的值"""
        df = df.copy()
        df["last"] = np.nan
        for c in df["classified"].unique():
            mask = df["classified"] == c
            df.loc[mask, "last"] = df.loc[mask, "value"].shift(1)
        return df["last"]

    def _create_misclassification_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建错误分类的差异矩阵"""
        misclassified = df[~df["exact"]].copy()

        if len(misclassified) == 0:
            logger.warning("没有发现分类错误")
            return pd.DataFrame(index=LEVELS, columns=LEVELS)

        # 计算各种错误组合的平均差异
        pivot_matrix = pd.pivot_table(
            misclassified,
            values="diff",
            index="expect",
            columns="classified",
            aggfunc="mean",
            fill_value=np.nan,
        )

        # 确保所有等级都存在
        return pivot_matrix.reindex(index=LEVELS, columns=LEVELS, fill_value=np.nan)

    def _run_significance_test(self):
        """运行蒙特卡洛显著性检验"""
        logger.info("开始蒙特卡洛模拟 (n=%d)", self.mc_runs)

        all_diff_matrices = []
        n_samples = self.n_samples

        for _ in tqdm(range(self.mc_runs), desc="MC模拟"):
            # 生成随机数据
            random_data = np.random.normal(0, 1, n_samples)
            random_expect = np.random.choice(LEVELS, size=n_samples, p=LEVELS_PROB)

            random_df = pd.DataFrame(
                {
                    "value": random_data,
                    "expect": random_expect,
                    "classified": classify(random_data),
                }
            )

            # 计算差异矩阵
            random_df["exact"] = random_df["expect"] == random_df["classified"]
            random_df["last"] = self._generate_last_column(random_df)
            random_df["diff"] = random_df["value"] - random_df["last"]

            diff_matrix = self._create_misclassification_matrix(random_df)
            if not diff_matrix.empty:
                all_diff_matrices.append(diff_matrix)

        if not all_diff_matrices:
            logger.warning("蒙特卡洛模拟中未发现错误分类")
            self.mc_mean_matrix = pd.DataFrame(index=LEVELS, columns=LEVELS)
            self.mc_std_matrix = pd.DataFrame(index=LEVELS, columns=LEVELS)
            self.p_value_matrix = pd.DataFrame(index=LEVELS, columns=LEVELS)
            return

        # 计算统计量
        combined_matrices = pd.concat(all_diff_matrices)
        self.mc_mean_matrix = combined_matrices.groupby(level=0).mean()
        self.mc_std_matrix = combined_matrices.groupby(level=0).std()

        # 确保索引一致
        for matrix in [self.mc_mean_matrix, self.mc_std_matrix]:
            matrix = matrix.reindex(index=LEVELS, columns=LEVELS)

        # 计算p值
        z_scores = (self.diff_matrix - self.mc_mean_matrix) / self.mc_std_matrix

        # 处理 NaN 值，将其转换为 float 类型避免 object dtype 问题
        z_scores = z_scores.astype(float)

        def safe_norm_cdf(z):
            """安全的正态分布CDF计算，处理NaN值"""
            if pd.isna(z):
                return np.nan
            return 2 * (1 - norm.cdf(abs(z)))

        self.p_value_matrix = z_scores.apply(lambda z: z.map(safe_norm_cdf))

    @overload
    def get_statistics_summary(
        self, *, as_str: Literal[True], weights: str = "quadratic"
    ) -> str:
        ...

    @overload
    def get_statistics_summary(
        self, *, as_str: Literal[False] = False, weights: str = "quadratic"
    ) -> dict:
        ...

    def get_statistics_summary(
        self,
        *,
        as_str: bool = False,
        weights: str = "quadratic",
    ) -> str | dict:
        """获取统计摘要

        Args:
            as_str: 是否返回字符串格式
            weights: 权重类型，可选 "quadratic", "linear", "none"

        Returns:
            str | dict: 统计摘要，根据 as_str 参数决定返回类型
        """
        kappa = cohen_kappa_score(self.true_clean, self.pred_clean, weights=weights)
        tau, tau_p_value = kendalltau(self.true_clean, self.pred_clean)
        # 准确率
        accuracy = (self.true_clean == self.pred_clean).mean()
        stats = {
            "kappa": kappa,
            "kendall_tau": tau,
            "tau_p_value": tau_p_value,
            "accuracy": accuracy,
            "n_samples": len(self.data),
        }

        if as_str:
            string = f"Kappa: {kappa:.2f}, Kendall's Tau: {tau:.2f}"
            string += "**" if tau_p_value < 0.05 else ""
            return string
        return stats

    def plot_confusion_matrix(
        self, ax: plt.Axes | None = None, title: str | None = None
    ) -> plt.Axes:
        """绘制混淆矩阵"""
        if title is None:
            title = self.get_statistics_summary(as_str=True)

        return plot_confusion_matrix(cm_df=self.cm_df, title=title, ax=ax)

    def plot_mismatch_analysis(self, ax: plt.Axes | None = None) -> plt.Axes:
        """绘制不匹配分析图"""
        if self.diff_matrix is None:
            raise ValueError("需要先调用 analyze_error_patterns() 进行错误分析")
        return plot_mismatch_matrix(
            actual_diff_aligned=self.diff_matrix,
            p_value_matrix=self.p_value_matrix,
            false_count_matrix=self.false_count_matrix,
            ax=ax,
        )

    def plot_heatmap(self, ax: plt.Axes | None = None) -> plt.Axes:
        """绘制差异热力图"""
        if self.diff_matrix is None:
            raise ValueError("需要先调用 analyze_error_patterns() 进行错误分析")
        return heatmap_with_annot(
            matrix=self.diff_matrix, p_value=self.p_value_matrix, ax=ax
        )

    def generate_report_figure(
        self, figsize: tuple = (10, 4), save_path: str | None = None
    ) -> plt.Figure:
        """生成完整的报告图表"""
        if self.diff_matrix is None:
            # 如果没有错误分析，只显示混淆矩阵
            fig, ax = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
            self.plot_confusion_matrix(ax=ax)
        else:
            # 完整报告：混淆矩阵 + 错误分析
            fig, axes = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=figsize,
                gridspec_kw={"width_ratios": [2, 1]},
                tight_layout=True,
            )
            # 混淆矩阵
            self.plot_confusion_matrix(ax=axes[0])
            # 不匹配分析
            self.plot_mismatch_analysis(ax=axes[1])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("报告图表已保存: %s", save_path)

        return fig

    def to_dict(self) -> dict:
        """导出完整分析结果为字典"""
        result = {
            "statistics": self.get_statistics_summary(),
            "confusion_matrix": self.cm_df.to_dict(),
            "mc_runs": self.mc_runs,
        }

        # 只有在进行过错误分析后才包含相关矩阵
        if hasattr(self, "diff_matrix") and self.diff_matrix is not None:
            result["diff_matrix"] = self.diff_matrix.to_dict()
        if hasattr(self, "p_value_matrix"):
            result["p_value_matrix"] = self.p_value_matrix.to_dict()
        if hasattr(self, "false_count_matrix"):
            result["false_count_matrix"] = self.false_count_matrix.to_dict()

        return result


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalRecords]:
    """读取自然和历史数据，以及不确定性"""
    log = logging.getLogger(__name__)
    start_year = cfg.years.start
    end_year = cfg.years.end
    log.info("加载自然数据 [%s-%s]...", start_year, end_year)
    log.debug("数据路径: %s", cfg.ds.noaa)
    log.debug("数据包括: %s", cfg.ds.includes)
    datasets, uncertainties = load_nat_data(
        folder=cfg.ds.noaa,
        includes=cfg.ds.includes,
        index_name="year",
        start_year=start_year,
    )
    log.info("加载历史数据 ...")
    history = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
    )
    return datasets, uncertainties, history


def report_mismatch(
    pred: pd.Series,
    true: pd.Series,
    value_series: pd.Series | None = None,
) -> MismatchReport:
    """创建不匹配报告

    Args:
        pred: 预测数据序列
        true: 真实数据序列

    Returns:
        MismatchReport: 完整的不匹配分析报告

    Note: 这是一个便捷函数，直接返回 MismatchReport 实例
    """
    mismatch_report = MismatchReport(pred=pred, true=true)
    mismatch_report.analyze_error_patterns(value_series=value_series)
    return mismatch_report
