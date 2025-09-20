#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm.auto import tqdm

from shifting_baseline.constants import LEVELS, LEVELS_PROB, TICK_LABELS
from shifting_baseline.filters import classify
from shifting_baseline.utils.plot import (
    heatmap_with_annot,
    plot_confusion_matrix,
    plot_mismatch_matrix,
)

if TYPE_CHECKING:
    from geo_dskit.utils.path import PathLike

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
        >>> print(error_df[['value', 'pred', 'true', 'exact', 'diff']])
    """

    def __repr__(self) -> str:
        return f"MismatchReport(n_samples={len(self.data)})"

    def __init__(
        self,
        pred: pd.Series,
        true: pd.Series,
        labels: list[str] | None = None,
        value_series: pd.Series | None = None,
    ):
        # 输入数据处理
        self.pred = pred
        self.true = true
        self.value_series = value_series  # 原始连续值序列，用于错误模式分析
        self.labels = labels if labels is not None else TICK_LABELS
        # 错误分析矩阵，只有在调用 analyze_error_patterns 后才创建
        self._analyzed = False
        self.diff_matrix: pd.DataFrame | None = None
        self.p_value_matrix: pd.DataFrame | None = None

        # 清理数据：移除缺失值
        self._clean_data()
        self._compute_confusion_matrix()

    @property
    def false_count_matrix(self) -> pd.DataFrame:
        """返回false count矩阵"""
        # 计算错误计数矩阵（对角线为0，其他为实际错误数）
        true = np.zeros((self.n_categories, self.n_categories), dtype=bool)
        np.fill_diagonal(true, 1)
        return pd.DataFrame(
            np.where(true, 0, self.cm_df.values),
            index=pd.Series(LEVELS, name="true"),
            columns=pd.Series(LEVELS, name="pred"),
        )

    @property
    def n_samples(self) -> int:
        """有效样本数"""
        return len(self.data)

    @property
    def n_raw_samples(self) -> int:
        """原始样本数"""
        return len(self.pred)

    @property
    def n_mismatches(self) -> int:
        """不匹配数"""
        return self.false_count_matrix.sum().sum()

    @property
    def error_analyzed(self) -> bool:
        """是否已经进行了错误分析"""
        return self._analyzed

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
        self.cm_df.index.name = "true"
        self.cm_df.columns.name = "pred"

    def analyze_error_patterns(
        self,
        value_series: pd.Series | None = None,
        shift: int = 1,
        mc_runs: int = 1000,
    ) -> pd.DataFrame:
        """分析分类错误的模式（公开方法）

        Args:
            value_series: 原始连续值序列。如果未提供，使用初始化时的 value_series

        Returns:
            pd.DataFrame: 包含错误分析的数据框

        Note:
            - value: 原始连续值（如自然数据）
            - pred: 预测等级 (pred_clean)
            - true: 真实等级 (true_clean)
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
        df.columns = ["value", "pred", "true"]
        # 检查分类准确性
        df["exact"] = df["pred"] == df["true"]
        # 计算同类别内的差异
        df["last"] = self._generate_last_column(df, shift=shift)
        df["diff"] = df["value"] - df["last"]

        # 创建错误分析矩阵
        self.diff_matrix = self._create_misclassification_matrix(df)
        self._analyzed = True
        self._run_significance_test(mc_runs=mc_runs, shift=shift)
        return self.diff_matrix

    def _generate_last_column(self, df: pd.DataFrame, shift: int = 1) -> pd.Series:
        """生成上一次同类别的值"""
        df = df.copy()
        df["last"] = np.nan
        for c in df["true"].unique():
            mask = df["true"] == c
            df.loc[mask, "last"] = df.loc[mask, "value"].shift(shift)
        return df["last"]

    def _create_misclassification_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建错误分类的差异矩阵"""
        mistrue = df[~df["exact"]].copy()

        if len(mistrue) == 0:
            logger.warning("没有发现分类错误")
            return pd.DataFrame(index=LEVELS, columns=LEVELS)

        # 计算各种错误组合的平均差异
        pivot_matrix = pd.pivot_table(
            mistrue,
            values="diff",
            index="pred",
            columns="true",
            aggfunc="mean",
            fill_value=np.nan,
        )

        # 确保所有等级都存在
        return pivot_matrix.reindex(index=LEVELS, columns=LEVELS, fill_value=np.nan)

    def _run_significance_test(self, mc_runs: int = 1000, shift: int = 1):
        """运行蒙特卡洛显著性检验"""
        logger.info("开始蒙特卡洛模拟 (n=%d)", mc_runs)

        all_diff_matrices = []
        n_samples = self.n_raw_samples

        for _ in tqdm(range(mc_runs), desc="MC模拟"):
            # 生成随机数据
            random_data = np.random.normal(0, 1, n_samples)
            random_pred = np.random.choice(LEVELS, size=n_samples, p=LEVELS_PROB)

            random_df = pd.DataFrame(
                {
                    "value": random_data,
                    "pred": random_pred,
                    "true": classify(random_data),
                }
            )

            # 计算差异矩阵
            random_df["exact"] = random_df["pred"] == random_df["true"]
            random_df["last"] = self._generate_last_column(random_df, shift=shift)
            random_df["diff"] = random_df["value"] - random_df["last"]

            diff_matrix = self._create_misclassification_matrix(random_df)
            if not diff_matrix.empty:
                all_diff_matrices.append(diff_matrix)

        if not all_diff_matrices:
            logger.warning("蒙特卡洛模拟中未发现错误分类")
            return

        # 计算统计量
        combined_matrices = pd.concat(all_diff_matrices)
        mc_mean_matrix = combined_matrices.groupby(level=0).mean()
        mc_std_matrix = combined_matrices.groupby(level=0).std()

        # 确保索引一致
        for matrix in [mc_mean_matrix, mc_std_matrix]:
            matrix = matrix.reindex(index=LEVELS, columns=LEVELS)

        # 计算p值
        z_scores = (self.diff_matrix - mc_mean_matrix) / mc_std_matrix

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
            "n_samples": self.n_samples,
            "n_raw_samples": self.n_raw_samples,
            "n_mismatches": self.n_mismatches,
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
        ax = plot_confusion_matrix(cm_df=self.cm_df, title=title, ax=ax)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        return ax

    def plot_mismatch_analysis(self, ax: plt.Axes | None = None) -> plt.Axes:
        """绘制不匹配分析图"""
        if self.diff_matrix is None:
            raise ValueError("需要先调用 analyze_error_patterns() 进行错误分析")
        if self.n_mismatches == 0:
            logger.warning("没有发现分类错误，无法绘制不匹配分析图")
            if ax is None:
                fig, ax = plt.subplots(figsize=(2, 3.5))
            return ax
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
        self,
        figsize: tuple = (5, 3),
        save_path: PathLike | None = None,
        **kwargs,
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
            fig.tight_layout(rect=[0, 0, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("报告图表已保存: %s", save_path)

        return fig

    def to_dict(self) -> dict:
        """导出完整分析结果为字典"""
        result = {
            "statistics": self.get_statistics_summary(),
            "confusion_matrix": self.cm_df.to_dict(),
        }

        # 只有在进行过错误分析后才包含相关矩阵
        result["false_count_matrix"] = self.false_count_matrix.to_dict()
        if self.error_analyzed:
            assert self.diff_matrix is not None
            assert self.p_value_matrix is not None
            result["diff_matrix"] = self.diff_matrix.to_dict()
            result["p_value_matrix"] = self.p_value_matrix.to_dict()

        return result
