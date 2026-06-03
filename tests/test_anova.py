#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""utils/anova.py 的 smoke 测试。

comprehensive_anova_analysis 是 reports/abm.ipynb 用的笔记本级分析工具，逻辑稳定。
测试合成数据能跑通、类型自动识别正确、注入的效应被判显著、返回结构齐全。
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from shifting_baseline.utils.anova import comprehensive_anova_analysis

matplotlib.use("Agg")  # 无显示后端，避免测试弹窗


@pytest.fixture
def anova_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "group": rng.choice(["a", "b", "c"], n),  # 字符串 → 分类
            "dose": rng.choice([1, 2], n),  # 唯一值少 → 分类
            "age": rng.normal(40, 10, n),  # 连续
            "y": rng.normal(0, 1, n),
        }
    )
    df.loc[df["group"] == "a", "y"] += 1.5  # 注入 group 效应
    return df


class TestComprehensiveAnovaSmoke:
    def _run(self, df):
        return comprehensive_anova_analysis(
            df, target_col="y", verbose=False, plot=False
        )

    def test_runs_and_returns_keys(self, anova_df):
        res = self._run(anova_df)
        for key in (
            "single_factor",
            "summary_table",
            "best_model",
            "best_model_name",
            "categorical_cols",
            "continuous_cols",
            "main_effects",
        ):
            assert key in res

    def test_auto_type_detection(self, anova_df):
        res = self._run(anova_df)
        assert set(res["categorical_cols"]) == {"group", "dose"}
        assert res["continuous_cols"] == ["age"]

    def test_detects_injected_effect(self, anova_df):
        res = self._run(anova_df)
        group_res = res["single_factor"]["group"]
        assert bool(group_res["significant"]) is True
        assert group_res["p_value"] < 0.05

    def test_summary_table_one_row_per_predictor(self, anova_df):
        res = self._run(anova_df)
        # group, dose, age → 3 行
        assert res["summary_table"].shape[0] == 3

    def test_no_predictors_is_handled(self):
        # 只有目标列，没有预测变量：不应崩溃
        df = pd.DataFrame({"y": np.random.default_rng(1).normal(0, 1, 50)})
        res = comprehensive_anova_analysis(
            df, target_col="y", verbose=False, plot=False
        )
        assert "best_model_name" in res
