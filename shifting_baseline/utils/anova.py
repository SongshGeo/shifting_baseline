#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from shifting_baseline.utils.log import get_logger

# 使用主logger，避免重复设置
log = get_logger()


def _identify_variable_types(
    df: pd.DataFrame, target_col: str, predictor_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    自动识别变量类型

    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    target_col : str
        因变量列名
    predictor_cols : List[str], optional
        预测变量列名列表

    Returns:
    --------
    Tuple[List[str], List[str]]
        分类变量列表和连续变量列表
    """
    if predictor_cols is None:
        predictor_cols = [col for col in df.columns if col != target_col]

    categorical_cols = []
    continuous_cols = []

    for col in predictor_cols:
        if df[col].dtype in ["object", "category"]:
            categorical_cols.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            unique_count = df[col].nunique()
            if unique_count <= 10:  # 少于等于10个唯一值视为分类
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            categorical_cols.append(col)  # 默认为分类

    return categorical_cols, continuous_cols


def _preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    continuous_cols: List[str],
    predictor_cols: List[str],
) -> pd.DataFrame:
    """
    数据预处理

    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    target_col : str
        因变量列名
    categorical_cols : List[str]
        分类变量列表
    continuous_cols : List[str]
        连续变量列表
    predictor_cols : List[str]
        预测变量列表

    Returns:
    --------
    pd.DataFrame
        清理后的数据框
    """
    df_clean = df.copy()

    # 确保分类变量为字符串类型
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

    # 确保数值变量为数值类型
    numeric_cols = continuous_cols + [target_col]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 删除缺失值
    df_clean = df_clean.dropna(subset=[target_col] + predictor_cols)

    return df_clean


def _single_factor_analysis(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    significance_level: float,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    单因素方差分析

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    target_col : str
        因变量列名
    categorical_cols : List[str]
        分类变量列表
    significance_level : float
        显著性水平
    verbose : bool
        是否打印详细结果

    Returns:
    --------
    Dict[str, Dict]
        单因素分析结果
    """
    single_factor_results = {}

    for col in categorical_cols:
        if col in df.columns:
            try:
                formula = f"{target_col} ~ C({col})"
                model = ols(formula, data=df).fit()
                anova = anova_lm(model, typ=2)

                f_stat = anova.loc[f"C({col})", "F"]
                p_val = anova.loc[f"C({col})", "PR(>F)"]
                r_squared = model.rsquared

                single_factor_results[col] = {
                    "f_statistic": f_stat,
                    "p_value": p_val,
                    "r_squared": r_squared,
                    "significant": p_val < significance_level,
                    "model": model,
                    "anova": anova,
                }

                if verbose:
                    significance = (
                        "***"
                        if p_val < 0.001
                        else "**"
                        if p_val < 0.01
                        else "*"
                        if p_val < 0.05
                        else ""
                    )
                    log.info(f"{col}: F={f_stat:.4f}, p={p_val:.4f} {significance}")
                    log.info(
                        f"  R²={r_squared:.4f}, 显著: {'是' if p_val < significance_level else '否'}"
                    )

            except Exception as e:
                if verbose:
                    log.error(f"{col}: 分析失败 - {str(e)}")
                single_factor_results[col] = {"error": str(e)}

    return single_factor_results


def _build_formula_terms(
    df: pd.DataFrame, categorical_cols: List[str], continuous_cols: List[str]
) -> List[str]:
    """构建公式项"""
    formula_terms = []
    for col in categorical_cols:
        if col in df.columns:
            formula_terms.append(f"C({col})")
    for col in continuous_cols:
        if col in df.columns:
            formula_terms.append(col)
    return formula_terms


def _build_interaction_terms(
    df: pd.DataFrame, categorical_cols: List[str]
) -> List[str]:
    """构建交互项"""
    interaction_terms = []
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1 :]:
            if col1 in df.columns and col2 in df.columns:
                interaction_terms.append(f"C({col1}):C({col2})")
    return interaction_terms


def _select_best_model(results: Dict[str, Any]) -> Tuple[Any, str]:
    """选择最佳模型"""
    if "interaction_effects" in results:
        if results["interaction_effects"]["aic"] < results["main_effects"]["aic"]:
            return results["interaction_effects"]["model"], "interaction_effects"
        else:
            return results["main_effects"]["model"], "main_effects"
    else:
        return results["main_effects"]["model"], "main_effects"


def _multi_factor_analysis(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    continuous_cols: List[str],
    include_interactions: bool,
    verbose: bool = True,
) -> Dict[str, Union[Dict[str, Any], str]]:
    """
    多因素方差分析

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    target_col : str
        因变量列名
    categorical_cols : List[str]
        分类变量列表
    continuous_cols : List[str]
        连续变量列表
    include_interactions : bool
        是否包含交互效应
    verbose : bool
        是否打印详细结果

    Returns:
    --------
    Dict[str, Union[Dict, str]]
        多因素分析结果
    """
    try:
        # 构建公式
        formula_terms = _build_formula_terms(df, categorical_cols, continuous_cols)

        if not formula_terms:
            if verbose:
                log.warning("没有可用的预测变量进行多因素分析")
            return {"multi_factor_error": "没有可用的预测变量"}

        main_effects_formula = f"{target_col} ~ " + " + ".join(formula_terms)

        # 仅主要效应模型
        model_main = ols(main_effects_formula, data=df).fit()
        anova_main = anova_lm(model_main, typ=2)

        results: Dict[str, Dict[str, Any] | str] = {
            "main_effects": {
                "model": model_main,
                "anova": anova_main,
                "r_squared": model_main.rsquared,
                "adj_r_squared": model_main.rsquared_adj,
                "aic": model_main.aic,
                "bic": model_main.bic,
            }
        }

        # 包含交互效应的模型
        if include_interactions and len(categorical_cols) >= 2:
            interaction_terms = _build_interaction_terms(df, categorical_cols)

            if interaction_terms:
                interaction_formula = (
                    main_effects_formula + " + " + " + ".join(interaction_terms)
                )
                model_interaction = ols(interaction_formula, data=df).fit()
                anova_interaction = anova_lm(model_interaction, typ=2)

                results["interaction_effects"] = {
                    "model": model_interaction,
                    "anova": anova_interaction,
                    "r_squared": model_interaction.rsquared,
                    "adj_r_squared": model_interaction.rsquared_adj,
                    "aic": model_interaction.aic,
                    "bic": model_interaction.bic,
                }

        # 选择最佳模型
        best_model, best_model_name = _select_best_model(results)
        results["best_model"] = best_model
        results["best_model_name"] = best_model_name

        if verbose:
            log.info("多因素方差分析:")
            main_effects = results["main_effects"]
            if isinstance(main_effects, dict):
                log.info(
                    f"仅主要效应模型: R²={main_effects['r_squared']:.2f}, AIC={main_effects['aic']:.2f}"
                )
            if "interaction_effects" in results:
                interaction_effects = results["interaction_effects"]
                if isinstance(interaction_effects, dict):
                    log.info(
                        f"包含交互效应模型: R²={interaction_effects['r_squared']:.2f}, AIC={interaction_effects['aic']:.2f}"
                    )
            log.info(f"推荐模型: {best_model_name}")

        return results

    except Exception as e:
        if verbose:
            log.error(f"多因素方差分析失败: {str(e)}")
        return {"multi_factor_error": str(e)}


def _generate_summary_table(
    categorical_cols: List[str],
    continuous_cols: List[str],
    single_factor_results: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    target_col: str,
    significance_level: float,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    生成统计摘要表

    Parameters:
    -----------
    categorical_cols : List[str]
        分类变量列表
    continuous_cols : List[str]
        连续变量列表
    single_factor_results : Dict[str, Dict]
        单因素分析结果
    df : pd.DataFrame
        数据框
    target_col : str
        因变量列名
    significance_level : float
        显著性水平
    verbose : bool
        是否打印详细结果

    Returns:
    --------
    pd.DataFrame
        统计摘要表
    """
    summary_data = []

    # 处理分类变量
    for col in categorical_cols:
        if col in single_factor_results and "error" not in single_factor_results[col]:
            result = single_factor_results[col]
            summary_data.append(
                {
                    "变量": col,
                    "类型": "分类",
                    "F统计量": f"{result['f_statistic']:.4f}",
                    "p值": f"{result['p_value']:.4f}",
                    "R²": f"{result['r_squared']:.4f}",
                    "显著性": "是" if result["significant"] else "否",
                    "显著性标记": (
                        "***"
                        if result["p_value"] < 0.001
                        else "**"
                        if result["p_value"] < 0.01
                        else "*"
                        if result["p_value"] < 0.05
                        else ""
                    ),
                }
            )

    # 处理连续变量
    for col in continuous_cols:
        if col in df.columns:
            try:
                corr_coef, corr_p = stats.pearsonr(df[col], df[target_col])
                summary_data.append(
                    {
                        "变量": col,
                        "类型": "连续",
                        "F统计量": "-",
                        "p值": f"{corr_p:.4f}",
                        "R²": f"{corr_coef**2:.4f}",
                        "显著性": "是" if corr_p < significance_level else "否",
                        "显著性标记": (
                            "***"
                            if corr_p < 0.001
                            else "**"
                            if corr_p < 0.01
                            else "*"
                            if corr_p < 0.05
                            else ""
                        ),
                    }
                )
            except Exception as e:
                if verbose:
                    log.error(f"连续变量 {col} 分析失败: {str(e)}")

    summary_df = pd.DataFrame(summary_data)

    if verbose:
        log.info("统计摘要表:")
        log.info(f"\n{summary_df.to_string(index=False)}")

    return summary_df


def _create_visualization(
    df: pd.DataFrame, target_col: str, categorical_cols: List[str], verbose: bool = True
) -> None:
    """
    创建可视化图表

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    target_col : str
        因变量列名
    categorical_cols : List[str]
        分类变量列表
    verbose : bool
        是否打印详细结果
    """
    if not categorical_cols:
        return

    try:
        n_cat = len(categorical_cols)
        n_cols = min(3, n_cat)
        n_rows = (n_cat + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # 处理axes的维度问题
        if n_cat == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                if n_rows == 1 and n_cols == 1:
                    ax = axes[0]
                elif n_rows == 1:
                    ax = axes[0, i]
                elif n_cols == 1:
                    ax = axes[i, 0]
                else:
                    ax = axes[i // n_cols, i % n_cols]

                sns.boxplot(data=df, x=col, y=target_col, ax=ax)
                ax.set_title(f"{col} -> {target_col}")
                ax.tick_params(axis="x", rotation=45)

        # 隐藏多余的子图
        for i in range(n_cat, n_rows * n_cols):
            if n_rows == 1 and n_cols == 1:
                pass  # 只有一个子图，不需要隐藏
            elif n_rows == 1:
                axes[0, i].set_visible(False)
            elif n_cols == 1:
                axes[i, 0].set_visible(False)
            else:
                axes[i // n_cols, i % n_cols].set_visible(False)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        if verbose:
            log.error(f"可视化失败: {str(e)}")


def comprehensive_anova_analysis(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    include_interactions: bool = True,
    significance_level: float = 0.05,
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    综合方差分析函数

    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    target_col : str
        因变量列名
    predictor_cols : List[str], optional
        所有预测变量列名列表，如果提供则自动识别类型
    categorical_cols : List[str], optional
        分类变量列名列表
    continuous_cols : List[str], optional
        连续变量列名列表
    include_interactions : bool, default True
        是否包含交互效应
    significance_level : float, default 0.05
        显著性水平
    verbose : bool, default True
        是否打印详细结果
    plot : bool, default True
        是否生成图表

    Returns:
    --------
    Dict[str, Union[pd.DataFrame, Dict, str, None]]
        包含所有分析结果的字典
    """
    if verbose:
        log.info("=" * 80)
        log.info("综合方差分析")
        log.info("=" * 80)

    # 自动识别变量类型
    if categorical_cols is None and continuous_cols is None:
        categorical_cols, continuous_cols = _identify_variable_types(
            df, target_col, predictor_cols
        )
    else:
        # 确保列表不为None
        if categorical_cols is None:
            categorical_cols = []
        if continuous_cols is None:
            continuous_cols = []

    if predictor_cols is None:
        predictor_cols = [col for col in df.columns if col != target_col]

    # 数据预处理
    df_clean = _preprocess_data(
        df, target_col, categorical_cols, continuous_cols, predictor_cols
    )

    if verbose:
        log.info("数据概况:")
        log.info(f"样本数: {len(df_clean)}")
        log.info(f"因变量: {target_col}")
        log.info(f"分类变量: {categorical_cols}")
        log.info(f"连续变量: {continuous_cols}")
        log.info(f"缺失值: {df_clean.isnull().sum().sum()}")

    # 单因素方差分析
    single_factor_results = _single_factor_analysis(
        df_clean, target_col, categorical_cols, significance_level, verbose
    )

    # 多因素方差分析
    multi_factor_results = _multi_factor_analysis(
        df_clean,
        target_col,
        categorical_cols,
        continuous_cols,
        include_interactions,
        verbose,
    )

    # 生成统计摘要表
    summary_table = _generate_summary_table(
        categorical_cols,
        continuous_cols,
        single_factor_results,
        df_clean,
        target_col,
        significance_level,
        verbose,
    )

    # 创建可视化
    if plot:
        _create_visualization(df_clean, target_col, categorical_cols, verbose)

    # 组装结果
    results = {
        "data": df_clean,
        "target_col": target_col,
        "categorical_cols": categorical_cols,
        "continuous_cols": continuous_cols,
        "significance_level": significance_level,
        "single_factor": single_factor_results,
        "summary_table": summary_table,
    }

    # 添加多因素分析结果
    results.update(multi_factor_results)

    # 确保best_model总是存在
    if "best_model" not in results:
        if "main_effects" in results:
            results["best_model"] = results["main_effects"]["model"]
            results["best_model_name"] = "main_effects"
        else:
            # 如果没有多因素模型，使用第一个单因素模型作为默认
            first_single_factor = None
            for col, result in single_factor_results.items():
                if "error" not in result:
                    first_single_factor = result["model"]
                    break

            if first_single_factor is not None:
                results["best_model"] = first_single_factor
                results["best_model_name"] = "single_factor"
            else:
                results["best_model"] = None
                results["best_model_name"] = "none"

    return results


def quick_anova_summary(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    快速方差分析函数 - 简化版本

    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    target_col : str
        因变量列名
    categorical_cols : List[str], optional
        分类变量列名列表，如果为None则自动识别
    plot : bool, default True
        是否生成图表

    Returns:
    --------
    pd.DataFrame
        统计摘要表
    """
    results = comprehensive_anova_analysis(
        df=df,
        target_col=target_col,
        categorical_cols=categorical_cols,
        continuous_cols=[],
        include_interactions=True,
        significance_level=0.05,
        verbose=True,
        plot=plot,
    )

    return results["summary_table"]
