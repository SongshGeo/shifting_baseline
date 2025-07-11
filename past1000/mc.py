#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler


def standardize_data(
    data: pd.DataFrame | pd.Series,
    method: str = "standard",
) -> tuple[pd.DataFrame | pd.Series, StandardScaler]:
    r"""使用 sklearn 标准化数据
    Standardize features by removing the mean and scaling to unit variance.

    $$ z = (x - \mu) / \sigma $$

    Args:
        data: 原始数据
        method: 'standard'

    Returns:
        tuple: (标准化后的数据, scaler对象)
    """
    # 选择标准化方法
    if method == "standard":
        scaler = StandardScaler()
    else:
        raise NotImplementedError("Only 'standard' method is supported now.")

    # 处理 DataFrame
    if isinstance(data, pd.DataFrame):
        std_data = pd.DataFrame(
            scaler.fit_transform(data), columns=data.columns, index=data.index
        )
    # 处理 Series
    else:
        std_data = pd.Series(
            scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index
        )
    return std_data, scaler


def standardize_both(
    data: pd.DataFrame | pd.Series,
    uncertainties: pd.DataFrame | pd.Series | None = None,
    method: str = "standard",
    window_size: int = 10,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """标准化数据和不确定性，自动处理缺失的不确定性

    Args:
        data: 原始数据
        uncertainties: 原始不确定性，如果为 None 则自动计算
        method: 标准化方法，'standard' 或 'minmax'
        window_size: 计算不确定性时的移动窗口大小

    Returns:
        tuple: (标准化后的数据, 标准化后的不确定性)
    """
    # 如果没有提供不确定性，使用移动窗口计算
    if uncertainties is None:
        uncertainties = compute_uncertainties(data, window_size)
    # 1. 标准化数据
    std_data, scaler = standardize_data(data, method=method)
    # 2. 标准化不确定性
    std_uncertainties = uncertainties / scaler.scale_
    return std_data, std_uncertainties


def compute_uncertainties(
    reconstructions: pd.DataFrame | pd.Series,
    window_size: int = 10,
    min_ratio: float = 0.2,  # 最小不确定性比例
    max_ratio: float = 2.0,  # 最大不确定性比例
) -> pd.DataFrame | pd.Series:
    """使用移动窗口计算时间序列的不确定性，并确保在合理范围内

    Args:
        reconstructions: 原始重建数据
        window_size: 移动窗口大小
        min_ratio: 最小不确定性与平均不确定性的比例
        max_ratio: 最大不确定性与平均不确定性的比例

    Returns:
        与输入数据相同格式的不确定性估计
    """
    uncertainties = (
        reconstructions.rolling(window_size, center=True).std().bfill().ffill()
    )

    # 确保不确定性在合理范围内
    if isinstance(uncertainties, pd.DataFrame):
        for col in uncertainties.columns:
            mean_uncertainty = uncertainties[col].mean()
            min_val = mean_uncertainty * min_ratio
            max_val = mean_uncertainty * max_ratio
            uncertainties[col] = uncertainties[col].clip(lower=min_val, upper=max_val)
    else:
        mean_uncertainty = uncertainties.mean()
        min_val = mean_uncertainty * min_ratio
        max_val = mean_uncertainty * max_ratio
        uncertainties = uncertainties.clip(lower=min_val, upper=max_val)

    return uncertainties


def check_data_quality(
    reconstructions: pd.DataFrame, uncertainties: pd.DataFrame | None = None
) -> dict:
    """检查重建数据和不确定性的质量

    Args:
        reconstructions: 重建数据
        uncertainties: 不确定性数据

    Returns:
        包含数据质量信息的字典
    """
    quality_info = {
        "reconstructions": {
            "shape": reconstructions.shape,
            "missing_values": reconstructions.isna().sum().to_dict(),
            "stats": reconstructions.describe().to_dict(),
        }
    }

    if uncertainties is not None:
        quality_info["uncertainties"] = {
            "shape": uncertainties.shape,
            "missing_values": uncertainties.isna().sum().to_dict(),
            "stats": uncertainties.describe().to_dict(),
            "zero_or_negative": (uncertainties <= 0).sum().to_dict(),
        }

    return quality_info


def combine_reconstructions(
    reconstructions: pd.DataFrame,
    uncertainties: pd.DataFrame | None = None,
    n_samples: int = 2000,
    n_tune: int = 1000,
    standardize: bool = True,
) -> tuple[pd.DataFrame, az.InferenceData]:
    """使用贝叶斯方法整合多个重建序列"""
    # 标准化数据
    if standardize:
        reconstructions, uncertainties = standardize_both(
            reconstructions, uncertainties
        )
    length, cols = reconstructions.shape

    with pm.Model():
        # 旱涝真值
        true_drought = pm.Normal("true_drought", mu=0, sigma=1, shape=length)
        # StudentT 分布的自由度
        nu = pm.Gamma("nu", alpha=2, beta=0.1)  # 弱信息先验

        # 为每个重建序列创建观测
        for col in reconstructions.columns:
            mask = (~reconstructions[col].isna()).to_numpy()
            obs_data = reconstructions[col][mask].values

            if uncertainties is not None:
                obs_sigma = uncertainties[col][mask].values
                # 确保不确定性在合理范围内
                # obs_sigma = np.clip(obs_sigma, 0.2, 1.0)
            else:
                obs_sigma = np.full_like(obs_data, fill_value=obs_data.std())

            # 使用 StudentT 分布
            pm.StudentT(
                f"obs_{col}",
                nu=nu,
                mu=true_drought[np.where(mask)[0]],
                sigma=obs_sigma,
                observed=obs_data,
            )

        # 使用更简单的采样设置
        trace = pm.sample(
            n_samples,
            tune=n_tune,
            return_inferencedata=True,
            chains=4,
            init="jitter+adapt_diag",  # 改回更简单的初始化方法
            target_accept=0.95,
            cores=4,
        )

    # 提取结果
    summary = az.summary(trace, var_names=["true_drought"])

    combined = pd.DataFrame(
        {
            "mean": summary["mean"].values,
            "sd": summary["sd"].values,
            "hdi_3%": summary["hdi_3%"].values,
            "hdi_97%": summary["hdi_97%"].values,
        },
        index=reconstructions.index,
    )

    return combined, trace


def plot_combined_reconstruction(combined, data):
    """绘制整合后的重建序列及其不确定性"""
    _, ax = plt.subplots(figsize=(11, 3), tight_layout=True)
    ax.plot(combined.index, combined["mean"], "k-", label="Combined mean")
    ax.fill_between(
        combined.index,
        combined["hdi_3%"],
        combined["hdi_97%"],
        alpha=0.2,
        # label="94% Credible Interval",
    )

    # 绘制原始数据
    for col in data.columns:
        ax.scatter(
            data.index,
            data[col],
            alpha=0.5,
            label=f"Original {col}",
        )

    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_title("Combined Reconstruction with Uncertainties")
    return ax
