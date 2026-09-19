#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

from shifting_baseline.utils.log import get_logger

log = get_logger()


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


def combine_reconstructions(
    reconstructions: pd.DataFrame,
    uncertainties: pd.DataFrame | None = None,
    n_samples: int = 2000,
    n_tune: int = 1000,
    standardize: bool = True,
    random_seed: int | None = 42,
    theta_sigma: float = 1.0,
    nu_prior: tuple[float, float] = (2.0, 0.1),
    nu_fixed: float | None = None,
) -> tuple[pd.DataFrame, az.InferenceData]:
    r"""使用贝叶斯方法整合多个重建序列

    Latent-variable model (defaults = the manuscript's specification):

    $$ \theta_t \sim \mathcal{N}(0, \sigma_\theta), \quad
       \nu \sim \mathrm{Gamma}(\alpha, \beta), \quad
       y_{i,t} \sim \mathrm{StudentT}(\nu, \theta_t, \sigma_{i,t}) $$

    where $\sigma_{i,t}$ is a fixed plug-in (rolling within-series SD), not a
    parameter. The prior arguments exist for the prior-sensitivity analysis
    (``scripts/bayes_prior_sensitivity.py``).

    Args:
        theta_sigma: 潜变量 $\theta_t$ 正态先验的标准差。
        nu_prior: 自由度 $\nu$ 的 Gamma(alpha, beta) 先验（rate 参数化）。
        nu_fixed: 若给定则固定 $\nu$ 而不估计；``np.inf`` 表示正态似然。
    """
    # 标准化数据
    if standardize:
        reconstructions, uncertainties = standardize_both(
            reconstructions, uncertainties
        )
    length, cols = reconstructions.shape

    with pm.Model():
        # 旱涝真值
        true_drought = pm.Normal("true_drought", mu=0, sigma=theta_sigma, shape=length)
        # StudentT 分布的自由度：默认估计（弱信息 Gamma 先验），也可固定；
        # nu_fixed = inf 取其极限，即正态似然（nu = None）
        if nu_fixed is None:
            nu = pm.Gamma("nu", alpha=nu_prior[0], beta=nu_prior[1])
        elif np.isinf(nu_fixed):
            nu = None
        else:
            nu = nu_fixed

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

            mu = true_drought[np.where(mask)[0]]
            if nu is None:
                pm.Normal(f"obs_{col}", mu=mu, sigma=obs_sigma, observed=obs_data)
            else:
                # 使用 StudentT 分布
                pm.StudentT(
                    f"obs_{col}",
                    nu=nu,
                    mu=mu,
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
            random_seed=random_seed,
        )

    diagnose_trace(trace)

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


def diagnose_trace(trace: az.InferenceData) -> dict[str, float]:
    """收敛诊断：对所有采样参数（含 nu，若被估计）报告 R-hat / ESS，供 Methods/SI 引用

    Convergence diagnostics over ALL sampled params (incl. nu), not just true_drought.
    Returns ``nu_*`` as NaN when nu was fixed rather than sampled.
    """
    var_names = [v for v in ("true_drought", "nu") if v in trace.posterior]
    diag = az.summary(trace, var_names=var_names)
    result = {
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
        "nu_mean": np.nan,
        "nu_hdi_low": np.nan,
        "nu_hdi_high": np.nan,
    }
    log.info(
        "MCMC 收敛诊断: max R-hat=%.4f, min ESS(bulk)=%.0f, min ESS(tail)=%.0f",
        result["max_rhat"],
        result["min_ess_bulk"],
        result["min_ess_tail"],
    )
    if "nu" in trace.posterior:
        nu_row = diag.loc["nu"]
        result.update(
            nu_mean=float(nu_row["mean"]),
            nu_hdi_low=float(nu_row["hdi_3%"]),
            nu_hdi_high=float(nu_row["hdi_97%"]),
        )
        log.info(
            "后验自由度 nu-hat=%.2f, 94%% HDI=[%.2f, %.2f]",
            result["nu_mean"],
            result["nu_hdi_low"],
            result["nu_hdi_high"],
        )
    return result


def summarize_latent(combined: pd.DataFrame) -> dict[str, float]:
    """汇总各年潜变量后验的不确定性（跨年份的中位数与 5–95% 范围）

    Args:
        combined: ``combine_reconstructions`` 的输出（含 sd / hdi_3% / hdi_97% 列）。
    """
    sd = combined["sd"]
    width = combined["hdi_97%"] - combined["hdi_3%"]
    return {
        "sd_median": float(sd.median()),
        "sd_q05": float(sd.quantile(0.05)),
        "sd_q95": float(sd.quantile(0.95)),
        "hdi94_width_median": float(width.median()),
        "hdi94_width_q05": float(width.quantile(0.05)),
        "hdi94_width_q95": float(width.quantile(0.95)),
    }
