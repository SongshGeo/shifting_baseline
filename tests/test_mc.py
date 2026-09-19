#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Tests for the prior options of the Bayesian N-WDI integration (``mc.py``)."""

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.mc import combine_reconstructions, summarize_latent


@pytest.fixture(name="recons")
def fixture_recons() -> pd.DataFrame:
    """三条带噪声的合成重建序列，共享同一真值。"""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=30)
    data = {f"s{i}": truth + rng.normal(scale=0.3, size=30) for i in range(3)}
    df = pd.DataFrame(data, index=pd.RangeIndex(1600, 1630, name="year"))
    df.iloc[:5, 0] = np.nan  # 缺测年份被掩膜
    return df


def _fit(recons: pd.DataFrame, **kwargs):
    return combine_reconstructions(
        recons, n_samples=100, n_tune=100, random_seed=1, **kwargs
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs, samples_nu",
    [
        ({}, True),
        ({"nu_prior": (1.0, 1 / 30)}, True),
        ({"nu_fixed": 4.0}, False),
        ({"nu_fixed": np.inf}, False),
    ],
)
def test_prior_options_switch_model(recons, kwargs, samples_nu):
    """nu 仅在未固定时被采样；输出结构不随先验变化。"""
    combined, trace = _fit(recons, **kwargs)
    assert ("nu" in trace.posterior) is samples_nu
    assert list(combined.columns) == ["mean", "sd", "hdi_3%", "hdi_97%"]
    assert combined.index.equals(recons.index)
    assert combined["mean"].notna().all()


@pytest.mark.slow
def test_theta_prior_scale_changes_shrinkage(recons):
    """更窄的 theta 先验向 0 收缩更多。"""
    narrow, _ = _fit(recons, theta_sigma=0.1)
    wide, _ = _fit(recons, theta_sigma=10.0)
    assert narrow["mean"].abs().mean() < wide["mean"].abs().mean()


def test_summarize_latent():
    combined = pd.DataFrame(
        {
            "mean": [0.0, 0.0, 0.0],
            "sd": [0.1, 0.2, 0.3],
            "hdi_3%": [-0.2, -0.4, -0.6],
            "hdi_97%": [0.2, 0.4, 0.6],
        }
    )
    result = summarize_latent(combined)
    assert result["sd_median"] == pytest.approx(0.2)
    assert result["hdi94_width_median"] == pytest.approx(0.8)
