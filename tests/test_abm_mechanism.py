#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""ABM 核心机制测试：观察者 perceive（感知/z-score）与 write_down（负性偏差记录）。

重点是 perceive 的 z-score 公式回归——审稿人 R2 曾指出运算优先级 bug
（``climate - baseline/std`` 应为 ``(climate - baseline)/std``），这里钉死正确式。
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from shifting_baseline import abm as abm_mod
from shifting_baseline.abm import ClimateObserver, ClimateObservingModel


def _cfg(memory_baseline: str = "personal", **overrides) -> DictConfig:
    base = {
        "years": 5,
        "max_age": 4,
        "min_age": 2,
        "new_agents": 3,
        "loss_rate": 0.0,
        "memory_baseline": memory_baseline,
        "mode": "test",
        "climate_process": "iid",
        "climate_sigma": 1.0,
        "climate_phi": 0.0,
        "step_per_year": 1,
        "subannual_aggregation": "mean",
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def _new_observer(model) -> ClimateObserver:
    return model.agents.new(ClimateObserver, 1, max_age=4, min_age=2)[0]


@pytest.fixture
def personal_obs():
    model = ClimateObservingModel(parameters=_cfg("personal"))
    return model, _new_observer(model)


class TestPerceiveZScore:
    def test_personal_formula_regression(self, personal_obs):
        """回归 R2 的优先级 bug：必须是 (climate - baseline) / std。"""
        _, obs = personal_obs
        obs._memory.clear()
        for v in [1.0, 2.0, 3.0, 4.0]:
            obs._memory.append(v)
        mem = np.array([1.0, 2.0, 3.0, 4.0])  # mean=2.5, std(ddof=0)=1.118
        z = obs.perceive(5.0)
        assert z == pytest.approx((5.0 - mem.mean()) / mem.std())
        # 绝不能等于 buggy 的 climate - baseline/std
        assert z != pytest.approx(5.0 - mem.mean() / mem.std())

    def test_empty_memory_nan_fallback(self, personal_obs):
        """空 memory → baseline=NaN→0, std=NaN→1 → z 等于原始 climate。"""
        _, obs = personal_obs
        obs._memory.clear()
        assert obs.perceive(1.5) == pytest.approx(1.5)

    def test_uses_model_baseline_when_configured(self):
        model = ClimateObservingModel(parameters=_cfg("model"))
        obs = _new_observer(model)
        mean, std = model.model_baseline_stats
        assert obs.perceive(2.0) == pytest.approx((2.0 - mean) / std)

    def test_invalid_baseline_raises(self, personal_obs):
        model, obs = personal_obs
        model.p.memory_baseline = "nonsense"
        with pytest.raises(ValueError, match="Invalid memory baseline"):
            obs.perceive(1.0)


class TestWriteDownNegativityBias:
    def test_f0_out_of_range_raises(self, personal_obs):
        _, obs = personal_obs
        with pytest.raises(ValueError, match="f0 must be"):
            obs.write_down(0.0, f0=0.6)
        with pytest.raises(ValueError, match="f0 must be"):
            obs.write_down(0.0, f0=-0.1)

    def test_extreme_more_likely_than_normal(self, personal_obs, monkeypatch):
        """同一随机抽样下，极端事件被记录、正常年不被记录（负性偏差）。"""
        _, obs = personal_obs
        monkeypatch.setattr(np.random, "random", lambda: 0.3)
        # z=0: 阈值=f0=0.1 → 0.3<0.1 → 不记录
        assert not obs.write_down(0.0, f0=0.1)
        # z=10（极端）: 阈值≈f0+0.5=0.6 → 0.3<0.6 → 记录
        assert obs.write_down(10.0, f0=0.1)

    def test_z0_records_with_prob_f0(self, personal_obs, monkeypatch):
        _, obs = personal_obs
        monkeypatch.setattr(np.random, "random", lambda: 0.05)
        assert obs.write_down(0.0, f0=0.1)  # 0.05 < 0.1
        monkeypatch.setattr(np.random, "random", lambda: 0.15)
        assert not obs.write_down(0.0, f0=0.1)  # 0.15 > 0.1
