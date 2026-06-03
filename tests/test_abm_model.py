#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""ABM 模型层逻辑测试（区别于 test_abm.py 的缓存、test_abm_mechanism.py 的感知）：

- ``__init__`` 的参数契约校验（step_per_year / subannual_aggregation）；
- ``_aggregate_to_yearly`` 的 mean/sum/last/恒等 聚合；
- ``archive_it`` 的 loss_rate 丢失逻辑（含全局 np.random 边界）；
- ``climate_df`` / ``is_nan`` 在未运行时的退化行为；
- 观察者生命周期：未达 min_age 不记录；
- personal 基线整体跑通后 climate_df 形状与索引对齐 spin-up。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

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


class TestInitValidation:
    def test_step_per_year_must_be_positive(self):
        with pytest.raises(ValueError, match="step_per_year must be >= 1"):
            ClimateObservingModel(parameters=_cfg(step_per_year=0))

    def test_subannual_aggregation_must_be_known(self):
        with pytest.raises(ValueError, match="subannual_aggregation must be one of"):
            ClimateObservingModel(parameters=_cfg(subannual_aggregation="median"))

    def test_spin_up_years_formula(self):
        """spin_up = new_agents * (max_age - min_age + 1)，是稳态人口的填充期。"""
        model = ClimateObservingModel(
            parameters=_cfg(new_agents=3, max_age=4, min_age=2)
        )
        assert model.spin_up_years == 3 * (4 - 2 + 1)
        # 总模拟年 = 业务年 + spin-up
        assert model._years == 5 + model.spin_up_years


class TestAggregateToYearly:
    def test_step_per_year_one_is_identity(self):
        model = ClimateObservingModel(parameters=_cfg(step_per_year=1))
        s = pd.Series(np.arange(6.0), index=range(6))
        # step_per_year==1 时直接原样返回（同一对象，零拷贝）
        assert model._aggregate_to_yearly(s) is s

    def test_mean_aggregation(self):
        model = ClimateObservingModel(
            parameters=_cfg(step_per_year=3, subannual_aggregation="mean")
        )
        s = pd.Series(np.arange(9.0), index=range(9))
        # 每 3 个 tick 归一年：[0,1,2]->1, [3,4,5]->4, [6,7,8]->7
        assert model._aggregate_to_yearly(s).tolist() == [1.0, 4.0, 7.0]

    def test_sum_aggregation(self):
        model = ClimateObservingModel(
            parameters=_cfg(step_per_year=3, subannual_aggregation="sum")
        )
        s = pd.Series(np.arange(9.0), index=range(9))
        assert model._aggregate_to_yearly(s).tolist() == [3.0, 12.0, 21.0]

    def test_last_aggregation(self):
        model = ClimateObservingModel(
            parameters=_cfg(step_per_year=3, subannual_aggregation="last")
        )
        s = pd.Series(np.arange(9.0), index=range(9))
        assert model._aggregate_to_yearly(s).tolist() == [2.0, 5.0, 8.0]


class TestArchiveItLossRate:
    def test_loss_rate_one_records_nothing(self):
        """loss_rate=1.0 → 每条记录都被丢失，归档恒空。"""
        model = ClimateObservingModel(parameters=_cfg(loss_rate=1.0))
        for _ in range(20):
            model.archive_it(2)
        assert sum(len(v) for v in model._archive.values()) == 0

    def test_loss_rate_zero_records_everything(self):
        """loss_rate=0.0 → 全部写入当前 tick。"""
        model = ClimateObservingModel(parameters=_cfg(loss_rate=0.0))
        for _ in range(5):
            model.archive_it(2)
        assert model._archive[model.time.tick] == [2, 2, 2, 2, 2]

    def test_loss_rate_uses_global_random(self, monkeypatch):
        """阈值判定 np.random.random() < loss_rate：抽样 0.3 < 0.5 → 丢弃。"""
        model = ClimateObservingModel(parameters=_cfg(loss_rate=0.5))
        monkeypatch.setattr(np.random, "random", lambda: 0.3)
        model.archive_it(1)  # 0.3 < 0.5 → 丢
        assert sum(len(v) for v in model._archive.values()) == 0
        monkeypatch.setattr(np.random, "random", lambda: 0.7)
        model.archive_it(1)  # 0.7 ≥ 0.5 → 记
        assert model._archive[model.time.tick] == [1]


class TestDegenerateBeforeRun:
    def test_is_nan_true_on_fresh_model(self):
        model = ClimateObservingModel(parameters=_cfg())
        # is_nan 返回 numpy bool，用真值判断而非 `is True` 身份比较
        assert bool(model.is_nan)

    def test_climate_df_raises_before_run(self):
        model = ClimateObservingModel(parameters=_cfg())
        with pytest.raises(ValueError, match="Collective memory is all NaN"):
            _ = model.climate_df


class TestObserverLifecycle:
    def test_young_observer_does_not_record(self, monkeypatch):
        """age < min_age 的观察者 step() 只更新记忆，不感知/记录。"""
        model = ClimateObservingModel(parameters=_cfg(min_age=2))
        obs = model.agents.new(ClimateObserver, 1, max_age=4, min_age=2)[0]
        called = {"write": 0, "archive": 0}
        monkeypatch.setattr(
            obs,
            "write_down",
            lambda *a, **k: called.__setitem__("write", called["write"] + 1),
        )
        monkeypatch.setattr(
            model,
            "archive_it",
            lambda *a, **k: called.__setitem__("archive", called["archive"] + 1),
        )
        # 刚出生 age()==0 < min_age==2
        obs.step()
        assert called["write"] == 0
        assert called["archive"] == 0
        # 但当前气候仍进入个人记忆
        assert len(obs.memory) == 1


class TestPersonalBaselineIntegration:
    def test_personal_run_produces_aligned_climate_df(self):
        """personal 基线整体跑通：climate_df 形状=业务年数，索引从 spin-up 起。"""
        model = ClimateObservingModel(parameters=_cfg(memory_baseline="personal"))
        model.run_model()
        df = model.climate_df
        assert list(df.columns) == ["climate", "collective_memory_climate"]
        assert len(df) == 5  # years
        assert df.index.min() == model.spin_up_years
