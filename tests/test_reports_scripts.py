#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""第 5 层编排脚本的纯函数测试（reports/ 下的脚本不是包模块，按文件路径加载）。

只测「有真实逻辑」的纯函数，不实跑子进程 / 不渲染图：
- ``run_sensitivity.parse_args``：子命令默认值契约（SLURM 脚本依赖它）；
- ``climate_scenario_experiments.summarize_scenario``：峰值窗口/强度提取；
- ``plot_sobol`` 的 ``baseline_of`` / ``failure_summary`` 纯标注辅助函数。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _load(name: str, filename: str):
    """按文件路径加载 reports/ 下的脚本为模块。"""
    spec = importlib.util.spec_from_file_location(name, REPORTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_sensitivity():
    return _load("_rs_under_test", "run_sensitivity.py")


@pytest.fixture(scope="module")
def climate_scenarios():
    return _load("_cse_under_test", "climate_scenario_experiments.py")


@pytest.fixture(scope="module")
def plot_sobol():
    return _load("_ps_under_test", "plot_sobol.py")


class TestRunSensitivityArgs:
    def test_smoke_uses_tiny_defaults(self, run_sensitivity):
        """smoke 子命令把 repeats/years 压到 2/20（SLURM smoke 依赖此默认）。"""
        args = run_sensitivity.parse_args(["smoke"])
        assert args.repeats == 2
        assert args.years == 20
        assert args.memory_baseline == "personal"
        assert args.func is run_sensitivity.cmd_smoke

    def test_sobol_defaults(self, run_sensitivity):
        args = run_sensitivity.parse_args(["sobol", "--memory-baseline", "collective"])
        assert args.N == 1024
        assert args.repeats == 30
        assert args.second_order is False
        assert args.memory_baseline == "collective"
        assert args.func is run_sensitivity.cmd_sobol

    def test_stage_is_required(self, run_sensitivity):
        with pytest.raises(SystemExit):
            run_sensitivity.parse_args([])

    def test_invalid_baseline_rejected(self, run_sensitivity):
        with pytest.raises(SystemExit):
            run_sensitivity.parse_args(["morris", "--memory-baseline", "nonsense"])


class TestSummarizeScenario:
    def test_peak_window_and_strength(self, climate_scenarios, tmp_path):
        """每列（replicate）取 idxmax→峰值窗口、max→峰值强度，再跨列平均。"""
        corr = pd.DataFrame(
            {"r1": [0.1, 0.3, 0.2], "r2": [0.2, 0.25, 0.4]},
            index=[10, 20, 30],
        )
        path = tmp_path / "correlations.csv"
        corr.to_csv(path)
        sc = climate_scenarios.Scenario(
            name="t", climate_process="iid", step_per_year=1, climate_phi=0.6
        )
        peaks, summary = climate_scenarios.summarize_scenario(sc, path)

        # r1 峰在窗口 20(0.3)，r2 峰在窗口 30(0.4)
        assert peaks["window_peak_location"].tolist() == [20, 30]
        assert peaks["window_peak_strength"].tolist() == [0.3, 0.4]
        assert summary["window_peak_location_mean"] == pytest.approx(25.0)
        assert summary["window_peak_strength_mean"] == pytest.approx(0.35)
        assert summary["n_runs"] == 2

    def test_single_run_std_is_zero(self, climate_scenarios, tmp_path):
        """单 replicate 时 ddof=1 std 无定义，回退 0.0 而非 NaN。"""
        corr = pd.DataFrame({"r1": [0.1, 0.5, 0.2]}, index=[10, 20, 30])
        path = tmp_path / "correlations.csv"
        corr.to_csv(path)
        sc = climate_scenarios.Scenario(
            name="solo", climate_process="iid", step_per_year=1
        )
        _, summary = climate_scenarios.summarize_scenario(sc, path)
        assert summary["n_runs"] == 1
        assert summary["window_peak_location_std"] == 0.0
        assert summary["window_peak_strength_std"] == 0.0


class TestPlotSobolHelpers:
    def test_baseline_of_extracts_token(self, plot_sobol):
        assert plot_sobol.baseline_of(Path("20260504-sobol-collective")) == "collective"
        assert plot_sobol.baseline_of(Path("x-sobol-personal")) == "personal"
        assert plot_sobol.baseline_of(Path("y-sobol-model")) == "model"
        # 无已知 token → 回退到目录名本身
        assert plot_sobol.baseline_of(Path("weird-name")) == "weird-name"

    def test_failure_summary_counts_ok(self, plot_sobol, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        pd.DataFrame(
            {"sample_idx": [0, 1, 2, 3], "status": ["ok", "ok", "error: x", "ok"]}
        ).to_csv(d / "raw_outputs.csv", index=False)
        total, ok, pct = plot_sobol.failure_summary(d)
        assert total == 4
        assert ok == 3
        assert pct == pytest.approx(25.0)

    def test_failure_summary_missing_file(self, plot_sobol, tmp_path):
        total, ok, pct = plot_sobol.failure_summary(tmp_path / "nope")
        assert total == 0
        assert ok == 0
