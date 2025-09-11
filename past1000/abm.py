#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging
import traceback
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from abses import Actor, Experiment, MainModel
from hydra import main
from omegaconf import DictConfig
from scipy.stats import norm

from past1000.calibration import MismatchReport
from past1000.compare import compare_corr_2d
from past1000.constants import MAX_AGE
from past1000.filters import calc_std_deviation, classify, classify_single_value
from past1000.utils.calc import rand_generate_from_std_levels
from past1000.utils.email import send_notification_email

if TYPE_CHECKING:
    from past1000.utils.types import CorrFunc

log = logging.getLogger(__name__)


class Model(MainModel):
    """Agent-based climate event model.

    Simulates a world with climate observers who record extreme climate events based on their perception.
    The collective records are compared with actual climate extremes.

    Attributes:
        _max_age (int): Maximum age of an observer.
        _new_agents (int): Number of new agents per step.
        _min_age (int): Minimum age for recording events.
        _mode_cache (Optional[pd.Series]): Cached mode results for current tick.
        _mode_cache_tick (int): Tick at which mode cache was generated.
        _last_event_years (dict): Cache for last occurrence year of each event level.
        final_corr (Optional[pd.Series]): Final correlation results.
        spin_up_years (int): Spin-up years for agent initialization.
        _years (int): Total simulation years.
        _climate (np.ndarray): Climate time series.
        _archive (dict[int, list[int]]): Archive of recorded events per year.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        years: int = self.p.get("years", 100)
        self._max_age: int = self.p.get("max_age", 40)
        self._new_agents: int = self.p.get("new_agents", 10)
        self._min_age: int = self.p.get("min_age", 10)
        self._mode_cache: Optional[pd.Series] = None
        self._mode_cache_tick: int = -1
        self._collective_cache: Optional[pd.Series] = None
        self._collective_cache_tick: int = -1
        self.spin_up_years: int = self._new_agents * (self._max_age - self._min_age + 1)
        self._years: int = years + self.spin_up_years
        self._climate: np.ndarray = np.random.normal(0, 1, self._years)
        self._archive: dict[int, list[int]] = {i: [] for i in range(self._years)}
        log.info(f"运行模式: {self.p.get('mode', 'exp')}")

    @property
    def climate_now(self) -> float:
        """Current climate value at the current tick.

        Returns:
            float: Current climate value.
        """
        return self._climate[self.time.tick]

    @property
    def climate_series(self) -> pd.Series:
        """Full climate time series.

        Returns:
            pd.Series: Climate values indexed by year.
        """
        return pd.Series(self._climate, index=range(self._years))

    @property
    def estimation(self) -> pd.DataFrame:
        """Estimation DataFrame containing value, expected (mode), and classified series.

        Returns:
            pd.DataFrame: DataFrame with columns 'value', 'expect', and 'classified'.
        """
        value = self.climate_series
        expect = classify(self.collective_memory_climate)
        classified = classify(self.climate_series)
        return (
            pd.DataFrame(
                {
                    "value": value,
                    "expect": expect,
                    "classified": classified,
                }
            )
            .loc[self.spin_up_years :]
            .dropna()
        )

    @property
    def collective_memory_climate(self) -> pd.Series:
        """Mean of recorded events per year, cached per tick.

        Returns:
            pd.Series: Rounded mean of recorded events for each year.
        """
        # Return cache if tick has not advanced
        if (
            self._collective_cache is not None
            and self._collective_cache_tick == self.time.tick
        ):
            return self._collective_cache

        current_archive = {k: v for k, v in self._archive.items() if v}
        series = pd.Series(
            {
                k: rand_generate_from_std_levels(np.array(v)).mean()
                for k, v in current_archive.items()
            }
        )
        # Update cache for current tick
        self._collective_cache = series
        self._collective_cache_tick = self.time.tick
        return series

    @property
    def mismatch_report(self) -> MismatchReport:
        """Mismatch report of the model.

        Returns:
            MismatchReport: Mismatch report of the model.
        """
        mismatch_report = MismatchReport(
            pred=classify(self.collective_memory_climate),
            true=classify(self.climate_series),
            value_series=self.climate_series,
        )
        mismatch_report.analyze_error_patterns()
        return mismatch_report

    def write_down(self, extreme: int) -> None:
        """Record an extreme climate event for the current tick.

        Args:
            extreme (int): The classified extreme event level.
        """
        loss_rate: float = self.p.get("loss_rate", 0.2)
        if np.random.random() < loss_rate:
            return
        self._archive[self.time.tick].append(extreme)
        # Cache is tick-based, so no need to clear unless logic changes.

    def step(self) -> None:
        """Advance the model by one tick: update global info, spawn new agents, and step all agents."""
        if self.time.tick == self._years - 1:
            self.running = False
        self.agents.new(ClimateObserver, self._new_agents, max_age=self._max_age)
        self.agents.do("step")

    @property
    def stable_df(self) -> pd.DataFrame:
        slice_ = slice(self.spin_up_years, None)
        return pd.DataFrame(
            {
                "climate": self.climate_series.loc[slice_],
                "collective_memory_climate": self.collective_memory_climate.loc[slice_],
            }
        )

    def end(self) -> None:
        """Save estimation results and compute final correlations."""
        min_period: int = self.p.get("min_period", 2)
        filter_side: str = self.p.get("filter_side", "right")
        corr_method: CorrFunc = self.p.get("corr_method", "kendall")
        windows = np.arange(2, 100)
        min_periods = np.repeat(min_period, 98)
        rs, _, _ = compare_corr_2d(
            self.stable_df["collective_memory_climate"],
            self.stable_df["climate"],
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            corr_method=corr_method,
            filter_side=filter_side,
        )
        corr_series = pd.Series(
            data=rs,
            index=windows,
            name=f"model_{self.run_id}",
        )
        # if test mode, don't save any files
        if self.p.get("mode", "exp") == "test":
            return
        output_file = self.outpath / "correlations.csv"
        # Smart save logic: create new file or append column
        if output_file.exists():
            # File exists, read and add new column
            existing_df = pd.read_csv(output_file, index_col=0)
            existing_df[corr_series.name] = corr_series
            existing_df.to_csv(output_file)
        else:
            # File doesn't exist, create new DataFrame
            new_df = pd.DataFrame({corr_series.name: corr_series})
            new_df.to_csv(output_file)


class ClimateObserver(Actor):
    """Climate observer agent.

    Observes, perceives, and records extreme climate events based on personal memory and collective memory (mode).
    More likely to record extreme events than normal years.

    Attributes:
        _memory (deque): Personal memory of extreme events.
        age (int): Observer's age.
        _max_age (int): Maximum age for the observer.
        _min_age (int): Minimum age to start recording events.
    """

    def __init__(self, *args, max_age: int = MAX_AGE, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory: deque = deque(maxlen=max_age)
        self.age: int = 1
        self._max_age: int = max_age
        self._min_age: int = self.p.get("min_age", 10)

    @property
    def memory(self) -> np.ndarray:
        """Personal memory of extreme climate events.

        Returns:
            np.ndarray: Array of remembered climate values.
        """
        return np.array(self._memory)

    def write_down(
        self,
        event: float,
        scale: float = 1,
        f0: float = 0.1,
    ) -> bool:
        """Decide whether to record an extreme event based on the 'negativity bias' principle.

        Args:
            event (float): Standardized z-score of the event.
            scale (float): Scale for the z-score; higher means less likely to record.
            f0 (float): Baseline probability to record the event (0 < f0 < 0.5).
        Returns:
            bool: Whether the event is recorded.
        Raises:
            ValueError: If f0 is not between 0 and 0.5.
        """
        if f0 > 0.5 or f0 < 0:
            raise ValueError("f0 must be between 0 and 0.5")
        prob = norm.sf(abs(event), scale=scale)
        return np.random.random() < f0 + 0.5 - prob

    def perception(self, climate: float) -> float:
        """Perceive the z-score of the current climate relative to memory.

        Args:
            climate (float): Current climate value.
        Returns:
            float: Z-score of the current climate.
        """
        if self.model.p.memory_baseline == "personal":
            baseline = self.memory.mean()
            std = self.memory.std()
        elif self.model.p.memory_baseline == "model":
            baseline = self.model.climate_series.mean()
            std = self.model.climate_series.std()
        elif self.model.p.memory_baseline == "collective":
            baseline = self.model.collective_memory_climate.mean()
            std = self.model.collective_memory_climate.std()
        else:
            raise ValueError("Invalid memory baseline")
        if np.isnan(baseline):
            baseline = 0
        if np.isnan(std):
            std = 1
        return climate - baseline / std

    def step(self) -> None:
        """Update observer state at each step.

        The observer:
        - Increases age by 1.
        - Updates memory with current climate.
        - If the observer is old enough, it perceives the climate and decides whether to record an event. When it records an event, it will be re-judged based on collective memory (mode).
        - If the observer is too old, it dies.
        """
        self.age += 1
        climate = self.model.climate_now
        self._memory.append(climate)
        if self.age < self._min_age:
            return
        z = self.perception(climate)
        if self.write_down(z):
            extreme_level = classify_single_value(z)
            self.model.write_down(extreme_level)
        if self.age > self._max_age:
            self.die()


@main(config_path="../config", config_name="config", version_base=None)
def repeat_run(cfg: Optional[DictConfig] = None) -> None:
    """Run the model multiple times and save correlation results.

    Args:
        cfg (Optional[DictConfig]): Configuration object.
    Raises:
        AssertionError: If cfg is None.
    """
    assert cfg is not None, "cfg is None"
    exp = Experiment.new(Model, cfg=cfg)
    log.info(f"运行模式: {exp.cfg.model.mode}")
    repeats = exp.cfg.model.repeats
    num_process = exp.cfg.model.num_process
    exp.batch_run(repeats=repeats, parallels=num_process)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"ABM 模型开始运行: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        repeat_run()
        send_notification_email(success=True, start_time=start_time)
        print("✅ ABM 模型运行成功完成")
    except Exception as e:  # pylint: disable=broad-except # 需要捕获所有异常以发送邮件通知
        error_msg = f"{str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
        send_notification_email(
            success=False, error_msg=error_msg, start_time=start_time
        )
        print(f"❌ ABM 模型运行失败: {e}")
