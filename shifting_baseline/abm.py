#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import logging
import traceback
from collections import deque
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from abses import Actor, Experiment, MainModel
from hydra import main
from omegaconf import DictConfig
from scipy.stats import norm

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.compare import compare_corr_2d
from shifting_baseline.constants import MAX_AGE
from shifting_baseline.filters import (
    calc_std_deviation,
    classify,
    classify_single_value,
)
from shifting_baseline.utils.calc import rand_generate_from_std_levels
from shifting_baseline.utils.email import send_notification_email

if TYPE_CHECKING:
    from shifting_baseline.utils.types import CorrFunc

from shifting_baseline.utils.log import get_logger

log = get_logger(__name__)


class ClimateObservingModel(MainModel):
    """An agent-based  model of climate event recording.

    Simulates a world with climate observers who record extreme climate events based on their perception.
    The collective records are compared with actual climate extremes.
    The model can be used to study the relationship between climate events and human behavior.

    Attributes:
        years (int): Total simulation years.
        _max_age (int): Maximum age of an observer.
        _new_agents (int): Number of new agents per step.
        _min_age (int): Minimum age for recording events.
        _collective_cache (Optional[pd.Series]): Cached collective memory results for current tick.
        _collective_cache_tick (int): Tick at which collective memory cache was generated.
        spin_up_years (int): Spin-up years for agent initialization.
        _years (int): Total simulation years.
        _climate (np.ndarray): Climate time series.
        _archive (dict[int, list[int]]): Archive of recorded events per year.
        final_corr (Optional[pd.Series]): Final correlation results.
        spin_up_years (int): Spin-up years for agent initialization.
        _years (int): Total simulation years.
        _climate (np.ndarray): Climate time series.
        _archive (dict[int, list[int]]): Archive of recorded events per year.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Years to simulate
        years: int = self.p.get("years", 100)
        # Maximum age for an observer
        self._max_age: int = self.p.get("max_age", 40)
        self._new_agents: int = self.p.get("new_agents", 10)
        # Minimum age for recording events
        self._min_age: int = self.p.get("min_age", 10)
        # Cache for collective memory
        self._collective_cache: Optional[pd.Series] = None
        self._collective_cache_tick: int = -1
        self.spin_up_years: int = self._new_agents * (self._max_age - self._min_age + 1)
        # Total simulation years
        self._years: int = years + self.spin_up_years
        # Climate time series
        self._climate: np.ndarray = np.random.normal(0, 1, self._years)
        # Archive of recorded events per year
        self._archive: dict[int, list[int]] = {i: [] for i in range(self._years)}
        log.info(f"运行模式: {self.p.get('mode', 'exp')}")

    @property
    def is_nan(self) -> bool:
        """Check if the collective memory is all NaN."""
        return self.collective_memory_climate.isna().all()

    @property
    def climate_now(self) -> float:
        """Current climate value at the current tick.
        (WDI, Z-score of the current climate)

        Returns:
            float: Current climate value.
        """
        return self._climate[self.time.tick]

    @cached_property
    def climate_series(self) -> pd.Series:
        """Full climate time series.

        Returns:
            pd.Series: Climate values indexed by year.
        """
        return pd.Series(self._climate, index=range(self._years))

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
        # Load simulated data
        climate_series = self.climate_df["climate"]
        collective_memory_climate = self.climate_df["collective_memory_climate"]
        # Create mismatch report
        mismatch_report = MismatchReport(
            pred=classify(collective_memory_climate, handle_na="skip"),
            true=classify(climate_series, handle_na="skip"),
            value_series=climate_series,
        )
        mismatch_report.analyze_error_patterns()
        return mismatch_report

    def archive_it(self, extreme: int) -> None:
        """Record an extreme climate event reported by an observer.

        Args:
            extreme (int): The classified extreme event level.
            Archive is a dictionary of lists, the key is the tick, the value is the list of extreme event levels reported by observers.
        """
        loss_rate: float = self.p.get("loss_rate", 0.2)
        if np.random.random() < loss_rate:
            return
        self._archive[self.time.tick].append(extreme)
        # Cache is tick-based, so no need to clear unless logic changes.

    @property
    def climate_df(self) -> pd.DataFrame:
        """Climate DataFrame, sliced by spin-up years.
        (Objective Climate, Collective Memory Climate)

        Returns:
            pd.DataFrame: DataFrame with columns 'climate' and 'collective_memory_climate'.

        Raises:
            ValueError: If the collective memory is all NaN.
        """
        if self.is_nan:
            raise ValueError("Collective memory is all NaN, did you run the model?")
        slice_ = slice(self.spin_up_years, None)
        return pd.DataFrame(
            {
                "climate": self.climate_series.loc[slice_],
                "collective_memory_climate": self.collective_memory_climate.loc[slice_],
            }
        )

    def get_corr_curve(
        self,
        window_length: int = 100,
        min_window: int = 2,
        corr_method: CorrFunc = "kendall",
        **rolling_kwargs,
    ) -> pd.DataFrame:
        """Get the correlation curve of the model."""
        min_period: int = self.p.get("min_period", 2)
        filter_side: str = self.p.get("filter_side", "right")
        windows = np.arange(min_window, window_length)
        min_periods = np.repeat(min_period, window_length - min_window)
        corr = compare_corr_2d(
            self.climate_df["collective_memory_climate"],
            self.climate_df["climate"],
            corr_method=corr_method,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            filter_side=filter_side,
            **rolling_kwargs,
        )
        return pd.DataFrame(
            {
                corr_method: corr[0],
                "p_value": corr[1],
                "n_samples": corr[2],
            },
            index=windows,
        )

    def step(self) -> None:
        """Advance the model by one tick, including:
        - Update global info
        - Spawn new agents
        - Step all agents, including:
            - Update observer perception
            - Update observer writing down
        """
        if self.time.tick == self._years - 1:
            self.running = False
        self.agents.new(ClimateObserver, self._new_agents, max_age=self._max_age)
        self.agents.do("step")

    def end(self) -> None:
        """Save estimation results and compute final correlations."""
        corr_method: CorrFunc = self.p.get("corr_method", "kendall")
        corr_df = self.get_corr_curve()
        corr_series = pd.Series(
            data=corr_df[corr_method],
            index=corr_df.index,
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

    Observes, perceives, and records extreme climate events.
    An observer only has two methods: perceive and write_down.
    1. perceive:
        - Perceive the current climate and decide whether to record an event.
    2. write_down:
        - Write down the current climate. More likely to record extreme events than normal years.

    Attributes:
        _memory (deque): Personal memory of extreme events.
        age (int): Observer's age.
        _max_age (int): Maximum age for the observer.
        _min_age (int): Minimum age to start recording events.
    """

    def __init__(self, *args, max_age: int = MAX_AGE, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory: deque = deque(maxlen=max_age)
        self._max_age: int = max_age
        self._min_age: int = self.p.get("min_age", 10)

    @property
    def memory(self) -> np.ndarray:
        """Personal memory of extreme climate Z-score values.

        Returns:
            np.ndarray: Array of remembered climate Z-score values.
        """
        return np.array(self._memory)

    def write_down(
        self,
        z_score: float,
        scale: float = 1,
        f0: float = 0.1,
    ) -> bool:
        """Decide whether to record an extreme event based on the 'negativity bias' principle.

        Args:
            z_score (float): Standardized z-score of the event.
            scale (float): Scale for the z-score; higher means less likely to record.
            f0 (float): Base probability to record the Z-score (0 < f0 < 0.5). When f0 is 0, the observer will never record the Z-score when there is no difference between the current climate z-score and the baseline. When f0 is 0.5, the observer will record the Z-score with a probability of 0.5.
        Returns:
            bool: Whether the Z-score is recorded.
        Raises:
            ValueError: If f0 is not between 0 and 0.5.
        """
        if f0 > 0.5 or f0 < 0:
            raise ValueError("f0 must be between 0 and 0.5")
        prob = norm.sf(abs(z_score), scale=scale)
        return np.random.random() < f0 + 0.5 - prob

    def perceive(self, climate: float) -> float:
        """Perceive the z-score of the current climate.
        We assume that the observer always perceive the climate with a baseline.
        Here, we have three types of baseline:
        - personal: the observer's personal memory
        - model: the model's climate time series (objective climate)
        - collective: the collective memory of the model (collective memory climate)
        We use the baseline to re-calculate the z-score of the current climate.
        The hypothesis is that the observer will compare the current climatic extreme with the baseline.

        Args:
            climate (float): Current climate Z-score value.
        Returns:
            float: Z-score of the current climate.
        """
        # Personal baseline
        if self.model.p.memory_baseline == "personal":
            baseline = self.memory.mean()
            std = self.memory.std()
        # Model baseline
        elif self.model.p.memory_baseline == "model":
            baseline = self.model.climate_series.mean()
            std = self.model.climate_series.std()
        # Collective baseline
        elif self.model.p.memory_baseline == "collective":
            baseline = self.model.collective_memory_climate.mean()
            std = self.model.collective_memory_climate.std()
        else:
            raise ValueError("Invalid memory baseline")
        # Handle NaN values
        if np.isnan(baseline):
            baseline = 0
        if np.isnan(std):
            std = 1
        # Calculate the z-score of the current climate
        return climate - baseline / std

    def step(self) -> None:
        """Update observer state at each step.

        The observer:
        - Updates memory with current climate.
        - If the observer is old enough, it perceives the climate and decides whether to record an event.
        - If the observer is too old, it dies.
        """
        climate = self.model.climate_now
        self._memory.append(climate)
        # If the observer is too young, it won't record any event
        if self.age() < self._min_age:
            return
        z_score = self.perceive(climate)
        # If the observer records an event, classify and archive it
        if self.write_down(z_score):
            extreme_level = classify_single_value(z_score)
            self.model.archive_it(extreme_level)
        # If the observer is too old, it dies
        if self.age() > self._max_age:
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
    exp = Experiment.new(ClimateObservingModel, cfg=cfg)
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
