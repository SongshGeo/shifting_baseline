#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from abses import Actor, MainModel
from hydra import main
from omegaconf import DictConfig
from scipy.stats import mode, norm

from past1000.compare import compare_corr_2d
from past1000.constants import MAX_AGE, THRESHOLDS
from past1000.filters import calc_std_deviation, classify
from past1000.utils.config import get_output_dir

if TYPE_CHECKING:
    from past1000.utils.types import CorrFunc


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
        self._last_event_years: dict[int, int] = {}
        self.final_corr: Optional[pd.Series] = None
        self.spin_up_years: int = self._new_agents * (self._max_age - self._min_age + 1)
        self._years: int = years + self.spin_up_years
        self._climate: np.ndarray = np.random.normal(0, 1, self._years)
        self._archive: dict[int, list[int]] = {i: [] for i in range(self._years)}

    @property
    def climate(self) -> float:
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
    def historical_mean(self) -> pd.Series:
        """Historical mean of recorded events per year.

        Returns:
            pd.Series: Mean of recorded events for each year.
        """
        return pd.Series({k: np.mean(v) for k, v in self._archive.items()})

    @property
    def classified(self) -> pd.Series:
        """Classified climate series.

        Returns:
            pd.Series: Classified climate values.
        """
        return classify(self._climate)

    @property
    def estimation(self) -> pd.DataFrame:
        """Estimation DataFrame containing value, expected (mode), and classified series.

        Returns:
            pd.DataFrame: DataFrame with columns 'value', 'expect', and 'classified'.
        """
        value = self.climate_series
        # expect = self.mode
        expect = self.rounded_mean
        classified = self.classified
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
    def mode(self) -> pd.Series:
        """Mode of recorded events per year, cached per tick.

        Returns:
            pd.Series: Mode of recorded events for each year.
        """
        if self._mode_cache_tick == self.time.tick:
            return self._mode_cache
        current_archive = {k: v for k, v in self._archive.items() if v}
        result = pd.Series({k: mode(v)[0] for k, v in current_archive.items()})
        self._mode_cache = result
        self._mode_cache_tick = self.time.tick
        return result

    @property
    def rounded_mean(self) -> pd.Series:
        """Rounded mean of recorded events per year, cached per tick.

        Returns:
            pd.Series: Rounded mean of recorded events for each year.
        """
        current_archive = {k: v for k, v in self._archive.items() if v}
        return pd.Series({k: round(np.mean(v)) for k, v in current_archive.items()})

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
        self.update_last_event_years()
        if self.time.tick == self._years - 1:
            self.running = False
        self.agents.new(ClimateObserver, self._new_agents, max_age=self._max_age)
        self.agents.do("step")

    def update_last_event_years(self) -> None:
        """Update and cache the last occurrence year for each event level (called at each step)."""
        # current_mode = self.mode
        current_mode = self.rounded_mean
        if current_mode.empty:
            return
        unique_levels = current_mode.unique()
        for level in unique_levels:
            last_year = current_mode[current_mode == level].last_valid_index()
            self._last_event_years[level] = last_year

    def get_last_event_year_for_level(self, level: int) -> Optional[int]:
        """Get the last year a specific event level was recorded (for agent use).

        Args:
            level (int): The event level to query.
        Returns:
            Optional[int]: The last year this level was recorded, or None if never.
        """
        return self._last_event_years.get(level, None)

    def end(self) -> None:
        """Save estimation results and compute final correlations."""
        min_period: int = self.p.get("min_period", 2)
        filter_side: str = self.p.get("filter_side", "right")
        corr_method: CorrFunc = self.p.get("corr_method", "kendall")
        self.estimation.to_csv(self.outpath / f"estimation_{self.run_id}.csv")
        windows = np.arange(2, 100)
        min_periods = np.repeat(min_period, 98)
        rs, _, _ = compare_corr_2d(
            self.historical_mean,
            self.climate_series,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            corr_method=corr_method,
            filter_side=filter_side,
        )
        self.final_corr = pd.Series(
            data=rs,
            index=windows,
            name=f"model_{self.run_id}",
        )


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

    @property
    def classified(self) -> pd.Series:
        """Classified memory of extreme events.

        Returns:
            pd.Series: Classified memory values.
        """
        return classify(self.memory)

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
        return (climate - self.memory.mean()) / self.memory.std()

    def judge_extreme(self, z: float) -> int:
        """Classify the z-score as an extreme event level.

        Args:
            z (float): Z-score of the event.
        Returns:
            int: Classified event level.
        """
        if z > THRESHOLDS[3]:
            return 2
        if z > THRESHOLDS[2]:
            return 1
        if z < THRESHOLDS[0]:
            return -1
        if z < THRESHOLDS[1]:
            return -2
        return 0

    def rejudge_based_on_mode(
        self,
        init_judgment: int,
        x0: float = 0.05,
        k: float = 27.72,
    ) -> int:
        """Re-evaluate judgment based on collective memory (mode).

        Args:
            init_judgment (int): Initial judgment based on personal memory.
            x0 (float): Offset for sigmoid probability.
            k (float): Steepness for sigmoid probability.
        Returns:
            int: Final judgment after possible adjustment.
        """
        last_event_year = self.model.get_last_event_year_for_level(init_judgment)
        if last_event_year is None:
            return init_judgment
        if (self.time.tick - self.age) > last_event_year:
            return init_judgment
        climate_then = self.model.climate_series.loc[last_event_year]
        climate_now = self.model.climate
        diff = climate_now - climate_then
        prob_to_adjust = 1 / (1 + np.exp(-k * (np.abs(diff) - x0)))
        if self.random.random() > prob_to_adjust:
            return init_judgment
        if diff > 0:
            return min(init_judgment + 1, 2)
        if diff < 0:
            return max(init_judgment - 1, -2)
        return init_judgment

    def step(self) -> None:
        """Update observer state: age, memory, and possibly record an event."""
        self.age += 1
        climate = self.model.climate
        self._memory.append(climate)
        if self.age < self._min_age:
            return
        z = self.perception(climate)
        if self.write_down(z):
            extreme = self.judge_extreme(z)
            if self.model.p.get("rejudge", True):
                extreme = self.rejudge_based_on_mode(extreme)
            self.model.write_down(extreme)
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
    path = get_output_dir()
    final_corr_datasets = []
    for i in range(cfg.how.repeat):
        model = Model(parameters={"model": cfg.how}, run_id=i, outpath=path)
        model.run_model()
        final_corr_datasets.append(model.final_corr)
    pd.concat(final_corr_datasets, axis=1).to_csv(path / "correlations.csv")


if __name__ == "__main__":
    repeat_run()
