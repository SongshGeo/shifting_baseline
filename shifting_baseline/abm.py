#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

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

try:  # optional run-completion notifier (author's private package)
    from twist_academic import notify
except ImportError:

    def notify(*_args, **_kwargs) -> None:
        """No-op fallback used when ``twist_academic`` is not installed."""


from shifting_baseline.calibration import MismatchReport
from shifting_baseline.climate_forcing import generate as generate_climate_forcing
from shifting_baseline.climate_forcing import sigma_tick_from_sigma_year
from shifting_baseline.compare import compare_corr_2d
from shifting_baseline.filters import (
    calc_std_deviation,
    classify,
    classify_single_value,
)
from shifting_baseline.utils.calc import rand_generate_from_std_levels

if TYPE_CHECKING:
    from shifting_baseline.utils.types import CorrFunc
    from shifting_baseline.utils.types import SubannualAggregation

from shifting_baseline.utils.log import get_logger

MAX_AGE: int = 40  # 主体气候观察者的最大年龄

# 使用主logger，避免重复设置
log = get_logger()


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
        self._step_per_year: int = int(self.p.get("step_per_year", 1))
        if self._step_per_year < 1:
            raise ValueError("step_per_year must be >= 1")
        raw_agg = self.p.get("subannual_aggregation", "mean")
        if raw_agg not in {"mean", "sum", "last"}:
            raise ValueError(
                "subannual_aggregation must be one of {'mean', 'sum', 'last'}"
            )
        self._subannual_aggregation: SubannualAggregation = raw_agg
        self._climate_process: str = self.p.get("climate_process", "ar1")
        # ``climate_sigma`` is interpreted as yearly-scale sigma. When
        # ``step_per_year > 1`` we rescale to tick-scale internally so that
        # annual and subannual runs remain comparable after yearly aggregation.
        self._climate_sigma_year: float = float(self.p.get("climate_sigma", 1.0))
        self._climate_sigma_tick: float = sigma_tick_from_sigma_year(
            sigma_year=self._climate_sigma_year,
            step_per_year=self._step_per_year,
            subannual_aggregation=self._subannual_aggregation,
        )
        self._climate_phi: float = float(self.p.get("climate_phi", 0.5))
        self._climate_trend: float = float(self.p.get("climate_trend", 0.0))
        # Years to simulate
        years: int = self.p.get("years", 100)
        # Maximum age for an observer
        self._max_age_years: int = self.p.get("max_age", 40)
        self._new_agents: int = self.p.get("new_agents", 5)
        # Minimum age for recording events
        self._min_age_years: int = self.p.get("min_age", 10)
        self._max_age_ticks: int = self._max_age_years * self._step_per_year
        self._min_age_ticks: int = self._min_age_years * self._step_per_year
        # Cache for collective memory
        self._collective_cache: Optional[pd.Series] = None
        self._collective_cache_tick: int = -1
        # Per-tick cache of the cohort-level (mean, std) used by every observer
        # under the "collective" baseline. The values are the same for all
        # observers in the same tick — recomputing them per agent burns ~ms
        # per call × ~10³ agents × ~10² ticks under the SA upper corner.
        self._collective_baseline_cache: Optional[tuple[float, float]] = None
        self._collective_baseline_cache_tick: int = -1
        # Per-year frozen mean of `rand_generate_from_std_levels(archive[k])`.
        # Once tick advances past year k, archive[k] is permanently fixed
        # (`archive_it` only writes to the current tick), so its random-sample
        # mean should also be fixed. Without freezing, every tick re-samples
        # the whole archive, which is the dominant O(N²) cost in collective
        # baseline runs.
        self._frozen_year_means: dict[int, float] = {}
        self.spin_up_years: int = self._new_agents * (
            self._max_age_years - self._min_age_years + 1
        )
        self.spin_up_ticks: int = self.spin_up_years * self._step_per_year
        # Total simulation years
        self._years: int = years + self.spin_up_years
        self._ticks: int = self._years * self._step_per_year
        # Climate time series
        self._climate: np.ndarray = self._generate_climate_series(self._ticks)
        # Archive of recorded events per year
        self._archive: dict[int, list[int]] = {i: [] for i in range(self._ticks)}
        log.info(f"运行模式: {self.p.get('mode', 'exp')}")

    def _generate_climate_series(self, n_ticks: int) -> np.ndarray:
        """Generate discrete climate forcing for the current scenario.

        Delegates to :mod:`shifting_baseline.climate_forcing`. Sigma is
        passed in tick-scale after internal rescaling from the yearly-scale
        ``climate_sigma`` config value.
        """
        return generate_climate_forcing(
            self._climate_process,  # type: ignore[arg-type]
            n_ticks,
            sigma=self._climate_sigma_tick,
            step_per_year=self._step_per_year,
            phi=self._climate_phi,
            trend_per_year=self._climate_trend,
        )

    def _aggregate_to_yearly(self, series: pd.Series) -> pd.Series:
        """Aggregate a tick-indexed series to a year-indexed series.

        When ``step_per_year == 1`` tick and year indices coincide and the
        series is returned unchanged. Otherwise ticks are grouped by
        ``tick // step_per_year`` and reduced by the configured aggregator.

        Args:
            series: Tick-level series indexed by tick id.

        Returns:
            Year-level aggregated series indexed by year id.
        """
        if self._step_per_year == 1:
            return series
        year_index = series.index // self._step_per_year
        if self._subannual_aggregation == "mean":
            return series.groupby(year_index).mean()
        if self._subannual_aggregation == "sum":
            return series.groupby(year_index).sum()
        if self._subannual_aggregation == "last":
            return series.groupby(year_index).last()
        raise ValueError("subannual_aggregation must be one of {'mean', 'sum', 'last'}")

    @property
    def is_nan(self) -> bool:
        """Check if the collective memory is all NaN."""
        return self.collective_memory_climate.isna().all()

    @property
    def climate_now(self) -> float:
        """Current climate forcing at the current tick.

        The model always reads climate as a discrete forcing sequence.
        This value is not assumed to be a continuous physical process.

        Returns:
            float: Current climate value.
        """
        return self._climate[self.time.tick]

    @cached_property
    def climate_series(self) -> pd.Series:
        """Full tick-level climate forcing series.

        Returns:
            pd.Series: Climate values indexed by model tick.
        """
        return pd.Series(self._climate, index=range(self._ticks))

    @property
    def collective_memory_climate(self) -> pd.Series:
        """Mean of recorded events per year.

        Two-level caching:
        - **Outer (tick-scoped):** the assembled Series is reused for all
          accesses within the same tick.
        - **Inner (year-scoped, permanent):** for any year ``k`` that is
          strictly past (``k < current_tick``), ``archive[k]`` is immutable
          (``archive_it`` only appends to ``self.time.tick``), so its
          random-sample mean is drawn once and frozen in
          ``self._frozen_year_means``. This collapses the per-tick rebuild
          from O(years_elapsed × records) to O(1 + new records this tick).
          The current-tick year is *not* frozen — it is still being written
          to, so its sample is drawn fresh on each cache miss.
        """
        current_tick = self.time.tick
        if (
            self._collective_cache is not None
            and self._collective_cache_tick == current_tick
        ):
            return self._collective_cache

        out: dict[int, float] = {}
        frozen = self._frozen_year_means
        for k, v in self._archive.items():
            if not v:
                continue
            if k < current_tick:
                cached = frozen.get(k)
                if cached is None:
                    cached = float(rand_generate_from_std_levels(np.array(v)).mean())
                    frozen[k] = cached
                out[k] = cached
            else:
                out[k] = float(rand_generate_from_std_levels(np.array(v)).mean())
        series = pd.Series(out)
        self._collective_cache = series
        self._collective_cache_tick = current_tick
        return series

    @property
    def collective_baseline_stats(self) -> tuple[float, float]:
        """Cohort-level (mean, std) of the collective memory at this tick.

        Cached per tick. Every observer under the "collective" baseline reads
        this same scalar pair — computing it once instead of N_agents times
        is a pure performance fix (no numerical change).
        """
        tick = self.time.tick
        if (
            self._collective_baseline_cache is not None
            and self._collective_baseline_cache_tick == tick
        ):
            return self._collective_baseline_cache
        series = self.collective_memory_climate
        mean_val = float(series.mean()) if len(series) else float("nan")
        std_val = float(series.std()) if len(series) else float("nan")
        self._collective_baseline_cache = (mean_val, std_val)
        self._collective_baseline_cache_tick = tick
        return self._collective_baseline_cache

    def collective_baseline_stats_window(
        self, start_tick: int, end_tick: int
    ) -> tuple[float, float]:
        """(mean, std) of collective memory over an inclusive tick window.

        Used by the ``collective_lifetime`` baseline: every observer reads
        from the same societal archive, but only the portion written during
        their own lifetime (``start_tick`` = current tick − age).
        """
        if end_tick < start_tick:
            return float("nan"), float("nan")
        series = self.collective_memory_climate
        window = series[(series.index >= start_tick) & (series.index <= end_tick)]
        if len(window) == 0:
            return float("nan"), float("nan")
        return float(window.mean()), float(window.std())

    @cached_property
    def model_baseline_stats(self) -> tuple[float, float]:
        """(mean, std) of the full climate forcing series; constant for the run.

        Cached once because the climate series doesn't change after init.
        """
        return float(self.climate_series.mean()), float(self.climate_series.std())

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
        loss_rate: float = self.p.get("loss_rate", 0.4)
        if np.random.random() < loss_rate:
            return
        self._archive[self.time.tick].append(extreme)
        # Cache is tick-based, so no need to clear unless logic changes.

    @property
    def climate_df(self) -> pd.DataFrame:
        """Model-vs-memory climate table after spin-up.

        When ``step_per_year > 1`` this method aggregates tick-level series
        to year-level by ``subannual_aggregation``.

        Returns:
            pd.DataFrame: DataFrame with columns 'climate' and 'collective_memory_climate'.

        Raises:
            ValueError: If the collective memory is all NaN.
        """
        if self.is_nan:
            raise ValueError("Collective memory is all NaN, did you run the model?")
        # Aggregate the full series to yearly first, then drop spin-up years.
        # Slicing before aggregation caused groupby to emit NaN year buckets
        # for the spin-up range and shortened the effective analysis window.
        climate = self._aggregate_to_yearly(self.climate_series)
        collective = self._aggregate_to_yearly(self.collective_memory_climate).reindex(
            climate.index
        )
        climate = climate.loc[self.spin_up_years :]
        collective = collective.loc[self.spin_up_years :]
        return pd.DataFrame(
            {
                "climate": climate,
                "collective_memory_climate": collective,
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
        if self.time.tick == self._ticks - 1:
            self.running = False
        self.agents.new(
            ClimateObserver,
            self._new_agents,
            max_age=self._max_age_ticks,
            min_age=self._min_age_ticks,
        )
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

    def __init__(
        self,
        *args,
        max_age: int = MAX_AGE,
        min_age: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._memory: deque = deque(maxlen=max_age)
        self._max_age: int = max_age
        self._min_age: int = min_age

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
        Here, we have four types of baseline:
        - personal: the observer's personal memory
        - model: the model's climate time series (objective climate)
        - collective: the collective memory of the model (collective memory climate)
        - collective_lifetime: same societal archive as collective, but each
          observer only uses records from their own lifetime window
        We use the baseline to re-calculate the z-score of the current climate.
        The hypothesis is that the observer will compare the current climatic extreme with the baseline.

        Args:
            climate (float): Current climate Z-score value.
        Returns:
            float: Z-score of the current climate.
        """
        # Personal baseline (per-agent state, can't be hoisted)
        if self.model.p.memory_baseline == "personal":
            baseline = self.memory.mean()
            std = self.memory.std()
        # Model baseline (constant across the run; cached once on the model)
        elif self.model.p.memory_baseline == "model":
            baseline, std = self.model.model_baseline_stats
        # Collective baseline (same scalar for every observer in this tick)
        elif self.model.p.memory_baseline == "collective":
            baseline, std = self.model.collective_baseline_stats
        # Collective archive, windowed to the observer's lifetime
        elif self.model.p.memory_baseline == "collective_lifetime":
            tick = self.model.time.tick
            start_tick = tick - self.age()
            baseline, std = self.model.collective_baseline_stats_window(
                start_tick, tick
            )
        else:
            raise ValueError("Invalid memory baseline")
        # Handle NaN values
        if np.isnan(baseline):
            baseline = 0
        if np.isnan(std):
            std = 1
        # Calculate the z-score of the current climate
        return (climate - baseline) / std

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
    log.info(f"ABM 模型开始运行: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        repeat_run()
        notify("✅ Shifting Baseline ABM 模型运行成功完成")
        log.info(
            f"✅ Shifting Baseline ABM 模型运行成功完成, 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:  # pylint: disable=broad-except # 需要捕获所有异常以发送邮件通知
        log.error(f"❌ Shifting Baseline ABM 模型运行失败: {e}")
        notify("❌ Shifting Baseline ABM 模型运行失败")
