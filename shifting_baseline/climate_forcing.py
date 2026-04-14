"""Discrete climate-forcing generators used by the ABM.

Each generator returns a 1-D ``numpy`` array of length ``n_ticks``. Ticks are
sub-annual units when ``step_per_year > 1`` and coincide with years otherwise.
The forcing is intentionally a *discrete sequence*, not a continuous physical
climate trajectory.

Three forcings are supported:

- ``iid``: independent Gaussian draws at each tick.
- ``ar1``: AR(1) process with persistence ``phi`` driven by Gaussian innovations.
- ``trend_plus_noise``: linear trend (per year) plus Gaussian noise.

The module also exposes :func:`sigma_tick_from_sigma_year` which converts a
yearly-scale standard deviation into the tick-scale innovation sigma, so that
subannual and annual forcings remain roughly comparable after yearly
aggregation. The conversion is exact only for ``iid``; see function docstring.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SubannualAggregation = Literal["mean", "sum", "last"]
ClimateProcess = Literal["iid", "ar1", "trend_plus_noise"]


def sigma_tick_from_sigma_year(
    *,
    sigma_year: float,
    step_per_year: int,
    subannual_aggregation: SubannualAggregation,
) -> float:
    """Convert yearly-scale sigma to tick-scale sigma for IID innovations.

    For IID tick-level noise aggregated to years:

    - ``mean`` of ``k`` IID samples has variance ``sigma_tick^2 / k``, so
      ``sigma_tick = sigma_year * sqrt(k)``.
    - ``sum`` of ``k`` IID samples has variance ``k * sigma_tick^2``, so
      ``sigma_tick = sigma_year / sqrt(k)``.
    - ``last`` of ``k`` IID samples has variance ``sigma_tick^2``, so
      ``sigma_tick = sigma_year``.

    For AR(1) this scaling is *approximate* because stationary variance is
    ``sigma_innov^2 / (1 - phi^2)`` and yearly aggregation of correlated
    samples does not divide variance by ``k``. For ``trend_plus_noise`` the
    scaling applies to the noise component only.

    Args:
        sigma_year: Target yearly-scale standard deviation.
        step_per_year: Number of ticks per year, ``>= 1``.
        subannual_aggregation: Aggregation used to map ticks to years.

    Returns:
        Tick-scale standard deviation.
    """
    if step_per_year < 1:
        raise ValueError("step_per_year must be >= 1")
    if step_per_year == 1:
        return sigma_year
    if subannual_aggregation == "mean":
        return sigma_year * (step_per_year**0.5)
    if subannual_aggregation == "sum":
        return sigma_year / (step_per_year**0.5)
    if subannual_aggregation == "last":
        return sigma_year
    raise ValueError("subannual_aggregation must be one of {'mean', 'sum', 'last'}")


def generate(
    process: ClimateProcess,
    n_ticks: int,
    *,
    sigma: float,
    step_per_year: int = 1,
    phi: float = 0.0,
    trend_per_year: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a discrete climate-forcing series.

    Args:
        process: Forcing type.
        n_ticks: Number of ticks to generate.
        sigma: Tick-scale innovation standard deviation. Callers that think
            in yearly units should first run the value through
            :func:`sigma_tick_from_sigma_year`.
        step_per_year: Ticks per year. Only affects ``trend_plus_noise``,
            where the trend is expressed per year and converted to per tick.
        phi: AR(1) persistence, used only by ``ar1``.
        trend_per_year: Linear trend per year, used only by
            ``trend_plus_noise``.
        rng: Optional random generator for reproducibility.

    Returns:
        Forcing series of length ``n_ticks``.
    """
    if n_ticks <= 0:
        raise ValueError("n_ticks must be > 0")
    generator = rng if rng is not None else np.random.default_rng()

    if process == "iid":
        return generator.normal(0.0, sigma, n_ticks)

    if process == "ar1":
        series = np.zeros(n_ticks, dtype=float)
        innovations = generator.normal(0.0, sigma, n_ticks)
        for tick in range(1, n_ticks):
            series[tick] = phi * series[tick - 1] + innovations[tick]
        return series

    if process == "trend_plus_noise":
        year_scale = np.arange(n_ticks) / step_per_year
        trend = trend_per_year * year_scale
        noise = generator.normal(0.0, sigma, n_ticks)
        return trend + noise

    raise ValueError(f"Unsupported climate process: {process!r}")
