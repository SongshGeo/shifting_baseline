"""Discrete climate-forcing generators used by the ABM.

Each generator returns a 1-D ``numpy`` array of length ``n_ticks``. Ticks are
sub-annual units when ``step_per_year > 1`` and coincide with years otherwise.
The forcing is intentionally a *discrete sequence*, not a continuous physical
climate trajectory.

Four forcings are supported:

- ``iid``: independent Gaussian draws at each tick — the "null" climate with
  no temporal structure. This is the most conservative baseline: any pattern
  that emerges from the ABM under iid forcing must be explained by the agent
  memory mechanism, not by climate persistence (Hasselmann 1976).
- ``ar1``: AR(1) process ``x_t = φ·x_{t-1} + ε_t`` with persistence ``φ``
  driven by Gaussian innovations. Captures the first-order autocorrelation
  structure that is typical of climate time series at annual resolution
  (Hasselmann 1976; Frankignoul & Hasselmann 1977; Mudelsee 2002).
- ``trend_plus_noise``: linear trend (per year) plus iid Gaussian noise.
- ``ar1_trend``: AR(1) persistence *plus* linear trend — the most realistic
  simple climate process, combining both features the reviewer requested.

The module also exposes :func:`sigma_tick_from_sigma_year` which converts a
yearly-scale standard deviation into the tick-scale innovation sigma, so that
subannual and annual forcings remain roughly comparable after yearly
aggregation. The conversion is exact only for ``iid``; see function docstring.

:func:`estimate_ar1_phi` provides empirical estimation of the AR(1) persistence
parameter from observed data, grounding the model's ``φ`` in real climate
statistics rather than arbitrary choices.

References
----------
- Hasselmann, K. (1976). Stochastic climate models Part I. Theory.
  *Tellus*, 28(6), 473–485. doi:10.3402/tellusa.v28i6.11316
- Frankignoul, C. & Hasselmann, K. (1977). Stochastic climate models, Part II.
  *Tellus*, 29(4), 289–305.
- Mudelsee, M. (2002). TAUEST: a computer program for estimating persistence
  in unevenly spaced weather/climate time series.
  *Computers & Geosciences*, 28(1), 69–72.
- Wilks, D.S. & Wilby, R.L. (1999). The weather generation game: a review of
  stochastic weather models. *Progress in Physical Geography*, 23(3), 329–357.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shifting_baseline.utils.types import ClimateProcess, SubannualAggregation

import numpy as np


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
    samples does not divide variance by ``k``. For ``trend_plus_noise`` and
    ``ar1_trend`` the scaling applies to the noise component only.

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


def estimate_ar1_phi(
    series: np.ndarray,
    *,
    method: Literal["acf", "ols"] = "acf",
) -> dict[str, float]:
    """Estimate the AR(1) persistence parameter φ from an observed series.

    Two methods:

    - ``acf``: lag-1 sample autocorrelation (fast, standard).
    - ``ols``: ordinary least-squares regression of ``x_t`` on ``x_{t-1}``
      (gives identical point estimate for large samples).

    Both methods derive ``sigma_innov`` from the stationary standard
    deviation via ``sigma_innov = sigma_stationary * sqrt(1 - phi^2)``
    rather than from regression residuals.

    Use this to ground the ABM's ``climate_phi`` in empirical data rather
    than an arbitrary choice — as recommended by Mudelsee (2002).

    Args:
        series: 1-D array of observed values (e.g. annual reconstruction).
        method: Estimation method.

    Returns:
        Dictionary with ``phi`` (persistence), ``sigma_innov`` (innovation
        standard deviation), and ``sigma_stationary`` (stationary standard
        deviation = sigma_innov / sqrt(1 - phi^2)).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        raise ValueError("Need at least 3 finite observations")

    if method == "acf":
        x_demean = x - x.mean()
        c0 = np.dot(x_demean, x_demean) / len(x)
        c1 = np.dot(x_demean[:-1], x_demean[1:]) / len(x)
        phi = c1 / c0 if c0 > 0 else 0.0
    elif method == "ols":
        y = x[1:]
        x_lag = x[:-1]
        x_lag_demean = x_lag - x_lag.mean()
        y_demean = y - y.mean()
        phi = (
            np.dot(x_lag_demean, y_demean) / np.dot(x_lag_demean, x_lag_demean)
            if np.dot(x_lag_demean, x_lag_demean) > 0
            else 0.0
        )
    else:
        raise ValueError("method must be 'acf' or 'ols'")

    phi = float(np.clip(phi, -0.999, 0.999))
    sigma_stationary = float(x.std(ddof=1))
    sigma_innov = sigma_stationary * np.sqrt(max(1 - phi**2, 1e-12))

    return {
        "phi": phi,
        "sigma_innov": float(sigma_innov),
        "sigma_stationary": sigma_stationary,
    }


def _ar1_stationary_init(
    phi: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """Draw initial value from the AR(1) stationary distribution.

    For |φ| < 1 the stationary distribution is N(0, σ²/(1−φ²)).
    """
    if abs(phi) >= 1.0:
        return rng.normal(0.0, sigma)
    stationary_std = sigma / np.sqrt(1 - phi**2)
    return float(rng.normal(0.0, stationary_std))


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
        process: Forcing type — one of ``iid``, ``ar1``, ``trend_plus_noise``,
            ``ar1_trend``.
        n_ticks: Number of ticks to generate.
        sigma: Tick-scale innovation standard deviation. Callers that think
            in yearly units should first run the value through
            :func:`sigma_tick_from_sigma_year`.
        step_per_year: Ticks per year. Only affects ``trend_plus_noise`` and
            ``ar1_trend``, where the trend is expressed per year and converted
            to per tick.
        phi: AR(1) persistence, used by ``ar1`` and ``ar1_trend``.
        trend_per_year: Linear trend per year, used by ``trend_plus_noise``
            and ``ar1_trend``.
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
        # Initialize from stationary distribution to avoid transient bias
        series[0] = _ar1_stationary_init(phi, sigma, generator)
        for tick in range(1, n_ticks):
            series[tick] = phi * series[tick - 1] + innovations[tick]
        return series

    if process == "trend_plus_noise":
        year_scale = np.arange(n_ticks) / step_per_year
        trend = trend_per_year * year_scale
        noise = generator.normal(0.0, sigma, n_ticks)
        return trend + noise

    if process == "ar1_trend":
        # Deterministic linear trend + zero-mean AR(1) residuals.
        #
        # x_t = trend(t) + r_t,  where r_t = φ·r_{t-1} + ε_t.
        #
        # Putting the trend constant *inside* the AR(1) recursion would
        # make E[x_t] converge to c/(1−φ) (a constant), not grow linearly.
        # The correct decomposition keeps trend outside the recursion so
        # that E[x_t] = trend_per_year · (t / step_per_year).
        year_scale = np.arange(n_ticks) / step_per_year
        trend = trend_per_year * year_scale
        residuals = np.zeros(n_ticks, dtype=float)
        innovations = generator.normal(0.0, sigma, n_ticks)
        residuals[0] = _ar1_stationary_init(phi, sigma, generator)
        for tick in range(1, n_ticks):
            residuals[tick] = phi * residuals[tick - 1] + innovations[tick]
        return trend + residuals

    raise ValueError(f"Unsupported climate process: {process!r}")
