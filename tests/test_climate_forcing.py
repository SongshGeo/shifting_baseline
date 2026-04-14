"""Tests for :mod:`shifting_baseline.climate_forcing`."""

from __future__ import annotations

import numpy as np
import pytest

from shifting_baseline.climate_forcing import generate, sigma_tick_from_sigma_year


def test_sigma_scaling_mean_preserves_yearly_variance():
    rng = np.random.default_rng(0)
    step = 4
    sigma_year = 1.0
    sigma_tick = sigma_tick_from_sigma_year(
        sigma_year=sigma_year, step_per_year=step, subannual_aggregation="mean"
    )
    ticks = generate("iid", 200_000, sigma=sigma_tick, rng=rng)
    yearly = ticks.reshape(-1, step).mean(axis=1)
    assert np.isclose(yearly.std(ddof=1), sigma_year, rtol=0.02)


def test_sigma_scaling_sum_preserves_yearly_variance():
    rng = np.random.default_rng(1)
    step = 4
    sigma_year = 1.0
    sigma_tick = sigma_tick_from_sigma_year(
        sigma_year=sigma_year, step_per_year=step, subannual_aggregation="sum"
    )
    ticks = generate("iid", 200_000, sigma=sigma_tick, rng=rng)
    yearly = ticks.reshape(-1, step).sum(axis=1)
    assert np.isclose(yearly.std(ddof=1), sigma_year, rtol=0.02)


def test_sigma_scaling_step_one_is_identity():
    assert (
        sigma_tick_from_sigma_year(
            sigma_year=1.5, step_per_year=1, subannual_aggregation="mean"
        )
        == 1.5
    )


def test_ar1_stationary_variance():
    rng = np.random.default_rng(2)
    phi = 0.6
    sigma = 1.0
    ticks = generate("ar1", 100_000, sigma=sigma, phi=phi, rng=rng)
    expected_var = sigma**2 / (1 - phi**2)
    # Drop burn-in to avoid the zero-initialized transient.
    assert np.isclose(ticks[1000:].var(ddof=1), expected_var, rtol=0.05)


def test_trend_plus_noise_recovers_slope():
    rng = np.random.default_rng(3)
    step = 4
    n_years = 500
    trend = 0.05
    ticks = generate(
        "trend_plus_noise",
        n_years * step,
        sigma=0.1,
        step_per_year=step,
        trend_per_year=trend,
        rng=rng,
    )
    yearly = ticks.reshape(-1, step).mean(axis=1)
    slope = np.polyfit(np.arange(n_years), yearly, 1)[0]
    assert np.isclose(slope, trend, rtol=0.05)


def test_generate_rejects_unknown_process():
    with pytest.raises(ValueError):
        generate("brownian", 10, sigma=1.0)  # type: ignore[arg-type]


def test_generate_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        generate("iid", 0, sigma=1.0)


def test_sigma_scaling_rejects_bad_aggregation():
    with pytest.raises(ValueError):
        sigma_tick_from_sigma_year(
            sigma_year=1.0, step_per_year=4, subannual_aggregation="median"  # type: ignore[arg-type]
        )
