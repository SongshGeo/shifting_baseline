"""Tests for :mod:`shifting_baseline.climate_forcing`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import polynomial as P

from shifting_baseline.climate_forcing import (
    _ar1_stationary_init,
    estimate_ar1_phi,
    generate,
    sigma_tick_from_sigma_year,
)

# ---------------------------------------------------------------------------
# generate — basic contract
# ---------------------------------------------------------------------------

PROCESSES = ["iid", "ar1", "trend_plus_noise", "ar1_trend"]


@pytest.mark.parametrize("process", PROCESSES)
def test_generate_length(process):
    """Output has the requested length."""
    n = 200
    x = generate(
        process,
        n,
        sigma=1.0,
        phi=0.5,
        trend_per_year=0.01,
        step_per_year=1,
        rng=np.random.default_rng(0),
    )
    assert x.shape == (n,)


@pytest.mark.parametrize("process", PROCESSES)
def test_generate_reproducible(process):
    """Same seed → identical output."""
    kw = dict(sigma=1.0, phi=0.5, trend_per_year=0.01, step_per_year=1)
    a = generate(process, 100, rng=np.random.default_rng(42), **kw)
    b = generate(process, 100, rng=np.random.default_rng(42), **kw)
    np.testing.assert_array_equal(a, b)


def test_generate_bad_process():
    with pytest.raises(ValueError, match="Unsupported"):
        generate("bogus", 10, sigma=1.0)


def test_generate_zero_length():
    with pytest.raises(ValueError, match="n_ticks"):
        generate("iid", 0, sigma=1.0)


# ---------------------------------------------------------------------------
# iid
# ---------------------------------------------------------------------------


def test_iid_mean_std():
    """iid output should have mean ≈ 0 and std ≈ sigma."""
    x = generate("iid", 100_000, sigma=2.0, rng=np.random.default_rng(7))
    assert abs(x.mean()) < 0.05
    assert abs(x.std(ddof=1) - 2.0) < 0.05


def test_iid_no_autocorrelation():
    """iid lag-1 ACF should be near zero."""
    x = generate("iid", 50_000, sigma=1.0, rng=np.random.default_rng(3))
    acf1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(acf1) < 0.02


# ---------------------------------------------------------------------------
# ar1
# ---------------------------------------------------------------------------


def test_ar1_stationary_variance():
    """AR(1) variance should converge to σ²/(1−φ²)."""
    phi = 0.6
    sigma = 1.0
    x = generate("ar1", 100_000, sigma=sigma, phi=phi, rng=np.random.default_rng(0))
    expected_var = sigma**2 / (1 - phi**2)
    assert abs(x.var(ddof=1) - expected_var) / expected_var < 0.03


def test_ar1_lag1_acf():
    """AR(1) lag-1 ACF should be ≈ φ."""
    phi = 0.7
    x = generate("ar1", 100_000, sigma=1.0, phi=phi, rng=np.random.default_rng(1))
    acf1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(acf1 - phi) < 0.02


def test_ar1_phi_zero_is_iid():
    """AR(1) with φ=0 should behave like iid."""
    x = generate("ar1", 50_000, sigma=1.0, phi=0.0, rng=np.random.default_rng(5))
    acf1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(acf1) < 0.02
    assert abs(x.std(ddof=1) - 1.0) < 0.03


# ---------------------------------------------------------------------------
# trend_plus_noise
# ---------------------------------------------------------------------------


def test_trend_plus_noise_slope():
    """Mean trajectory should have the requested slope."""
    trend = 0.02
    n = 10_000
    n_runs = 500
    means = np.zeros(n)
    for i in range(n_runs):
        means += generate(
            "trend_plus_noise",
            n,
            sigma=1.0,
            trend_per_year=trend,
            rng=np.random.default_rng(i),
        )
    means /= n_runs
    coeffs = P.polyfit(np.arange(n), means, 1)
    assert abs(coeffs[1] - trend) / trend < 0.05


# ---------------------------------------------------------------------------
# ar1_trend — the bug-regression test
# ---------------------------------------------------------------------------


def test_ar1_trend_mean_is_linear():
    """E[x_t] of ar1_trend must grow linearly, NOT converge to c/(1−φ).

    This is the regression test for the bug where putting the trend constant
    inside the AR(1) recursion made the expected value saturate at a constant
    instead of growing linearly.
    """
    phi = 0.6
    trend = 0.05
    n = 300
    n_runs = 3000

    trajectories = np.array(
        [
            generate(
                "ar1_trend",
                n,
                sigma=1.0,
                phi=phi,
                trend_per_year=trend,
                step_per_year=1,
                rng=np.random.default_rng(i),
            )
            for i in range(n_runs)
        ]
    )
    mean_traj = trajectories.mean(axis=0)

    # Linear fit of the ensemble mean
    coeffs = P.polyfit(np.arange(n), mean_traj, 1)
    fitted_slope = coeffs[1]

    # The slope should be ≈ trend_per_year, not ≈ 0
    assert abs(fitted_slope - trend) / trend < 0.05, (
        f"ar1_trend slope = {fitted_slope:.5f}, expected ≈ {trend}. "
        f"Bug: trend saturated to constant c/(1-φ) = {trend/(1-phi):.4f}?"
    )

    # Also verify that late-time mean is NOT the buggy constant c/(1−φ)
    buggy_mean = trend / (1 - phi)
    late_mean = mean_traj[200:].mean()
    expected_late_mean = trend * np.mean(np.arange(200, n))
    assert abs(late_mean - expected_late_mean) < 1.0, (
        f"Late mean = {late_mean:.3f}, expected ≈ {expected_late_mean:.1f}, "
        f"buggy would be ≈ {buggy_mean:.4f}"
    )


def test_ar1_trend_residuals_are_ar1():
    """After removing the linear trend, residuals should have AR(1) structure."""
    phi = 0.5
    trend = 0.01
    n = 50_000
    x = generate(
        "ar1_trend",
        n,
        sigma=1.0,
        phi=phi,
        trend_per_year=trend,
        step_per_year=1,
        rng=np.random.default_rng(42),
    )

    # Remove linear trend
    detrended = x - trend * np.arange(n)

    # lag-1 ACF of residuals should be ≈ φ
    acf1 = np.corrcoef(detrended[:-1], detrended[1:])[0, 1]
    assert abs(acf1 - phi) < 0.02


def test_ar1_trend_subannual():
    """ar1_trend with step_per_year > 1 should still have linear E[x_t]."""
    phi = 0.5
    trend = 0.02
    step = 4
    n_ticks = 400 * step  # 400 years
    n_runs = 1000

    trajectories = np.array(
        [
            generate(
                "ar1_trend",
                n_ticks,
                sigma=1.0,
                phi=phi,
                trend_per_year=trend,
                step_per_year=step,
                rng=np.random.default_rng(i),
            )
            for i in range(n_runs)
        ]
    )
    mean_traj = trajectories.mean(axis=0)

    # Linear fit — slope should be trend_per_year / step_per_year (per tick)
    coeffs = P.polyfit(np.arange(n_ticks), mean_traj, 1)
    expected_slope_per_tick = trend / step
    assert abs(coeffs[1] - expected_slope_per_tick) / expected_slope_per_tick < 0.10


# ---------------------------------------------------------------------------
# _ar1_stationary_init
# ---------------------------------------------------------------------------


def test_stationary_init_distribution():
    """Initial draws should have variance ≈ σ²/(1−φ²)."""
    phi = 0.7
    sigma = 1.0
    rng = np.random.default_rng(99)
    draws = [_ar1_stationary_init(phi, sigma, rng) for _ in range(10_000)]
    expected_std = sigma / np.sqrt(1 - phi**2)
    assert abs(np.std(draws, ddof=1) - expected_std) / expected_std < 0.05


# ---------------------------------------------------------------------------
# estimate_ar1_phi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["acf", "ols"])
def test_estimate_ar1_phi_recovery(method):
    """estimate_ar1_phi should recover the true φ from a long AR(1) sample."""
    true_phi = 0.6
    x = generate("ar1", 50_000, sigma=1.0, phi=true_phi, rng=np.random.default_rng(42))
    est = estimate_ar1_phi(x, method=method)
    assert abs(est["phi"] - true_phi) < 0.02
    assert est["sigma_innov"] > 0
    assert est["sigma_stationary"] > est["sigma_innov"]


def test_estimate_ar1_phi_iid():
    """For iid data, estimated φ should be near 0."""
    x = generate("iid", 50_000, sigma=1.0, rng=np.random.default_rng(7))
    est = estimate_ar1_phi(x)
    assert abs(est["phi"]) < 0.02


def test_estimate_ar1_phi_too_short():
    with pytest.raises(ValueError, match="at least 3"):
        estimate_ar1_phi(np.array([1.0, 2.0]))


def test_estimate_ar1_phi_bad_method():
    with pytest.raises(ValueError, match="method"):
        estimate_ar1_phi(np.ones(10), method="magic")


# ---------------------------------------------------------------------------
# sigma_tick_from_sigma_year
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg", ["mean", "sum", "last"])
@pytest.mark.parametrize("step", [1, 2, 4, 12])
def test_sigma_rescaling(agg, step):
    """Yearly-aggregated std should match target regardless of step_per_year."""
    sigma_year = 1.0
    s_tick = sigma_tick_from_sigma_year(
        sigma_year=sigma_year, step_per_year=step, subannual_aggregation=agg
    )
    x = generate("iid", 200_000 * step, sigma=s_tick, rng=np.random.default_rng(99))
    yearly = x.reshape(-1, step)
    if agg == "mean":
        y = yearly.mean(axis=1)
    elif agg == "sum":
        y = yearly.sum(axis=1)
    else:
        y = yearly[:, -1]
    assert abs(y.std(ddof=1) - sigma_year) / sigma_year < 0.01


def test_sigma_rescaling_passthrough():
    """step_per_year=1 should return sigma unchanged."""
    assert (
        sigma_tick_from_sigma_year(
            sigma_year=3.14, step_per_year=1, subannual_aggregation="mean"
        )
        == 3.14
    )


def test_sigma_rescaling_bad_step():
    with pytest.raises(ValueError):
        sigma_tick_from_sigma_year(
            sigma_year=1.0, step_per_year=0, subannual_aggregation="mean"
        )


def test_sigma_rescaling_bad_agg():
    with pytest.raises(ValueError):
        sigma_tick_from_sigma_year(
            sigma_year=1.0, step_per_year=4, subannual_aggregation="median"
        )
