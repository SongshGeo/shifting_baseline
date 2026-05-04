"""Tests for `shifting_baseline.abm` performance caches.

These tests verify the cohort-baseline cache that lifts pandas .mean()/.std()
out of the per-agent perceive() loop. Without this cache, every observer in
every tick recomputes the same scalar from a shared Series, dominating
wall-clock under collective baseline at large agent populations.

The tests are numerical-parity tests, not benchmarks: they assert that the
cached scalar equals the value the old "naive" code path would have produced.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from shifting_baseline.abm import ClimateObservingModel


def _make_cfg(memory_baseline: str = "collective", **overrides) -> DictConfig:
    """Build a small, deterministic cfg under a `model:` key.

    Tiny knobs (years=5, ages 2-4, 3 new agents/tick) keep the test under a
    second while still producing a non-empty archive.
    """
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
        "min_period": 2,
        "corr_method": "kendall",
        "filter_side": "right",
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def test_collective_baseline_stats_matches_naive_mean_std() -> None:
    """Cached cohort scalars must equal the value pandas would compute directly.

    This is the numerical-parity guarantee: switching from per-agent
    .mean()/.std() calls to a model-level cache cannot change behaviour.
    """
    model = ClimateObservingModel(parameters=_make_cfg())
    model.run_model()

    series = model.collective_memory_climate
    expected_mean = float(series.mean())
    expected_std = float(series.std())

    cached_mean, cached_std = model.collective_baseline_stats
    assert cached_mean == expected_mean
    assert cached_std == expected_std


def test_collective_baseline_stats_caches_per_tick() -> None:
    """Same tick → same cached tuple object; cache invalidation forces recompute."""
    model = ClimateObservingModel(parameters=_make_cfg())
    model.run_model()

    first = model.collective_baseline_stats
    second = model.collective_baseline_stats
    # Both calls served from cache → same object identity.
    assert first is second

    # Force invalidation: pretend a new tick arrived.
    model._collective_baseline_cache_tick = -1
    third = model.collective_baseline_stats
    # Different object, but value-equal (model state is unchanged).
    assert third == first
    assert third is not first


def test_collective_baseline_stats_does_not_rebuild_series() -> None:
    """After warming the cache, repeated reads must not touch collective_memory_climate.

    This test guards the perf optimization: if a regression re-enables the
    O(years × n_agents) recomputation, the spy will fire.
    """
    model = ClimateObservingModel(parameters=_make_cfg())
    model.run_model()
    # Warm cache.
    model.collective_baseline_stats

    call_count = {"n": 0}
    # Pull the property descriptor out of __dict__ to dodge mypy's
    # method-assign / attr-defined warnings on `cls.attr.fget` and
    # `cls.attr = property(...)`.
    real_descriptor = ClimateObservingModel.__dict__["collective_memory_climate"]
    real_fget = real_descriptor.fget

    def spy(self):
        call_count["n"] += 1
        return real_fget(self)

    setattr(ClimateObservingModel, "collective_memory_climate", property(spy))
    try:
        for _ in range(50):
            _ = model.collective_baseline_stats
    finally:
        setattr(ClimateObservingModel, "collective_memory_climate", real_descriptor)

    assert call_count["n"] == 0, (
        f"cohort cache regressed: rebuilt collective_memory_climate "
        f"{call_count['n']} times in 50 reads"
    )


def test_model_baseline_stats_matches_climate_series() -> None:
    """`model` baseline is constant; cached scalar equals climate_series.mean/std."""
    model = ClimateObservingModel(parameters=_make_cfg(memory_baseline="model"))
    series = model.climate_series
    cached_mean, cached_std = model.model_baseline_stats
    assert cached_mean == float(series.mean())
    assert cached_std == float(series.std())


def test_collective_baseline_stats_handles_empty_archive() -> None:
    """Before any observer records, both stats are NaN and don't raise."""
    model = ClimateObservingModel(parameters=_make_cfg())
    # Fresh model — tick 0, archive is all-empty lists.
    mean_val, std_val = model.collective_baseline_stats
    # Not asserting NaN explicitly because pd.Series([]).mean() is float('nan');
    # the contract is only "scalars, no exception".
    assert isinstance(mean_val, float)
    assert isinstance(std_val, float)
