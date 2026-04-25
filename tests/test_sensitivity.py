"""Tests for `shifting_baseline.sensitivity`.

Fast tests cover the pure helpers (problem definition, override building,
metric extraction). The integration test marked `slow` actually invokes the
ABM subprocess and is what `reports/run_sensitivity.py smoke` exercises end
to end. Run it with `uv run pytest tests/test_sensitivity.py -m slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.sensitivity import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    build_overrides,
    coerce_value,
    compute_metrics,
    define_problem,
    iter_param_dicts,
    run_one,
)


def test_define_problem_default_returns_all_params() -> None:
    problem = define_problem()
    assert problem["num_vars"] == len(PARAM_NAMES)
    assert tuple(problem["names"]) == PARAM_NAMES
    assert all(lo < hi for lo, hi in problem["bounds"])


def test_define_problem_subset() -> None:
    problem = define_problem(["max_age", "loss_rate"])
    assert problem["num_vars"] == 2
    assert problem["bounds"] == [
        list(PARAM_BOUNDS[n]) for n in ("max_age", "loss_rate")
    ]


def test_coerce_value_casts_integers() -> None:
    assert coerce_value("max_age", 39.7) == 40
    assert coerce_value("new_agents", 4.2) == 4
    assert isinstance(coerce_value("loss_rate", 0.4), float)
    assert coerce_value("loss_rate", 0.4) == pytest.approx(0.4)


def test_build_overrides_has_required_keys() -> None:
    params = {
        "max_age": 50.0,
        "new_agents": 6.0,
        "loss_rate": 0.3,
        "climate_sigma": 1.2,
        "climate_phi": 0.4,
    }
    overrides = build_overrides(
        params,
        memory_baseline="collective",
        repeats=10,
        years=80,
        num_process=2,
        climate_process="ar1",
        extra={"hydra.run.dir": "/tmp/x"},
    )
    joined = "\n".join(overrides)
    assert "model.repeats=10" in joined
    assert "model.memory_baseline=collective" in joined
    assert "model.climate_process=ar1" in joined
    # Integer params must not carry a decimal point.
    assert "model.max_age=50" in joined and "model.max_age=50.0" not in joined
    assert "model.new_agents=6" in joined and "model.new_agents=6.0" not in joined
    # Float params keep float form.
    assert any(o.startswith("model.loss_rate=0.3") for o in overrides)
    assert "hydra.run.dir=/tmp/x" in joined


def test_compute_metrics_basic() -> None:
    # Three replicates, peaks at windows 30, 25, 35.
    idx = pd.Index([10, 20, 25, 30, 35, 40], name="window")
    df = pd.DataFrame(
        {
            "model_a": [0.10, 0.20, 0.25, 0.40, 0.30, 0.20],
            "model_b": [0.05, 0.18, 0.45, 0.35, 0.20, 0.10],
            "model_c": [0.10, 0.15, 0.20, 0.30, 0.50, 0.25],
        },
        index=idx,
    )
    out = compute_metrics(df)
    assert out["n_replicates"] == 3
    assert out["peak_window_mean"] == pytest.approx((30 + 25 + 35) / 3)
    assert out["peak_strength_mean"] == pytest.approx((0.40 + 0.45 + 0.50) / 3)
    assert out["peak_window_std"] > 0
    assert out["peak_strength_std"] > 0


def test_compute_metrics_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_metrics(pd.DataFrame())


def test_iter_param_dicts_round_trips() -> None:
    matrix = np.array(
        [
            [40.0, 5.0, 0.4, 1.0, 0.5],
            [60.5, 3.2, 0.1, 1.5, 0.8],
        ]
    )
    dicts = iter_param_dicts(matrix)
    assert len(dicts) == 2
    assert dicts[0]["max_age"] == 40 and isinstance(dicts[0]["max_age"], int)
    assert dicts[1]["new_agents"] == 3 and isinstance(dicts[1]["new_agents"], int)
    assert dicts[1]["climate_phi"] == pytest.approx(0.8)


@pytest.mark.slow
def test_run_one_smoke(tmp_path: Path) -> None:
    """End-to-end: actually invoke the ABM subprocess once with tiny knobs.

    Verifies the subprocess plumbing, override propagation, and metric
    extraction all work. Skipped from the default suite via the `slow` marker.
    """
    sample = [40.0, 5.0, 0.4, 1.0, 0.5]
    result = run_one(
        sample,
        memory_baseline="collective",
        run_dir=tmp_path / "smoke",
        param_names=PARAM_NAMES,
        repeats=2,
        years=20,
        num_process=1,
        climate_process="ar1",
        cleanup=False,
    )
    assert (result.run_dir / "correlations.csv").exists()
    assert result.metrics["n_replicates"] >= 1
    assert np.isfinite(result.metrics["peak_window_mean"])
    assert np.isfinite(result.metrics["peak_strength_mean"])
    # The casted params should match the integer coercions.
    assert result.params["max_age"] == 40
    assert result.params["new_agents"] == 5
    # Persist a small artifact for manual inspection.
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "params": result.params,
                "metrics": result.metrics,
                "elapsed_seconds": result.elapsed_seconds,
            },
            indent=2,
            default=str,
        )
    )
