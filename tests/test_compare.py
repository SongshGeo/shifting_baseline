#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Regression tests for shifting_baseline.compare.compare_corr.

Focused on Bug 3 (calibration/compare feedback from Reviewer #2):
the default ``min_periods`` at compare.py:58 was written as
``min(np.log2(n), 2)`` while the accompanying comment reads
"默认最小窗口为样本数的对数". For any ``n >= 4`` the ``min`` call
collapses to ``2``, making the gatekeeper at compare.py:62 rarely
fire. The intended default is ``max(int(np.log2(n)), 2)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shifting_baseline import compare


def _mk_series(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n))


@pytest.mark.parametrize("n", [100, 200, 400])
def test_default_min_periods_scales_with_log_n(monkeypatch, n):
    """Default ``min_periods`` should track ``log2(n)``, not collapse to 2.

    We monkeypatch ``pd.Series.rolling`` to capture the kwargs that
    ``compare_corr`` actually passes through. Pre-fix the captured value
    is a constant ``2`` for any ``n >= 4``; post-fix it is
    ``max(int(log2(n)), 2)``.

    ``n`` is kept large enough that the default window
    (``n // 10``) is well above ``min_periods + n_diff_w`` so the
    gatekeeper at compare.py:62 does not short-circuit before
    ``.rolling()`` is reached. For smaller ``n`` the gatekeeper is
    exercised by :func:`test_small_window_triggers_gatekeeper_after_fix`.
    """
    captured: list[dict] = []
    original_rolling = pd.Series.rolling

    def capture(self, *args, **kwargs):
        captured.append(kwargs)
        return original_rolling(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "rolling", capture)

    a = _mk_series(n, seed=1)
    b = _mk_series(n, seed=2)
    compare.compare_corr(
        a,
        b,
        filter_func=np.mean,
        filter_side="both",
        corr_method="kendall",
        window_error="nan",
    )

    assert captured, "compare_corr did not call .rolling() as expected"
    min_periods = captured[0]["min_periods"]
    expected_floor = max(int(np.log2(n)), 2)
    assert min_periods >= expected_floor, (
        f"default min_periods={min_periods!r} but expected >= {expected_floor} "
        f"for n={n} (i.e. int(log2(n))={int(np.log2(n))})"
    )


def test_small_window_triggers_gatekeeper_after_fix():
    """When the default window is too small relative to ``min_periods + n_diff_w``
    the gatekeeper must kick in (``window_error='nan'`` → ``(nan, nan, n)``).

    With n=100 the default window is 10 and the fixed default
    ``min_periods`` is ``max(int(log2(100)), 2) == 6``. Choosing
    ``n_diff_w=5`` makes ``window (10) <= min_periods (6) + n_diff_w (5) == 11``
    and so the gatekeeper fires. Under the pre-fix code ``min_periods=2``
    and the check is ``10 <= 7`` which is false, so a numeric (non-NaN)
    correlation would be returned instead — this is exactly the
    "underconstrained rolling window" symptom Reviewer #2 flagged.
    """
    n = 100
    a = _mk_series(n, seed=1)
    b = _mk_series(n, seed=2)

    r, p, _n_out = compare.compare_corr(
        a,
        b,
        filter_func=np.mean,
        filter_side="both",
        corr_method="kendall",
        window_error="nan",
        n_diff_w=5,
    )

    assert np.isnan(
        r
    ), f"expected NaN from gatekeeper when window is too small, got r={r!r}"
    assert np.isnan(p)


def test_no_filter_func_takes_early_return():
    """When ``filter_func`` is None the function returns the unfiltered
    correlation immediately and never constructs ``default_kwargs``.

    Regression guard so the Bug 3 fix does not accidentally shift the
    non-filter code path (e.g. the ``base_corr`` call at compare.py:186
    inside ``experiment_corr_2d``).
    """
    a = _mk_series(200, seed=1)
    b = _mk_series(200, seed=2)
    r, p, n_out = compare.compare_corr(a, b, corr_method="kendall")

    assert n_out == 200
    assert not np.isnan(r)
    assert not np.isnan(p)
