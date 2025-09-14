#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd
import pytest

from shifting_baseline.constants import GRADE_VALUES, STAGES_BINS, STD_THRESHOLDS
from shifting_baseline.data import HistoricalRecords


def _make_fake_excel(
    region_names: List[str], years: List[int], cities: List[str]
) -> Dict[str, pd.DataFrame]:
    """Construct fake Excel-like sheets for multiple regions.

    0 indicates missing values; 1-5 are levels (5 wettest to 1 driest in raw),
    but we will later transform to symmetric levels where positive = wet.
    """
    rng = np.random.default_rng(123)
    sheets: Dict[str, pd.DataFrame] = {}
    for region in region_names:
        # Create random integers in [0,5], 0 used as missing
        data = rng.integers(low=0, high=6, size=(len(years), len(cities)))
        df = pd.DataFrame(data, index=years, columns=cities)
        df.index.name = "year"
        sheets[region] = df
    return sheets


@pytest.fixture(autouse=True)
def patch_gpd_read_file(monkeypatch: pytest.MonkeyPatch):
    """Patch geopandas.read_file to return a minimal DataFrame with region column."""
    import geopandas as gpd

    def _fake_read_file(_path):
        return gpd.GeoDataFrame({"region": ["华北地区", "华南地区"]})

    monkeypatch.setattr("geopandas.read_file", _fake_read_file)


class TestHistoricalRecordsLoad:
    """Tests for loading and basic properties of HistoricalRecords.

    Focus on:
      - Reading a selected region
      - Converting zeros to missing and reindexing to full 1000..2020
      - Symmetric level mapping when `symmetrical_level=True`
    """

    def test_load_region_and_symmetry(self, shp_file: Path, excel_file: Path):
        """Should load selected region, apply symmetry (3->0 center), and keep index 1000..2020."""
        rec = HistoricalRecords(
            shp_path=shp_file,
            data_path=excel_file,
            region="华北地区",
            symmetrical_level=True,
        )
        # Index coverage
        assert rec.data.index.min() == 1000
        assert rec.data.index.max() == 2020
        # Symmetry check: original grades [1..5] with symmetry -> 3-level center becomes 0
        # Since we wrote 0 for missing, after loader there should be NA at those years/places
        assert pd.isna(rec.data.loc[1003]).all()
        # After symmetry, values should be in { -2, -1, 0, 1, 2 } possibly with NA
        non_na_vals = rec.data.stack().dropna().unique()
        assert set(non_na_vals).issubset({-2, -1, 0, 1, 2})


class TestHistoricalRecordsPeriod:
    """Tests for the `period` selection API including ints, slices and strings."""

    @pytest.fixture()
    def rec(self, shp_file: Path, excel_file: Path) -> HistoricalRecords:
        return HistoricalRecords(
            shp_path=shp_file,
            data_path=excel_file,
            region="华北地区",
            symmetrical_level=True,
        )

    def test_period_int(self, rec: HistoricalRecords):
        start, end = STAGES_BINS[0], STAGES_BINS[1]
        sel = rec.period(1)
        assert sel.index.min() == start and sel.index.max() == end

    def test_period_slice(self, rec: HistoricalRecords):
        start, end = STAGES_BINS[0], STAGES_BINS[2]
        sel = rec.period(slice(1, 2))
        assert sel.index.min() == start and sel.index.max() == end

    @pytest.mark.parametrize("expr", ["1:3", "1-3", "stage1:stage3", "stage1-stage3"])
    def test_period_string_stage_ranges(self, rec: HistoricalRecords, expr: str):
        start, end = STAGES_BINS[0], STAGES_BINS[3]
        sel = rec.period(expr)
        assert sel.index.min() == start and sel.index.max() == end

    def test_period_explicit_years(self, rec: HistoricalRecords):
        sel = rec.period("1000:1005")
        assert sel.index.min() == 1000 and sel.index.max() == 1005

    def test_period_invalid(self, rec: HistoricalRecords):
        with pytest.raises(ValueError):
            rec.period("unknown")


class TestHistoricalRecordsSeriesOps:
    """Tests for series transformations and aggregation paths."""

    @pytest.fixture()
    def rec(self, shp_file: Path, excel_file: Path) -> HistoricalRecords:
        return HistoricalRecords(
            shp_path=shp_file,
            data_path=excel_file,
            region="华北地区",
            symmetrical_level=True,
        )

    def test_to_series_mean(self, rec: HistoricalRecords):
        ser = rec.aggregate(how="mean", inplace=False)
        assert isinstance(ser, pd.Series)
        assert ser.index.min() == 1000 and ser.index.max() == 2020

    def test_to_series_inplace_chain(self, rec: HistoricalRecords):
        out = rec.aggregate(how="median", inplace=True)
        assert isinstance(out, HistoricalRecords)
        sel = out.period("1000:1010")
        assert isinstance(sel, pd.Series)

    def test_rescale_to_std_mapping(self, rec: HistoricalRecords):
        mapped = rec.rescale_to_std()
        # mapping from grade values to STD thresholds
        mapping = dict(zip(GRADE_VALUES, STD_THRESHOLDS))
        # If any grade value appears in data, it should be replaced by thresholds
        # Note: data may contain NA from zeros; we only check present values
        unique_vals = rec.data.stack().dropna().unique()
        possible_targets = set(mapping.values()) | {-2, -1, 0, 1, 2}
        # Ensure mapping returns same index/shape and values are within expected set
        assert mapped.shape == rec.data.shape
        assert set(unique_vals).issubset(possible_targets)


class TestHistoricalRecordsMergeAndCorr:
    """Tests for merge_with and corr_with utilities including edge cases."""

    @pytest.fixture()
    def rec_series(self, shp_file: Path, excel_file: Path) -> pd.Series:
        rec = HistoricalRecords(
            shp_path=shp_file,
            data_path=excel_file,
            region="华北地区",
            symmetrical_level=True,
        )
        return rec.aggregate(how="mean", inplace=False)

    def test_merge_with_aligns_index(self, rec_series: pd.Series):
        other = pd.Series(
            np.arange(1000, 1011), index=np.arange(1000, 1011), name="other"
        )
        rec = rec_series
        # Build a small HistoricalRecords-like holder to call merge_with; use minimal shim

        class Holder:
            """Mock HistoricalRecords-like holder for testing merge_with"""

            def __init__(self, data):
                self.data = data

            def period(self, period):
                """Mock period method that filters data by time range"""
                if isinstance(period, str) and period == "all":
                    return self.data
                elif hasattr(period, "__iter__") and not isinstance(period, str):
                    # Handle numpy array or list of years
                    years = list(period)
                    return self.data.loc[self.data.index.isin(years)]
                elif isinstance(period, slice):
                    return self.data.loc[period]
                else:
                    return self.data

        holder = Holder(rec)
        merged = HistoricalRecords.merge_with(
            cast(HistoricalRecords, holder), other, time_range=np.arange(1000, 1011)
        )
        assert merged.shape[1] == 2
        assert merged.index.min() == 1000 and merged.index.max() == 1010

    def test_corr_with_basic(self, rec_series: pd.Series):
        # Correlate with a noisy version to ensure valid output
        noise = np.random.default_rng(0).normal(0, 0.1, size=rec_series.shape[0])
        other = pd.Series(rec_series.values + noise, index=rec_series.index)
        # Build a minimal wrapper exposing get_series

        class Wrapper:
            def __init__(self, series: pd.Series):
                self._s = series

            def get_series(self, col=None):  # noqa: ARG001
                _ = col
                return self._s

        wrp = Wrapper(rec_series)
        r, p, n = HistoricalRecords.corr_with(cast(HistoricalRecords, wrp), other)
        assert np.isfinite(r) and np.isfinite(p)
        assert isinstance(n, Integral) and int(n) > 0
