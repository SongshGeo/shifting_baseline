#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.constants import LEVELS, TICK_LABELS


@pytest.fixture(name="perfect_data")
def fixture_perfect_data() -> tuple[pd.Series, pd.Series]:
    """Create perfectly matching prediction and true data.

    Returns:
        tuple: (predictions, true_values) where all values match exactly
    """
    rng = np.random.default_rng(42)
    values = rng.choice(LEVELS, size=100, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    index = pd.date_range("2000-01-01", periods=100, freq="D")

    pred = pd.Series(values, index=index, name="predictions")
    true = pd.Series(values, index=index, name="true_values")

    return pred, true


@pytest.fixture(name="realistic_data")
def fixture_realistic_data() -> tuple[pd.Series, pd.Series]:
    """Create realistic data with some mismatches and missing values.

    Returns:
        tuple: (predictions, true_values) with controlled error patterns
    """
    rng = np.random.default_rng(123)
    n_samples = 200
    index = pd.date_range("2000-01-01", periods=n_samples, freq="D")

    # Base true values
    true_values = rng.choice(LEVELS, size=n_samples, p=[0.1, 0.25, 0.3, 0.25, 0.1])

    # Add systematic prediction errors
    pred_values = true_values.copy()
    # 10% off-by-one errors
    error_mask = rng.random(n_samples) < 0.1
    error_direction = rng.choice([-1, 1], size=error_mask.sum())
    pred_values[error_mask] = np.clip(
        pred_values[error_mask] + error_direction, min(LEVELS), max(LEVELS)
    )

    # Add some missing values (5%)
    missing_mask = rng.random(n_samples) < 0.05

    # Convert to pandas Series first, then set missing values
    pred = pd.Series(pred_values, index=index, name="predictions")
    true = pd.Series(true_values, index=index, name="true_values", dtype="Int64")
    true[missing_mask] = pd.NA

    return pred, true


@pytest.fixture(name="extreme_data")
def fixture_extreme_data() -> tuple[pd.Series, pd.Series]:
    """Create data with extreme cases for edge testing.

    Returns:
        tuple: (predictions, true_values) with various edge cases
    """
    index = pd.date_range("2000-01-01", periods=50, freq="D")

    # Predictions are all one extreme
    pred = pd.Series([min(LEVELS)] * 50, index=index, name="predictions")
    # True values are all the other extreme
    true = pd.Series([max(LEVELS)] * 50, index=index, name="true_values")

    return pred, true


@pytest.fixture(name="sparse_data")
def fixture_sparse_data() -> tuple[pd.Series, pd.Series]:
    """Create very sparse data with lots of missing values.

    Returns:
        tuple: (predictions, true_values) with 80% missing values
    """
    rng = np.random.default_rng(456)
    n_samples = 100
    index = pd.date_range("2000-01-01", periods=n_samples, freq="D")

    # Start with some data
    values = rng.choice(LEVELS, size=n_samples)
    pred = pd.Series(values, index=index, name="predictions")
    true = pd.Series(values, index=index, name="true_values")

    # Make 80% missing
    missing_mask = rng.random(n_samples) < 0.8
    pred[missing_mask] = np.nan
    true[missing_mask] = np.nan

    return pred, true


class TestMismatchReportInitialization:
    """Tests for MismatchReport initialization and data validation."""

    def test_successful_initialization_with_valid_data(self, realistic_data):
        """Should successfully initialize with valid prediction and true data.

        Verifies that:
        - Object is created without errors
        - Input data is properly stored
        - Data cleaning produces non-empty results
        - All analysis components are computed
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)  # mc_runs moved to analyze_error_patterns

        # Check basic properties
        assert isinstance(report, MismatchReport)
        assert report.pred.equals(pred)
        assert report.true.equals(true)
        assert len(report.pred_clean) > 0
        assert len(report.true_clean) > 0

        # Check that basic analysis components exist
        assert hasattr(report, "cm_df")
        assert hasattr(report, "n_samples")
        assert report.n_samples == len(report.pred_clean)

        # Error analysis components should not exist until analyze_error_patterns is called
        assert report.diff_matrix is None

    def test_initialization_with_custom_parameters(self, realistic_data):
        """Should respect custom initialization parameters.

        Tests:
        - Custom monte carlo runs
        - Custom labels
        - Parameters are stored correctly
        """
        pred, true = realistic_data
        custom_labels = ["A", "B", "C", "D", "E"]

        report = MismatchReport(pred, true, labels=custom_labels)

        # mc_runs is no longer stored in the report object
        assert report.labels == custom_labels
        assert list(report.cm_df.index) == custom_labels
        assert list(report.cm_df.columns) == custom_labels

    def test_initialization_with_all_missing_data_raises_error(self):
        """Should raise ValueError when all data is missing after cleaning.

        Edge case: when concat().dropna() results in empty DataFrame
        """
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        pred = pd.Series([np.nan] * 10, index=index)
        true = pd.Series([np.nan] * 10, index=index)

        with pytest.raises(ValueError, match="清理缺失值后没有有效数据"):
            MismatchReport(pred, true)

    def test_initialization_with_non_overlapping_indices(self):
        """Should handle data with non-overlapping time indices.

        Tests robustness when pred and true have different index ranges.
        """
        pred_index = pd.date_range("2000-01-01", periods=10, freq="D")
        true_index = pd.date_range("2000-02-01", periods=10, freq="D")

        pred = pd.Series([0, 1] * 5, index=pred_index)
        true = pd.Series([0, 1] * 5, index=true_index)

        with pytest.raises(ValueError, match="清理缺失值后没有有效数据"):
            MismatchReport(pred, true)


class TestMismatchReportDataCleaning:
    """Tests for data cleaning and preprocessing functionality."""

    def test_missing_value_removal(self, sparse_data):
        """Should properly remove missing values while preserving valid data.

        Verifies:
        - Missing values are removed from both series
        - Remaining data maintains proper alignment
        - Valid data count is correct
        """
        pred, true = sparse_data
        original_valid = (~pred.isna() & ~true.isna()).sum()

        report = MismatchReport(pred, true)

        assert len(report.pred_clean) == original_valid
        assert len(report.true_clean) == original_valid
        assert len(report.pred_clean) == len(report.true_clean)
        assert not report.pred_clean.isna().any()
        assert not report.true_clean.isna().any()

    def test_index_alignment_preservation(self, realistic_data):
        """Should maintain proper index alignment after cleaning.

        Tests that cleaned data maintains consistent indexing between
        prediction and true value series.
        """
        pred, true = realistic_data
        # Introduce some NAs at different positions
        pred.iloc[5:10] = np.nan
        true.iloc[15:20] = np.nan

        report = MismatchReport(pred, true)

        # Check that remaining data is properly aligned
        assert report.pred_clean.index.equals(report.true_clean.index)
        assert len(report.pred_clean) > 0


class TestMismatchReportStatistics:
    """Tests for statistical computation functionality."""

    def test_perfect_match_statistics(self, perfect_data):
        """Should compute correct statistics for perfectly matching data.

        Expected results for perfect match:
        - Accuracy = 1.0
        - Kappa = 1.0
        - Kendall's Tau = 1.0 (or very close)
        """
        pred, true = perfect_data
        report = MismatchReport(pred, true)

        stats = report.get_statistics_summary()
        assert stats["accuracy"] == 1.0
        assert stats["kappa"] == pytest.approx(1.0, abs=1e-10)
        assert stats["kendall_tau"] == pytest.approx(1.0, abs=1e-2)
        assert stats["tau_p_value"] < 0.05  # Should be highly significant

    def test_extreme_mismatch_statistics(self, extreme_data):
        """Should compute reasonable statistics for completely mismatched data.

        Expected for opposite extremes:
        - Accuracy = 0.0
        - Kappa <= 0 (no better than random)
        - Tau should be computable (may be NaN for constant predictions)
        """
        pred, true = extreme_data
        report = MismatchReport(pred, true)

        stats = report.get_statistics_summary()
        assert stats["accuracy"] == 0.0
        assert stats["kappa"] <= 0.0
        # Tau may be NaN when predictions are constant, so we just check it's computed
        assert "kendall_tau" in stats

    def test_statistics_summary_formats(self, realistic_data):
        """Should provide statistics in both dictionary and string formats.

        Tests:
        - Dictionary format contains all expected keys
        - String format is properly formatted
        - Significance marking works correctly
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Test dictionary format
        stats_dict = report.get_statistics_summary(as_str=False)
        expected_keys = {
            "kappa",
            "kendall_tau",
            "tau_p_value",
            "accuracy",
            "n_samples",
            "n_raw_samples",
            "n_mismatches",
        }
        assert set(stats_dict.keys()) == expected_keys
        assert all(
            isinstance(v, (int, float, np.integer, np.floating))
            for v in stats_dict.values()
        )

        # Test string format
        stats_str = report.get_statistics_summary(as_str=True)
        assert isinstance(stats_str, str)
        assert "Kappa:" in stats_str
        assert "Tau:" in stats_str


class TestMismatchReportConfusionMatrix:
    """Tests for confusion matrix computation and properties."""

    def test_confusion_matrix_shape_and_labels(self, realistic_data):
        """Should create confusion matrix with correct shape and labels.

        Verifies:
        - Matrix is square with correct dimensions
        - Row/column labels match input labels
        - All values are non-negative integers
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        assert report.cm_df.shape == (5, 5)
        assert list(report.cm_df.index) == TICK_LABELS
        assert list(report.cm_df.columns) == TICK_LABELS
        assert (report.cm_df >= 0).all().all()
        assert report.cm_df.dtypes.apply(lambda x: np.issubdtype(x, np.integer)).all()

    def test_confusion_matrix_total_count(self, realistic_data):
        """Should have total count equal to cleaned data size.

        Sum of all confusion matrix entries should equal the number
        of valid (non-missing) data points.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        assert report.cm_df.sum().sum() == len(report.pred_clean)


class TestMismatchReportErrorAnalysis:
    """Tests for error pattern analysis functionality."""

    def test_analyze_error_patterns_with_value_series(self, realistic_data):
        """Should analyze error patterns when value_series is provided."""
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Create natural value series
        natural_data = pd.Series(
            np.random.normal(0, 1, len(pred)), index=pred.index, name="natural_values"
        )

        # Analyze error patterns
        error_df = report.analyze_error_patterns(natural_data, mc_runs=10)

        assert isinstance(error_df, pd.DataFrame)
        assert len(error_df) > 0
        assert_frame_equal(error_df, report.diff_matrix)

        # Check that analysis matrices are created
        assert hasattr(report, "diff_matrix")
        assert hasattr(report, "p_value_matrix")
        assert hasattr(report, "false_count_matrix")

    def test_analyze_error_patterns_without_value_series(self, realistic_data):
        """Should handle gracefully when no value_series is provided."""
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Call without providing value_series
        error_df = report.analyze_error_patterns(mc_runs=10)

        # Should return empty DataFrame and log warning
        assert isinstance(error_df, pd.DataFrame)
        assert len(error_df) == 0

    def test_analyze_error_patterns_with_initialization_value_series(
        self, realistic_data
    ):
        """Should use value_series provided during initialization."""
        pred, true = realistic_data
        natural_data = pd.Series(
            np.random.normal(0, 1, len(pred)), index=pred.index, name="natural_values"
        )

        # Initialize with value_series
        report = MismatchReport(pred, true, value_series=natural_data)

        # Call analyze_error_patterns with mc_runs
        error_df = report.analyze_error_patterns(mc_runs=10)

        assert isinstance(error_df, pd.DataFrame)
        assert len(error_df) > 0


class TestMismatchReportMonteCarlo:
    """Tests for Monte Carlo significance testing."""

    def test_monte_carlo_with_sufficient_runs(self, realistic_data):
        """Should complete Monte Carlo simulation with reasonable number of runs.

        Tests:
        - Simulation completes without errors
        - Results matrices have expected shape
        - P-values are within valid range [0, 1]
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need to call analyze_error_patterns first
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=20)

        # After analyze_error_patterns, these matrices should be available
        assert hasattr(report, "diff_matrix")
        assert hasattr(report, "p_value_matrix")

        # Check shapes
        assert report.p_value_matrix.shape == (5, 5)

        # Check p-value validity
        p_values = report.p_value_matrix.values.flatten()
        valid_p_values = p_values[~np.isnan(p_values)]
        if len(valid_p_values) > 0:
            assert (valid_p_values >= 0).all()
            assert (valid_p_values <= 1).all()

    def test_monte_carlo_with_minimal_runs(self, realistic_data):
        """Should handle Monte Carlo simulation with very few runs.

        Edge case: tests behavior with mc_runs=1 to ensure
        no division by zero or similar numerical issues.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need to call analyze_error_patterns first
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=1)

        # Should complete without errors
        assert hasattr(report, "p_value_matrix")
        # mc_runs is no longer stored in the report object

    def test_monte_carlo_with_no_misclassifications(self, perfect_data):
        """Should handle case where no misclassifications occur.

        When all predictions are perfect, there should be no
        misclassification patterns to analyze.
        """
        pred, true = perfect_data
        report = MismatchReport(pred, true)

        # For perfect data, we can still analyze error patterns
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=10)

        # Should create matrices without crashing (may be empty/NaN for perfect matches)
        assert hasattr(report, "diff_matrix")
        assert hasattr(report, "p_value_matrix")


class TestMismatchReportVisualization:
    """Tests for plotting and visualization functionality."""

    def test_plot_confusion_matrix_creation(self, realistic_data):
        """Should create confusion matrix plot without errors.

        Tests:
        - Plot is created successfully
        - Returns matplotlib Axes object
        - Can specify custom title
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Test with default title
        ax1 = report.plot_confusion_matrix()
        assert isinstance(ax1, plt.Axes)

        # Test with custom title
        ax2 = report.plot_confusion_matrix(title="Custom Title")
        assert isinstance(ax2, plt.Axes)

        plt.close("all")  # Clean up

    def test_plot_mismatch_analysis_creation(self, realistic_data):
        """Should create mismatch analysis plot without errors.

        Tests plot creation for the flow/difference matrix visualization.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need error analysis for mismatch analysis plot
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=10)

        ax = report.plot_mismatch_analysis()
        assert isinstance(ax, plt.Axes)

        plt.close("all")

    def test_plot_heatmap_creation(self, realistic_data):
        """Should create heatmap plot without errors.

        Tests heatmap visualization of difference matrix with p-values.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need error analysis for heatmap
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=10)

        ax = report.plot_heatmap()
        assert isinstance(ax, plt.Axes)

        plt.close("all")

    def test_generate_report_figure_creation(self, realistic_data):
        """Should create complete report figure with multiple subplots.

        Tests:
        - Figure is created with correct subplot structure
        - Can save to file when path is provided
        - Returns matplotlib Figure object
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need error analysis for complete report figure
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=10)

        # Test without saving
        fig = report.generate_report_figure()
        assert isinstance(fig, plt.Figure)
        assert (
            len(fig.axes) >= 2
        )  # Should have at least 2 subplots (may have colorbars)

        # Test with saving
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_report.png"
            fig2 = report.generate_report_figure(save_path=str(save_path))
            assert isinstance(fig2, plt.Figure)
            assert save_path.exists()

        plt.close("all")


class TestMismatchReportDataExport:
    """Tests for data export and serialization functionality."""

    def test_to_dict_export(self, realistic_data):
        """Should export complete analysis results to dictionary format.

        Tests:
        - Dictionary contains all expected top-level keys
        - Nested structures are properly serialized
        - Statistics are included
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Need error analysis for complete export
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=10)

        result_dict = report.to_dict()

        expected_keys = {
            "statistics",
            "confusion_matrix",
            "diff_matrix",
            "p_value_matrix",
            "false_count_matrix",
        }
        assert set(result_dict.keys()) == expected_keys

        # Check that statistics is a dictionary
        assert isinstance(result_dict["statistics"], dict)

        # Check that matrices are serialized
        assert isinstance(result_dict["confusion_matrix"], dict)
        assert isinstance(result_dict["diff_matrix"], dict)

    def test_repr_string_representation(self, realistic_data):
        """Should provide informative string representation.

        Tests __repr__ method returns properly formatted string
        with key information about the report.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        repr_str = repr(report)
        assert isinstance(repr_str, str)
        assert "MismatchReport" in repr_str
        assert "n_samples=" in repr_str


@pytest.mark.parametrize("mc_runs", [1, 5, 10, 50])
class TestMismatchReportParameterization:
    """Parameterized tests for different Monte Carlo run counts."""

    def test_varying_mc_runs_performance(self, realistic_data, mc_runs):
        """Should handle different Monte Carlo run counts appropriately.

        Tests that the analysis works correctly across different
        computational complexity levels.
        """
        pred, true = realistic_data
        report = MismatchReport(pred, true)

        # Test with different mc_runs values by calling analyze_error_patterns
        natural_data = pd.Series(np.random.normal(0, 1, len(pred)), index=pred.index)
        report.analyze_error_patterns(natural_data, mc_runs=mc_runs)

        # Test statistics
        stats = report.get_statistics_summary()
        assert isinstance(stats["accuracy"], (float, np.floating))
        assert isinstance(stats["kappa"], (float, np.floating))
        assert isinstance(stats["kendall_tau"], (float, np.floating))

        # All runs should produce finite statistics
        assert np.isfinite(stats["kappa"])
        assert np.isfinite(stats["kendall_tau"])
        assert np.isfinite(stats["accuracy"])


@pytest.mark.parametrize(
    "data_fixture", ["perfect_data", "realistic_data", "extreme_data"]
)
class TestMismatchReportDataVariations:
    """Tests across different data patterns and characteristics."""

    def test_different_data_patterns(self, data_fixture, request):
        """Should handle various data patterns correctly.

        Tests robustness across:
        - Perfect matches
        - Realistic error patterns
        - Extreme mismatches
        """
        pred, true = request.getfixturevalue(data_fixture)
        report = MismatchReport(pred, true)

        # Basic integrity checks that should pass for all data types
        assert len(report.pred_clean) > 0
        assert len(report.true_clean) > 0

        stats = report.get_statistics_summary()
        assert 0 <= stats["accuracy"] <= 1
        assert isinstance(stats["kappa"], (float, np.floating))
        assert isinstance(stats["kendall_tau"], (float, np.floating))

        # Should be able to create confusion matrix plot
        ax = report.plot_confusion_matrix()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        # Test error analysis if we have enough data
        if len(report.pred_clean) > 5:  # Only test with sufficient data
            natural_data = pd.Series(
                np.random.normal(0, 1, len(pred)), index=pred.index
            )
            error_df = report.analyze_error_patterns(natural_data, mc_runs=5)
            assert isinstance(error_df, pd.DataFrame)

            # Should be able to create full report after error analysis
            fig = report.generate_report_figure()
            assert isinstance(fig, plt.Figure)
            plt.close("all")


class TestMismatchReportEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_data_point(self):
        """Should handle case with only one valid data point.

        Edge case: minimal data that should still produce some results.
        """
        pred = pd.Series([0], index=[0])
        true = pd.Series([1], index=[0])

        report = MismatchReport(pred, true)

        assert len(report.pred_clean) == 1
        assert len(report.true_clean) == 1

        stats = report.get_statistics_summary()
        assert stats["accuracy"] == 0.0  # Single mismatch = 0% accuracy

    def test_all_same_class_predictions(self):
        """Should handle case where all predictions are the same class.

        Edge case: no variation in predictions should still produce
        valid statistical measures.
        """
        pred = pd.Series([0] * 20, index=range(20))
        true = pd.Series(list(range(5)) * 4, index=range(20))  # Varied true values

        report = MismatchReport(pred, true)

        assert report.pred_clean.nunique() == 1
        assert report.true_clean.nunique() > 1

        stats = report.get_statistics_summary()
        assert 0 <= stats["accuracy"] <= 1

    def test_all_same_class_true_values(self):
        """Should handle case where all true values are the same class.

        Edge case: no variation in ground truth.
        """
        pred = pd.Series(list(range(5)) * 4, index=range(20))  # Varied predictions
        true = pd.Series([2] * 20, index=range(20))

        report = MismatchReport(pred, true)

        assert report.pred_clean.nunique() > 1
        assert report.true_clean.nunique() == 1

        stats = report.get_statistics_summary()
        assert 0 <= stats["accuracy"] <= 1

    def test_very_sparse_misclassifications(self):
        """Should handle case with very few misclassifications.

        Tests behavior when error rate is extremely low.
        """
        rng = np.random.default_rng(789)
        n_samples = 100

        # Start with perfect match
        values = rng.choice(LEVELS, size=n_samples)
        pred = pd.Series(values.copy())
        true = pd.Series(values.copy())

        # Introduce just 1-2 errors
        error_indices = rng.choice(n_samples, size=2, replace=False)
        for idx in error_indices:
            # Flip to a different class
            current = pred.iloc[idx]
            alternatives = [x for x in LEVELS if x != current]
            pred.iloc[idx] = rng.choice(alternatives)

        report = MismatchReport(pred, true)

        stats = report.get_statistics_summary()
        assert 0.95 <= stats["accuracy"] <= 1.0  # Should be very high accuracy
        assert stats["kappa"] > 0.8  # Should still have good agreement
