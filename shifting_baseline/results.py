#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Reproducible builders for the manuscript's key rendered numbers (``results.json``).

This module is the single source of truth for the numbers quoted in the paper's
Results section. The figure notebooks (``reports/{natural,mismatch,history}.ipynb``)
and the regression guardrail (``tests/test_results_regression.py``) both go through
here, so a refactor of the underlying package cannot silently move a headline
number without the guardrail turning red.

Each ``compute_resultsN`` function mirrors the corresponding notebook cell exactly,
reusing the same package entry points (``load_data``, ``MismatchReport``,
``experiment_corr_2d``, ``sweep_max_corr_year`` …) with the same parameters and the
same rounding rules. Determinism is guaranteed by the global ``cfg.random_seed``
(default 42), which seeds the ``to_std=sampling`` historical aggregation and the
Monte-Carlo randomisation null inside the mismatch analysis. (With the default
``recalculate_data=false`` the N-WDI is read from the committed cache rather than
re-run through MCMC, so the reported numbers are fully deterministic.)

The regression guardrail (``tests/test_results_regression.py``) locks these
functions against a golden snapshot — i.e. against *themselves*, not against the
notebooks. The one-to-one fidelity with the notebook cells is maintained by hand;
keep it in sync when editing either side, or a manuscript number can drift silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

from shifting_baseline.calibration import MismatchReport
from shifting_baseline.compare import (
    experiment_corr_2d,
    sweep_max_corr_year,
    sweep_slices,
)
from shifting_baseline.constants import END, STAGE1, VALIDATION_PERIOD
from shifting_baseline.data import load_data, load_validation_data, to_validation_period
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.utils.calc import calc_corr

__all__ = [
    "build_results",
    "compute_results1",
    "compute_results2",
    "compute_results3",
    "site_corr_table",
    "site_significance_table",
]

# tau 类与 p 值保留 3 位小数（区分 0.176/0.18/0.19；p 按期刊要求到三位），其余 2 位
_TAU_KEYS = {
    "kendall_tau",
    "kendall_tau_validation",
    "tau_ci_low",
    "tau_ci_high",
    "tau_ci_low_validation",
    "tau_ci_high_validation",
    "tau_p_value",
    "tau_p_value_validation",
    "kappa_p_value",
    "kappa_p_value_validation",
}


def spatial_corr(series: xr.DataArray, validation_z: xr.DataArray) -> xr.DataArray:
    """Grid-wise Pearson r / p / n between the N-WDI series and the validation grid.

    Shared with natural.ipynb (Fig 2c, SI Fig 4): the validation grid is restricted
    to the validation period (1901–2000 CE) before correlating over the overlapping
    years, returning stacked (r, p, n) DataArrays over the spatial dims.
    """
    validation_z = to_validation_period(validation_z)
    common_years = np.intersect1d(series.year.values, validation_z.year.values)
    return xr.apply_ufunc(
        calc_corr,
        validation_z.sel(year=common_years),
        series.sel(year=common_years),
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        output_dtypes=[float, float, int],
    )


def _site_corr(corr: tuple[xr.DataArray, ...], region_gdf) -> pd.DataFrame:
    """Nearest-grid Pearson r / p / n at each historical-archive site (mirrors calc_sites_corr)."""
    rows = [
        {
            "site": name,
            "lon": lon,
            "lat": lat,
            **{
                key: da.sel(x=lon, y=lat, method="nearest").item()
                for key, da in zip(("r", "p", "n"), corr)
            },
        }
        for name, lon, lat in zip(region_gdf.name_en, region_gdf.lon, region_gdf.lat)
    ]
    return pd.DataFrame(rows).set_index("site")


def _round_stats(results: dict) -> dict:
    """Apply the notebook rounding convention: tau keys → 3 dp, others → 2 dp, counts → int."""
    rounded = {}
    for key, value in results.items():
        if isinstance(value, (np.floating, float)):
            rounded[key] = round(float(value), 3 if key in _TAU_KEYS else 2)
        elif isinstance(value, (np.integer, int)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def compute_results1(cfg: DictConfig, combined: pd.DataFrame | None = None) -> dict:
    """Reproduce ``results1`` (natural.ipynb): reconstruction/validation summary.

    Fields (all mirror the manuscript's §2.1 / Fig 2 definitions). Everything uses
    the single ``using_val_data`` validation dataset (default ``china``) for
    consistency — both the reported correlation and the spatial site significance
    come from the same product, so Fig 2a and Fig 2c refer to the same data.

    - ``n_datasets``: number of integrated tree-ring reconstructions.
    - ``corr``: point-to-point Pearson r between the N-WDI and the ``using_val_data``
      instrumental validation series over the validation period (1901–2000 CE).
    - ``n_pass_years``: the low-pass filter window quoted in the text — a config
      value (``cfg.low_pass.window_size``), not a fitted statistic.
    - ``n_sites_sig005`` / ``n_sites_sig01``: number of the historical-archive sites
      whose nearest validation grid cell correlates with the N-WDI at p<0.05 / p<0.1
      (mirrors ``calc_sites_corr``; site r are positive in practice).
    - ``sig_sites_percentage``: ``n_sites_sig005 / n_sites`` in percent (Fig 2c caption).
    - ``sig_sites_ratio``: ``n_sites_sig01 / n_sites`` (legacy key, p<0.1).
    """
    combined, uncertainties, history = load_data(cfg, combined)
    tree_ring = combined["mean"]

    ds = cfg.ds.validation[cfg.using_val_data]
    summer_precip_z, regional_z = load_validation_data(
        data_path=ds.z_nc,
        resolution=cfg.resolution,
        csv_save_to=ds.csv,
        nc_save_to=ds.z_nc,
    )

    # Point-to-point validation correlation (Fig 2a), using_val_data (default china)
    corr = float(
        to_validation_period(regional_z).corr(tree_ring.loc[VALIDATION_PERIOD])
    )

    # Spatial significance at the historical-archive sites (Fig 2c), same val grid
    history.setup()  # restrict shp to the study region (华北, 32 sites)
    region_gdf = history.shp
    spatial = spatial_corr(tree_ring.to_xarray(), summer_precip_z)
    p_sites = _site_corr(spatial, region_gdf)["p"].to_numpy()
    n_sites = int(len(region_gdf))
    n_sites_sig005 = int((p_sites < 0.05).sum())
    n_sites_sig01 = int((p_sites < 0.1).sum())

    return {
        "n_datasets": int(uncertainties.shape[1]),
        "corr": round(corr, 2),
        "n_pass_years": int(cfg.low_pass.window_size),
        "sig_sites_ratio": round(n_sites_sig01 / n_sites, 2),
        "n_sites_sig005": n_sites_sig005,
        "n_sites_sig01": n_sites_sig01,
        "sig_sites_percentage": round(n_sites_sig005 / n_sites * 100, 2),
    }


def site_corr_table(cfg: DictConfig) -> pd.DataFrame:
    """Per-site Pearson r / two-sided p / n behind Fig 2c (site-significance counts).

    Same data and nearest-grid lookup as ``compute_results1``: the N-WDI against the
    ``using_val_data`` grid over the validation period, at each of the study-region
    historical-archive sites.
    """
    combined, _, history = load_data(cfg)
    ds = cfg.ds.validation[cfg.using_val_data]
    summer_precip_z, _ = load_validation_data(
        data_path=ds.z_nc,
        resolution=cfg.resolution,
        csv_save_to=ds.csv,
        nc_save_to=ds.z_nc,
    )
    history.setup()
    spatial = spatial_corr(combined["mean"].to_xarray(), summer_precip_z)
    table = _site_corr(spatial, history.shp)
    table["n"] = table["n"].astype(int)
    return table


def site_significance_table(cfg: DictConfig) -> pd.DataFrame:
    """Per-validation-dataset site-significance summary (for SI Table S6).

    For each validation product (china/gpcc/cru), reports how many of the
    historical-archive sites have a nearest grid cell significantly correlated with
    the N-WDI, at both p<0.05 and p<0.1, plus the corresponding percentage/ratio.
    This makes the two thresholds explicit so the main text (p<0.05 count) and any
    SI table (p<0.1 ratio) can be reconciled to one convention.
    """
    combined, _, history = load_data(cfg)
    series = combined["mean"].to_xarray()
    history.setup()
    region_gdf = history.shp
    n_sites = int(len(region_gdf))

    rows = []
    for name, vds in cfg.ds.validation.items():
        summer_precip_z, _ = load_validation_data(
            data_path=vds.z_nc,
            resolution=cfg.resolution,
            csv_save_to=vds.csv,
            nc_save_to=vds.z_nc,
        )
        p_sites = _site_corr(spatial_corr(series, summer_precip_z), region_gdf)[
            "p"
        ].to_numpy()
        n05 = int((p_sites < 0.05).sum())
        n10 = int((p_sites < 0.1).sum())
        rows.append(
            {
                "dataset": name,
                "n_sites": n_sites,
                "n_sig_p<0.05": n05,
                "pct_p<0.05": round(n05 / n_sites * 100, 1),
                "n_sig_p<0.1": n10,
                "ratio_p<0.1": round(n10 / n_sites, 3),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def compute_results2(cfg: DictConfig, combined: pd.DataFrame | None = None) -> dict:
    """Reproduce ``results2`` (mismatch.ipynb): H-WDI vs N-WDI confusion statistics.

    Mirrors mismatch.ipynb cells 6/9/14 exactly: main mismatch report statistics,
    ``mean_diff`` under the shift=2 convention, plus validation-period stats
    restricted to the validation period (1901–2000 CE, ``VALIDATION_PERIOD``), with the
    instrumental z-scores re-standardised within that period.
    """
    combined, _, history = load_data(cfg, combined)
    tree_ring = combined["mean"]
    # Seed the Monte-Carlo null so the analysis is reproducible. (results2's fields
    # come from the observed confusion matrix / diff matrix and don't read the MC
    # p-values, so the seed doesn't move the numbers — it just removes nondeterminism.)
    seed = cfg.get("random_seed", 42)

    # 校验期报告：Constructed N-WDI vs Instrument N-WDI
    ds = cfg.ds.validation[cfg.using_val_data]
    _, validation_data = load_validation_data(
        data_path=ds.z_nc,
        resolution=cfg.resolution,
        csv_save_to=ds.csv,
        nc_save_to=ds.z_nc,
    )
    validation_mismatch_report = MismatchReport(
        pred=classify(to_validation_period(validation_data)),
        true=classify(tree_ring.loc[VALIDATION_PERIOD]),
        value_series=tree_ring.loc[VALIDATION_PERIOD],
    )
    validation_mismatch_report.analyze_error_patterns(random_seed=seed)

    # 主报告：H-WDI vs N-WDI（stage 2-3，连续值后分类）
    history.setup()
    pred, true = history.aggregate(
        how=cfg.agg_method,
        to_int=False,
        inplace=True,
    ).merge_with(tree_ring, time_range="2-3", split=True)
    mismatch_report = MismatchReport(
        pred=classify(pred, handle_na="skip"),
        true=classify(true, handle_na="skip"),
        value_series=true,
    )
    mismatch_report.analyze_error_patterns(random_seed=seed)

    results = mismatch_report.get_statistics_summary()
    results.update(mismatch_report.get_agreement_inference(random_seed=seed))
    # mean_diff 采用 shift=2 口径（与正文表述一致），而非整体平均
    mismatch_report.analyze_error_patterns(shift=2, random_seed=seed)
    assert mismatch_report.diff_matrix is not None  # set by analyze_error_patterns
    results["mean_diff"] = float(mismatch_report.diff_matrix.abs().mean().mean())

    validation_stats = validation_mismatch_report.get_statistics_summary()
    validation_stats.update(
        validation_mismatch_report.get_agreement_inference(random_seed=seed)
    )
    for key in (
        "kendall_tau",
        "tau_p_value",
        "tau_ci_low",
        "tau_ci_high",
        "kappa",
        "kappa_p_value",
        "kappa_ci_low",
        "kappa_ci_high",
        "n_samples",
    ):
        results[f"{key}_validation"] = validation_stats[key]

    return _round_stats(results)


def compute_results3(cfg: DictConfig, combined: pd.DataFrame | None = None) -> dict:
    """Reproduce ``results3`` (history.ipynb): sliding-window re-standardisation gains.

    Mirrors history.ipynb cells 6/7/8/15/16. The whole-period optimum uses the Fig4b
    variant (``std_offset=0, max_window=51``); the per-segment optimum comes from the
    ``sweep_max_corr_year`` scan. Percentages follow the "value × 100" convention.
    """
    combined, _, history = load_data(cfg, combined)
    tree_ring = combined["mean"]
    slice_now = slice(STAGE1, END)

    # cell 6：聚合并做 2-3 期切片（结果留在 history.data 供后续全期合并复用）
    history.setup()
    history.aggregate(
        how=cfg.agg_method,
        to_int=cfg.to_int,
        inplace=True,
    ).merge_with(tree_ring, time_range="2-3", split=True)

    # cell 7：分段扫描最优窗口（起点用 STAGE1，避免与档案分析窗起点失同步）
    slices, _mid_year, _slice_labels = sweep_slices(
        start_year=STAGE1,
        window_size=200,
        step_size=20,
        end_year=END,
    )
    data1, data2 = history.merge_with(tree_ring, split=True)
    max_corr_year, max_corr, r_benchmark_list, _p_value_list = sweep_max_corr_year(
        data1=data1,
        data2=data2,
        slices=slices,
        corr_method=cfg.corr_method,
        windows=np.arange(2, 100),
        min_periods=np.repeat(cfg.min_period, 98),
        filter_func=calc_std_deviation,
        ratio=cfg.ratio,
    )
    # cell 8：分段平均基准与最优
    r_benchmark_avg = np.nanmean(np.array(r_benchmark_list, dtype=float))
    max_corr_avg = np.nanmean(np.stack(max_corr))

    # cell 15：整段 Fig4b（std_offset=0, max_window=51）
    df, r_benchmark, _ax = experiment_corr_2d(
        data1=data1,
        data2=data2,
        time_slice=slice_now,
        corr_method=cfg.corr_method,
        filter_func=calc_std_deviation,
        filter_side=cfg.filter_side,
        n_diff_w=5,
        std_offset=0,
        max_window=51,
    )

    # cell 16：写入 results3
    return {
        "tau": round(float(r_benchmark), 3),
        "w_optimal": int(df.max().idxmax()),
        "max_increase": round(
            (max_corr_avg - r_benchmark_avg) / r_benchmark_avg * 100, 2
        ),
        "optimal_segment_w": int(round(float(np.stack(max_corr_year).mean()))),
        "whole_period_increase_pct": round(
            (float(np.nanmax(df.values)) - r_benchmark) / r_benchmark * 100, 2
        ),
    }


def build_results(cfg: DictConfig, combined: pd.DataFrame | None = None) -> dict:
    """Assemble the full ``results.json`` payload (results1 + results2 + results3).

    ``combined`` optionally replaces the cached N-WDI (see ``load_data``), e.g. to
    recompute the headline numbers for an alternative prior.
    """
    return {
        "results1": compute_results1(cfg, combined),
        "results2": compute_results2(cfg, combined),
        "results3": compute_results3(cfg, combined),
    }
