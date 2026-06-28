#!/usr/bin/env python
"""Robustness check for Reviewer #2: does the discrete->continuous mapping of
ordinal WDI levels (stochastic truncated-normal *sampling* vs. deterministic
midpoint *mapping*) change the window-dependent Kendall's tau?

We reuse the exact empirical pipeline behind Figure 4: per-site historical
grades are mapped to continuous values, averaged across sites to a regional
H-WDI, re-classified to five levels, and correlated (Kendall's tau) against the
re-standardised natural N-WDI over a grid of re-standardisation windows.

- sampling : rand_generate_from_std_levels (truncated-normal, 100 draws) -> mean
- mapping  : constants.MAP midpoints {-1.5,-0.5,0,0.5,1.5} -> mean  (deterministic)

If the two window-tau curves coincide (same peak window, near-identical tau),
the continuous mapping only breaks ties and does not drive the result.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from shifting_baseline.compare import compare_corr
from shifting_baseline.constants import END, MAP, STAGE1
from shifting_baseline.data import HistoricalRecords
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.utils.calc import rand_generate_from_std_levels

REPO = Path(__file__).resolve().parent.parent
WINDOWS = np.arange(5, 60, 2)
MIN_PERIOD = 5
N_SAMPLES = 100


def _tau_curve(pred: pd.Series, true_full: pd.Series, cfg) -> dict[int, float]:
    """window -> Kendall tau (NaN if p>=0.05), for one regional H-WDI realisation."""
    out = {}
    ci = pred.index.intersection(true_full.index)
    for w in WINDOWS:
        r, p, _ = compare_corr(
            classify(pred.loc[ci], handle_na="skip"),
            true_full.loc[ci],
            window_error="nan",
            filter_func=calc_std_deviation,
            filter_side=cfg.filter_side,
            corr_method=cfg.corr_method,
            n_diff_w=0,
            window=int(w),
            min_periods=MIN_PERIOD,
        )
        out[int(w)] = r if p < 0.05 else np.nan
    return out


def _abm_curve(coll: pd.Series, climate: pd.Series, cfg) -> pd.Series:
    """window -> Kendall tau between a collective series and re-standardised climate."""
    out = {}
    for w in WINDOWS:
        r, p, _ = compare_corr(
            coll,
            climate,
            filter_func=calc_std_deviation,
            filter_side="right",
            corr_method=cfg.corr_method,
            window=int(w),
            min_periods=MIN_PERIOD,
            n_diff_w=0,
            window_error="nan",
        )
        out[int(w)] = r
    return pd.Series(out)


def abm_check(cfg):
    """Same model run, collective archive integrated two ways: stochastic
    truncated-normal sampling (the model's default) vs deterministic MAP midpoints."""
    from omegaconf import open_dict

    from shifting_baseline.abm import ClimateObservingModel

    with open_dict(cfg):
        cfg.model.years = 300  # longer single run -> smoother curve
        cfg.model.memory_baseline = "personal"  # H1, the main scenario
        cfg.model.climate_process = "ar1"
        cfg.model.mode = "test"  # suppress correlations.csv disk write

    model = ClimateObservingModel(parameters=cfg)
    model.run_model()
    climate = model.climate_df["climate"]
    coll_samp = model.climate_df["collective_memory_climate"]  # sampling (default)

    # deterministic MAP integration from the raw per-year archive records
    map_raw = pd.Series(
        {
            k: np.mean([MAP[int(x)] for x in v if not pd.isna(x)])
            for k, v in model._archive.items()
            if v
        }
    )
    yearly_index = model._aggregate_to_yearly(model.climate_series).index
    coll_map = (
        model._aggregate_to_yearly(map_raw)
        .reindex(yearly_index)
        .loc[model.spin_up_years :]
        .reindex(climate.index)
    )

    cs = _abm_curve(coll_samp, climate, cfg)
    cm = _abm_curve(coll_map, climate, cfg)
    return cs, cm


def main() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="config.yaml")

    folder = Path(cfg.ds.processed) / cfg.ds.setname
    tree_ring = pd.read_csv(folder / "integrated.csv", index_col=0)["mean"]
    true_full = pd.Series(tree_ring.loc[STAGE1:END])

    hist = HistoricalRecords(
        shp_path=cfg.ds.atlas.shp,
        data_path=cfg.ds.atlas.file,
        symmetrical_level=True,
        to_std=None,
    )
    grades = hist.data  # years x sites, integer levels in {-2..2}
    idx = grades.index

    # --- sampling: 100 truncated-normal realisations of the regional H-WDI ----
    samp = np.nanmean(
        rand_generate_from_std_levels(grades, n_samples=N_SAMPLES), axis=2
    )  # (N_SAMPLES, years)
    samp_curves = [
        _tau_curve(pd.Series(samp[i, :], index=idx).loc[STAGE1:END], true_full, cfg)
        for i in range(N_SAMPLES)
    ]
    samp_df = pd.DataFrame(samp_curves)  # rows=samples, cols=windows
    samp_mean = samp_df.mean(axis=0)

    # --- mapping: deterministic midpoint, single regional H-WDI ---------------
    gnum = grades.apply(pd.to_numeric, errors="coerce")  # object(+NA) -> float
    map_pred = gnum.replace(MAP).mean(axis=1)  # mean across sites
    map_curve = pd.Series(_tau_curve(map_pred.loc[STAGE1:END], true_full, cfg))

    # --- peaks ---------------------------------------------------------------
    sp_w = int(samp_mean.idxmax())
    sp_t = float(samp_mean.max())
    mp_w = int(map_curve.idxmax())
    mp_t = float(map_curve.max())
    common = samp_mean.dropna().index.intersection(map_curve.dropna().index)
    shape_r = float(np.corrcoef(samp_mean[common], map_curve[common])[0, 1])

    print("=== sampling vs mapping (empirical window-tau) ===")
    print(f"  sampling: peak window = {sp_w:2d} yr,  tau = {sp_t:.3f}")
    print(f"  mapping : peak window = {mp_w:2d} yr,  tau = {mp_t:.3f}")
    print(f"  curve-shape correlation (sampling vs mapping) = {shape_r:.4f}")
    print(
        f"  max |tau_sampling - tau_mapping| over windows = "
        f"{(samp_mean[common] - map_curve[common]).abs().max():.4f}"
    )

    # --- ABM-side check (Reviewer #2 asked specifically about ABM outputs) ----
    abm_s, abm_m = abm_check(cfg)
    asw, amw = int(abm_s.idxmax()), int(abm_m.idxmax())
    abm_common = abm_s.dropna().index.intersection(abm_m.dropna().index)
    abm_shape_r = float(np.corrcoef(abm_s[abm_common], abm_m[abm_common])[0, 1])
    print("\n=== sampling vs mapping (ABM collective archive) ===")
    print(f"  sampling: peak window = {asw:2d} yr,  tau = {abm_s.max():.3f}")
    print(f"  mapping : peak window = {amw:2d} yr,  tau = {abm_m.max():.3f}")
    print(f"  curve-shape correlation (sampling vs mapping) = {abm_shape_r:.4f}")
    print(
        f"  max |tau_sampling - tau_mapping| over windows = "
        f"{(abm_s[abm_common] - abm_m[abm_common]).abs().max():.4f}"
    )

    # --- figure (2 panels: empirical pipeline + ABM outputs) ------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.axvspan(20, 40, color="grey", alpha=0.10, lw=0)
    lo = samp_df.quantile(0.05, axis=0)
    hi = samp_df.quantile(0.95, axis=0)
    ax.fill_between(
        samp_mean.index, lo, hi, color="#4C72B0", alpha=0.18, label="sampling 5-95%"
    )
    ax.plot(
        samp_mean.index,
        samp_mean.values,
        color="#4C72B0",
        lw=2,
        label=f"sampling (peak {sp_w} yr)",
    )
    ax.plot(
        map_curve.index,
        map_curve.values,
        color="#DD8452",
        lw=2,
        ls="--",
        label=f"mapping (peak {mp_w} yr)",
    )
    ax.set_xlabel("re-standardisation window (yr)")
    ax.set_ylabel(r"Kendall's $\tau$")
    ax.set_title(
        f"(a) Empirical H-WDI vs N-WDI (shape $r$={shape_r:.2f})",
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=8, framealpha=0.9)

    ax2.axvspan(20, 40, color="grey", alpha=0.10, lw=0)
    ax2.plot(
        abm_s.index,
        abm_s.values,
        color="#4C72B0",
        lw=2,
        label=f"sampling (peak {asw} yr)",
    )
    ax2.plot(
        abm_m.index,
        abm_m.values,
        color="#DD8452",
        lw=2,
        ls="--",
        label=f"mapping (peak {amw} yr)",
    )
    ax2.set_xlabel("re-standardisation window (yr)")
    ax2.set_ylabel(r"Kendall's $\tau$")
    ax2.set_title(
        f"(b) ABM collective archive (shape $r$={abm_shape_r:.2f})",
        fontsize=10,
        loc="left",
    )
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Continuous mapping (sampling vs deterministic) does not drive "
        "the window-$\\tau$ pattern",
        fontsize=11,
    )
    fig.tight_layout()
    out = REPO / "reports/results/sampling_vs_mapping_robustness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
