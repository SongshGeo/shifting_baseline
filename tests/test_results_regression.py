#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Regression guardrail for the manuscript's headline numbers (``results.json``).

This locks the numbers quoted in the paper's Results section so that any refactor
of the underlying package (``compare`` / ``calibration`` / ``filters`` / ``data`` …)
that silently moves a value turns this test red. It reproduces the numbers through
:func:`shifting_baseline.results.build_results`, the same code path the figure
notebooks use.

The test needs the real datasets (the cached ``${ds.processed}/*.csv`` intermediates
and the historical atlas). Those are gitignored / multi-GB and absent on CI or a
fresh clone, so the test **skips cleanly** when the inputs are missing. Run it
locally before and after a refactor:

    uv run pytest tests/test_results_regression.py -m slow -v

All three blocks are regenerated deterministically (seeded by ``cfg.random_seed``)
and asserted exactly against the committed golden file.
"""

import json
from pathlib import Path

import matplotlib
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

import shifting_baseline  # noqa: F401 — import loads .env so ${oc.env:...} resolves

matplotlib.use("Agg")  # headless: experiment_corr_2d draws a heatmap

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
GOLDEN_PATH = Path(__file__).parent / "data" / "golden_results.json"

pytestmark = pytest.mark.slow


def _load_real_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    OmegaConf.resolve(cfg)
    return cfg


def _required_inputs_present(cfg) -> bool:
    """True only when the cached intermediates + atlas this test needs exist locally."""
    val = cfg.ds.validation[cfg.using_val_data]
    candidates = [
        cfg.ds.out.tree_ring,  # combined_mean.csv (N-WDI posterior)
        cfg.ds.out.tree_ring_uncertainty,  # combined_uncertainty.csv
        val.csv,  # regional validation z-score
        val.z_nc,  # gridded validation z-score
        cfg.ds.atlas.file,  # historical drought/flood atlas
    ]
    return all(Path(p).exists() for p in candidates)


@pytest.fixture(scope="module", name="cfg")
def fixture_real_cfg():
    try:
        cfg = _load_real_cfg()
    except OmegaConfBaseException:
        # ${oc.env:SBS_*} paths unresolved (no .env / env vars) — nothing to test
        pytest.skip(
            "SBS_* env/.env not configured; results regression needs local data."
        )
    if not _required_inputs_present(cfg):
        pytest.skip(
            "results.json regression needs local datasets (cached CSVs + atlas); "
            "run locally with the data present."
        )
    return cfg


@pytest.fixture(scope="module", name="golden")
def fixture_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def test_results2_matches_manuscript(cfg, golden):
    """H-WDI vs N-WDI confusion statistics (kappa/tau/accuracy/mean_diff/…)."""
    from shifting_baseline.results import compute_results2

    assert compute_results2(cfg) == golden["results2"]


def test_results3_matches_manuscript(cfg, golden):
    """Sliding-window re-standardisation gains (tau/w_optimal/optimal_segment_w/…)."""
    from shifting_baseline.results import compute_results3

    assert compute_results3(cfg) == golden["results3"]


def test_results1_matches_manuscript(cfg, golden):
    """Reconstruction/validation summary (n_datasets/corr/site-significance/…)."""
    from shifting_baseline.results import compute_results1

    assert compute_results1(cfg) == golden["results1"]
