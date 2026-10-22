# Shifting Baseline

Research codebase for Song et al. (2026), *Nature Communications*
([doi:10.1038/s41467-026-78922-x](https://doi.org/10.1038/s41467-026-78922-x)):
*“Archival and palaeoenvironmental documentation of extreme climate events reveals
generational amnesia in collective memory.”*

It compares historical Chinese climate archives (1470–1900 CE) against tree-ring
hydroclimate reconstructions and instrumental data (1901–2000 CE) to study
**Shifting Baseline Syndrome (SBS)**, and uses an **agent-based model (ABM)** to
reproduce the observed perceptual-bias pattern.

!!! tip "Language"
    Use the language selector in the top bar to switch between **English** and **中文**.
    使用顶部的语言选择器在 **English** 与 **中文** 之间切换。

## What this project does

- **Two Wet/Dry Index (WDI) series.** The perception-mediated **H-WDI** (historical
  archives) is compared against the more independent **N-WDI** (natural tree-ring
  proxy), both classified into five ordinal levels (SD, MD, N, MW, SW).
- **Mismatch analysis.** Where the two disagree, the archive shows a systematic
  *shifted-comparison* bias — years are judged relative to recent experience rather
  than to an absolute baseline.
- **Sliding-window re-standardisation.** Re-standardising the natural series within a
  ~20–40-year (optimal ≈ 30-year) window improves agreement — the central finding.
- **Agent-based model.** A minimal ABM discriminates the two SBS mechanisms
  (**H1 generational amnesia** vs **H2 collective illusion**); only generational
  amnesia reproduces the empirical generational optimum. Robustness is established
  with AR(1) climate forcing and a global (Sobol) sensitivity analysis.

## Reproduce it

We share the **two derived series** the analysis actually uses — the historical-archive
**H-WDI** (ordinal levels) and the tree-ring **N-WDI** (z-score) — so you can reproduce
the central result and plug in your own data. The raw corpus and the data-production
steps are not required. Start at **[Data & Reproduction](guide/data.md)**.

## Analysis modules

| Module | Role |
| --- | --- |
| [`filters`](api/filters.md) | Sliding-window re-standardisation & ordinal classification |
| [`compare`](api/compare.md) | Correlation machinery & the window sweep (~30-year optimum) |
| [`calibration`](api/calibration.md) | `MismatchReport` confusion-matrix / error-pattern analysis + MC null |
| [`abm`](api/abm.md) | `ClimateObservingModel` and observer agents |
| [`climate_forcing`](api/climate_forcing.md) | i.i.d. / AR(1) / trend climate generators |
| [`sensitivity`](api/sensitivity.md) | Variance-based Sobol sensitivity analysis |
| [`constants`](api/constants.md), [`utils`](api/utils.md) | Year boundaries, maps, plotting, logging |

## Quick start

```bash
uv sync                                   # install runtime + dev deps
uv run python shifting_baseline/abm.py    # run the ABM (self-contained, no data needed)
uv run pytest                             # run the test suite
```

See **[Getting Started](getting-started/installation.md)** to install, the
**[User Guide](guide/overview.md)** for the science, and
**[Data & Reproduction](guide/data.md)** to run the analysis from the shared series.

## Citation

```bibtex
@article{song2026archival,
  title     = {Archival and palaeoenvironmental documentation of extreme climate events reveals generational amnesia in collective memory},
  author    = {Song, Shuang and Hu, Bo and Findley, David Max and Wu, Xutong and Liu, Wei and Middeke-Conlin, Robert W. and Caetano-Andrade, Victor L. and Yang, Yuda and Wang, Shuai and Renn, J{\"u}rgen and Roberts, Patrick},
  journal   = {Nature Communications},
  year      = {2026},
  doi       = {10.1038/s41467-026-78922-x},
  url       = {https://doi.org/10.1038/s41467-026-78922-x}
}
```

## Contributing

This is a research codebase under active development. If you're interested in
contributing — extending the ABM, the analysis pipeline, or building a front end —
please get in touch:

- 📧 **song[at]gea.mpg.de**
- 🌐 [cv.songshgeo.com](https://cv.songshgeo.com/)
- 🐛 [GitHub issues](https://github.com/SongshGeo/shifting_baseline/issues)
