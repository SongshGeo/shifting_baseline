# Installation

The project targets **Python 3.11 only** and manages dependencies with
[**uv**](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

## Requirements

- Python 3.11 (exactly)
- `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git

## Install

```bash
git clone https://github.com/SongshGeo/shifting_baseline
cd shifting_baseline

uv sync                 # runtime + dev dependencies
uv sync --group docs    # add the docs toolchain (mkdocs, mkdocstrings, i18n …)
```

`uv sync` creates a `.venv/`. Run commands with `uv run …`, or activate the env:

```bash
source .venv/bin/activate
```

The scientific stack (xarray, pandas, numpy, scipy, arviz, pymc, geopandas,
cartopy, netcdf4, hydra-core, and the ABM framework
[`abses`](https://pypi.org/project/abses/)) is installed automatically from
`pyproject.toml` / `uv.lock`.

## Verify

```bash
uv run python -c "import shifting_baseline; print('ok')"
uv run pytest -m "not slow"     # quick smoke test (skips slow-marked tests)
```

## Build the docs locally

```bash
uv run mkdocs serve             # live preview at http://127.0.0.1:8000
uv run mkdocs build             # static site into ./site
```

The site is bilingual (English + 中文) — use the language selector in the top bar.

## Troubleshooting

- **`uv` out of date** — `uv self update`, then `uv sync --refresh`.
- **NetCDF / Cartopy build errors** — the wheels usually just work; if not, install
  system libs (`libhdf5-dev libnetcdf-dev libproj-dev libgeos-dev`) or use a
  conda-forge base, then re-run `uv sync`.
