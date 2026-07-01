# 安装

本项目**仅支持 Python 3.11**，并使用 [**uv**](https://docs.astral.sh/uv/)
（`pyproject.toml` + `uv.lock`）管理依赖。

## 环境要求

- Python 3.11（严格）
- `uv` —— `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git

## 安装

```bash
git clone https://github.com/SongshGeo/shifting_baseline
cd shifting_baseline

uv sync                 # 运行时 + 开发依赖
uv sync --group docs    # 追加文档工具链（mkdocs、mkdocstrings、i18n 等）
```

`uv sync` 会创建 `.venv/`。可用 `uv run …` 运行命令，或激活环境：

```bash
source .venv/bin/activate
```

科学计算栈（xarray、pandas、numpy、scipy、arviz、pymc、geopandas、cartopy、
netcdf4、hydra-core，以及 ABM 框架 [`abses`](https://pypi.org/project/abses/)）会依据
`pyproject.toml` / `uv.lock` 自动安装。

## 验证

```bash
uv run python -c "import shifting_baseline; print('ok')"
uv run pytest -m "not slow"     # 快速冒烟测试（跳过 slow 标记）
```

## 本地构建文档

```bash
uv run mkdocs serve             # 实时预览 http://127.0.0.1:8000
uv run mkdocs build             # 生成静态站点到 ./site
```

站点为中英双语——使用顶部语言选择器切换。

## 常见问题

- **`uv` 版本过旧** —— `uv self update`，再 `uv sync --refresh`。
- **NetCDF / Cartopy 构建报错** —— 通常预编译轮子即可；若失败，安装系统库
  （`libhdf5-dev libnetcdf-dev libproj-dev libgeos-dev`）或使用 conda-forge 基础环境后
  重新 `uv sync`。
