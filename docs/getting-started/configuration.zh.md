# 配置 (Hydra)

所有运行行为由 [Hydra](https://hydra.cc/) 控制。`config/config.yaml` 组合三个可切换
配置组，并暴露若干被流水线反复读取的顶层参数。

## 组合方式

```yaml
defaults:
  - ds: pure        # 数据源
  - how: process    # __main__ 运行哪种分析
  - model: exp      # ABM 参数
```

| 组 | 选项（`config/<组>/*.yaml`） | 含义 |
| --- | --- | --- |
| `ds` | `pure`（默认）、`best`、`mac` | 档案、重建、验证数据（`china`/`gpcc`/`cru`）、PMIP 输出的路径 |
| `how` | `process`（默认）、`correlation`、`compare` | 名义上的分析选择器——**注意**：`__main__` 目前不读 `cfg.how`（仅 `process.py` 的独立 CLI 读取），故覆盖它不会改变主流程 |
| `model` | `exp`（默认）、`test` | ABM 参数（`exp` 做实验、`test` 快速运行） |

命令行选择组，例如 `ds=best model=test`。

## 顶层参数

这些参数贯穿整个流水线（并通过 `${...}` 插值进 ABM 配置，改一处即全局生效）：

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `corr_method` | `kendall` | 相关方法（`pearson`/`kendall`/`spearman`） |
| `filter_side` | `right` | 滑动窗口滤波取哪一侧 |
| `agg_method` | `mean` | 档案等级的跨站点聚合 |
| `to_std` | `sampling` | 离散→连续映射（`sampling`=截断正态；否则用中点） |
| `resolution` | `0.5` | 验证数据网格分辨率 |
| `min_period` | `10` | 每个滚动窗口的最小样本数 |
| `ratio` | `0.10` | 极值区间的前百分比 |
| `low_pass.window_size` | `30` | 约 30 年滑动窗口——**核心发现，牵一发动全身** |
| `using_val_data` | `china` | 验证数据（`china`/`gpcc`/`cru`） |
| `violin_windows` | `[20, 40, 60]` | 小提琴图的窗口尺寸 |
| `random_seed` | `42` | 固定随机种子以复现采样/贝叶斯整合（设 `null` 则随机） |
| `test_mode` | `false` | 仅日志的冒烟测试 |
| `recalculate_data` | `false` | 重算中间量，而非复用 `${ds.processed}/*.csv` |

## 覆盖参数

```bash
# 修改参数
uv run python -m shifting_baseline low_pass.window_size=25 corr_method=spearman

# 追加所选组里不存在的键（Hydra struct 模式）
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# multirun 扫描
uv run python shifting_baseline/abm.py --multirun model.max_age=30,40,50
```

!!! warning "struct 模式"
    覆盖所选配置中不存在的键会报错。用 `+键=值` 语法**追加**新键（例如所选 `model`
    组未定义时用 `+model.climate_process=ar1`）。

## 运行产物

每次运行在 `outputs/`（单次）或 `multirun/`（扫描）下创建时间戳目录，包含图件、日志与
`.hydra/config.yaml`——即该次运行的解析配置，是唯一真源。
