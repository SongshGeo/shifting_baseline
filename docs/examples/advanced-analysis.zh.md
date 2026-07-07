# 进阶分析

这些示例复现核心实证结果：给出约 30 年最优的窗口扫描，以及误配偏差的蒙特卡洛显著性检验。

## 窗口扫描（约 30 年最优）

`compare.sweep_slices` 沿记录生成滚动时间窗；对每个切片，`sweep_max_corr_year` 求使相关
最大的再标准化窗口——这是标志性图背后的分析。

```python
from shifting_baseline.compare import sweep_slices, sweep_max_corr_year

# 每 20 年一个、跨度 200 年的滚动切片
slices = sweep_slices(start=1469, span=200, step=20)

# 对每个切片，求使 Kendall τ 最大的窗口尺寸
optima = sweep_max_corr_year(
    h_cat, n_wdi,
    slices=slices,
    corr_method=cfg.corr_method,
    filter_side=cfg.filter_side,
    min_periods=cfg.min_period,
)
print(optima)   # 每切片最优窗口——集中在 20–40 年带
```

!!! note
    函数签名会演进——精确的当前参数请查看由代码生成的 [`compare` API](../api/compare.md)。

## 相关曲面

```python
from shifting_baseline.compare import experiment_corr_2d

corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=h_cat,
    data2=n_wdi,
    corr_method=cfg.corr_method,
)
# corr_df：在（窗口尺寸 × 最小样本）上的 τ；r_benchmark：不再标准化的基准
```

## 蒙特卡洛显著性（[`calibration`](../api/calibration.md)）

移位比较偏差以一个零分布检验：在固定边际频率下独立重抽档案等级。该检验内置于
[`MismatchReport`][shifting_baseline.calibration.MismatchReport]，由 `mc_runs`（默认 **1000**）控制：

```python
from shifting_baseline.calibration import MismatchReport

report = MismatchReport(pred=h_cat, true=classify(n_wdi), value_series=n_wdi)
report.analyze_error_patterns(mc_runs=1000)   # 相对零分布的逐格 z / p 值
fig = report.generate_report_figure()
```

## 数据情景稳健性

在不同代用整合情景（`ds=pure|best|…`）或不同验证数据（`using_val_data=china|gpcc|cru`）下
重跑，以确认模式并非单一数据集的伪影：

```bash
uv run python -m shifting_baseline ds=best using_val_data=gpcc
```

下一步：**[ABM 模拟](abm-simulation.md)**。
