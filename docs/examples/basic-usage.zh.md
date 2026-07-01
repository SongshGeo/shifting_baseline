# 基础用法

针对当前 API 的可直接复制片段，从两条[共享序列](../guide/data.md)出发。

## 加载共享数据

```python
import pandas as pd

h_wdi = pd.read_csv("h_wdi.csv", index_col="year")["level"]  # 序数 H-WDI（等级）
n_wdi = pd.read_csv("n_wdi.csv", index_col="year")["z"]      # 连续 N-WDI z-score

idx = h_wdi.index.intersection(n_wdi.index)                  # 共享年份
h_wdi, n_wdi = h_wdi.loc[idx], n_wdi.loc[idx]
```

## 构造可比的类别序列

```python
from shifting_baseline.filters import classify

h_cat = h_wdi                # H-WDI 本就是序数
n_cat = classify(n_wdi)      # N-WDI z-score → 5 级序数
```

## 用滑动窗口再标准化做相关

```python
from shifting_baseline.filters import calc_std_deviation
from shifting_baseline.compare import compare_corr

r, p, n = compare_corr(
    h_wdi, n_wdi,
    filter_func=calc_std_deviation,   # 对自然序列再标准化
    filter_side="right",
    corr_method="kendall",
    window=30,                        # 约 30 年最优
    min_periods=10,
)
print(f"Kendall τ = {r:.3f}  (p = {p:.3g}, n = {n})")
```

## 误配报告

```python
from shifting_baseline.calibration import MismatchReport

report = MismatchReport(
    pred=h_cat,          # 历史类别
    true=n_cat,          # 自然类别
    value_series=n_wdi,  # 底层连续值
)
report.analyze_error_patterns(mc_runs=1000)
fig = report.generate_report_figure()
```

下一步：**[进阶分析](advanced-analysis.md)**（窗口扫描 + 蒙特卡洛检验）与
**[ABM 模拟](abm-simulation.md)**。
