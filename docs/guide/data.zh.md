# 数据与复现

我们**不**公开完整的原始语料（站点级文献记录与各条树轮重建），而是共享分析真正使用的
**两条派生序列**，以便读者复现核心结果、并接入自己的数据：

| 共享序列 | 含义 | 类型 |
| --- | --- | --- |
| **H-WDI** | 由历史文献推断的**整体（区域）旱涝等级** | 序数，5 类 |
| **N-WDI** | 由树轮重建推断的**旱涝 z-score**（贝叶斯整合、已标准化） | 连续 |

它们上游的一切——读取原始档案、空间聚合、重建的贝叶斯整合——都属于*数据生产*，复现分析
并不需要。复现的入口就是这两条序列。

## 期望格式

两者都是以公元年份为索引的年度序列。一个最小的、可直接替换成你自己数据的模式：

```text
# h_wdi.csv  —— 历史档案 WDI（区域聚合）
year,level
1470,-1
1471,0
1472,2
...            # level ∈ {-2,-1,0,1,2} = {SD, MD, N, MW, SW}；约 1470–1900 CE

# n_wdi.csv  —— 自然代用 WDI（树轮 z-score）
year,z
1470,-0.42
1471,0.15
1472,1.83
...            # z = 标准化异常（均值 0、标准差 1）；约 1470–2000 CE
```

- **H-WDI** 是逐年的*区域*等级（已跨站点聚合），以序数整数编码。`-2 … 2` 对应重旱 → 重涝
  （见 [`constants`](../api/constants.md) 的 `MAP`）。
- **N-WDI** 是连续 z-score；器测前时段（约 1470–1900 CE）与 H-WDI 重叠，验证时段
  （1901–2000 CE）用于器测校验。

!!! tip "接入你自己的数据"
    任何能分成**五级序数**的档案，以及任何能表达为**以年份为索引的连续 z-score 序列**的
    代用指标，都能直接套用下面的流水线——不假设其他预处理。

## 复现核心结果

```python
import pandas as pd
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.compare import compare_corr
from shifting_baseline.calibration import MismatchReport

h = pd.read_csv("h_wdi.csv", index_col="year")["level"]   # 序数 H-WDI
n = pd.read_csv("n_wdi.csv", index_col="year")["z"]        # 连续 N-WDI z-score

# 在共享（器测前）重叠区间上对齐
idx = h.index.intersection(n.index)
h, n = h.loc[idx], n.loc[idx]

# 1) 滑动窗口再标准化 + 秩相关 → 约 30 年最优
r, p, k = compare_corr(
    h, n,
    filter_func=calc_std_deviation, filter_side="right",
    corr_method="kendall", window=30, min_periods=10,
)
print(f"Kendall τ (30 年窗口) = {r:.3f}  (p = {p:.3g}, n = {k})")

# 2) 误配 / 移位比较偏差（对比蒙特卡洛零分布）
report = MismatchReport(pred=h, true=classify(n), value_series=n)
report.analyze_error_patterns(mc_runs=1000)
fig = report.generate_report_figure()
```

- `classify(n)` 将 N-WDI z-score 转成与 H-WDI 相同的五级序数（经验阈值 ±1.17σ、±0.33σ）。
- `calc_std_deviation` 执行被检验的滑动窗口再标准化。扫描 `window`（见
  [分析流水线](pipeline.md)）即可重现 20–40 年最优。

## ABM 无需共享数据

智能体模型自行生成合成气候，因此可独立完整复现——见 **[智能体模型](abm.md)** 指南。
代际遗忘机制正是在那里与上面得到的实证窗口相互印证。
