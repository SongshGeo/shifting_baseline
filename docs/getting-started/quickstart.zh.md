# 快速上手

两个可复现入口：**基于共享数据的实证分析**，以及**自包含的 ABM**。二者都不需要原始语料——
共享序列及其格式见 [数据与复现](../guide/data.md)。

## 1. 复现实证结果（基于共享数据）

加载两条共享序列——历史档案 **H-WDI**（序数等级）与树轮 **N-WDI**（z-score）——运行滑动
窗口相关：

```python
import pandas as pd
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.compare import compare_corr

h = pd.read_csv("h_wdi.csv", index_col="year")["level"]   # H-WDI 序数等级
n = pd.read_csv("n_wdi.csv", index_col="year")["z"]        # N-WDI z-score
idx = h.index.intersection(n.index); h, n = h.loc[idx], n.loc[idx]

# 滑动窗口再标准化 + 秩相关（约 30 年最优）
r, p, k = compare_corr(
    h, n,
    filter_func=calc_std_deviation, filter_side="right",
    corr_method="kendall", window=30, min_periods=10,
)
print(f"Kendall τ (30 年窗口) = {r:.3f}  (p = {p:.3g}, n = {k})")
```

完整流程——误配/移位比较检验与窗口扫描——见 **[数据与复现](../guide/data.md)** 与
**[示例](../examples/basic-usage.md)**。

## 2. 运行 ABM（自包含）

智能体模型自行生成合成气候，无需数据即可运行。它是一个支持 `--multirun` 扫描的 Hydra 应用：

```bash
# 主分析：personal 基线（H1）下的 AR(1) 强迫
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# 判别机制：扫描四种基线
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model
```

每次运行会把图件、日志与解析后的 `.hydra/config.yaml` 写入 `outputs/`（单次）或
`multirun/`（扫描）下的时间戳目录。详见 **[智能体模型指南](../guide/abm.md)**。

## 3. 配置

ABM 参数与分析参数（如 `corr_method`、`low_pass.window_size`）由 Hydra 配置、可在命令行
覆盖——见 **[配置](configuration.md)**。
