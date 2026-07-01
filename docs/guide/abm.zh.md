# 智能体模型 (ABM)

ABM（[`abm.ClimateObservingModel`][shifting_baseline.abm]）是一个刻意极简的启发式模型，
用途是**机制判别**：实证的约 30 年最优究竟来自*代际遗忘*还是*集体错觉*？

## 运行机制

一个不断更替、寿命有限的观察者群体注视着逐年气候信号。每年，主体**相对于某个基线**感知
当前异常，偏离越大越可能记录。被记录的事件汇入共享档案，聚合并分级为与实证相同的五级
WDI，从而模拟记录与真实记录以完全相同的方式比较。

- **感知：** `z = (cₜ − μ) / σ`，其中 `μ, σ` 来自所选基线。
- **记录倾向：** `p = f₀ + 0.5 − Φ(|z|)` —— 负性偏差（极端更易被记录）。
- **档案丢失：** 每条记录以概率 `1 − loss_rate` 保留。
- **人口：** 每年进入 `new_agents` 个主体；超过 `max_age` 的主体离开。

## 气候强迫 —— [`climate_forcing`](../api/climate_forcing.md)

`climate_process` 选择驱动过程：

| 过程 | 含义 |
| --- | --- |
| `iid` | 独立高斯噪声——"零"气候（配置默认） |
| `ar1` | AR(1) 持续性 `cₜ = φ·cₜ₋₁ + εₜ`（φ = `climate_phi`）——论文的**主分析** |
| `trend_plus_noise` | 线性趋势 + 噪声（均值低频漂移） |
| `ar1_trend` | AR(1) 持续性 + 线性趋势 |

变率为 `climate_sigma`。强迫矩阵（i.i.d. / AR(1) / 趋势，年与亚年分辨率）是稳健性检验——
代际最优在所有情形下都成立。

## 2×2 基线设计

`memory_baseline` 交叉两个因子——参照的*来源*（自身经验 vs 共享档案）与其*时间视野*
（自身寿命 vs 全部记录）：

| | 代际视野（单一寿命） | 累积视野（全部记录） |
| --- | --- | --- |
| **个体**（自身记忆） | `personal` —— **H1：代际遗忘** | *(理想极限 → `model`)* |
| **集体**（共享档案） | `collective_lifetime` | `collective` —— **H2：集体错觉** |

- `personal`（**H1**）—— 相对于主体自身亲历记忆再标准化。**只有它复现 20–40 年代际最优。**
- `collective`（**H2**）—— 使用全部累积档案；相关随窗口单调上升，无代际峰。
- `collective_lifetime` —— 共享档案但限于寿命窗口；表现同 H2，说明该效应需要*个体经验*，
  而非仅仅有限视野。
- `model` —— 参照客观气候本身；非感知基准。

## 全局敏感性 —— [`sensitivity`](../api/sensitivity.md)

对五个参数做基于方差的 **Sobol** 分析（Saltelli 采样）：

| 参数 | 范围 |
| --- | --- |
| `max_age`（寿命） | 15–80 年 |
| `new_agents` | 1–15 |
| `loss_rate` | 0–0.8 |
| `climate_sigma` | 0.5–2.0 |
| `climate_phi` | 0.0–0.9 |

在 H1 下，最优窗口位置几乎完全由**主体寿命**决定（`max_age` 的总效应 `S_T ≈ 1`）并与之
成正比——一个真正代际性的成因。在 H2 下，没有任何参数能复现代际尺度的最优。

## 运行

```bash
# 主分析：personal 基线（H1）下的 AR(1) 强迫
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# 基线扫描
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model
```

默认值（`config/model/exp.yaml`）：`max_age=40`、`min_age=10`、`new_agents=5`、
`loss_rate=0.4`、`memory_baseline=personal`、`repeats=100`。类与方法见
[`abm` API](../api/abm.md)。
