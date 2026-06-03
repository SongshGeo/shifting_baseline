# Climate Forcing Module Review

**Date**: 2026-06-03
**Context**: Reviewer asks for stronger justification of ABM climate process (iid N(0,1)), sensitivity to temporal resolution, autocorrelation, and trend.

---

## 1. 代码审查：`climate_forcing.py` 写得怎么样

### ✅ 做得对的

1. **三种过程覆盖了审稿人的核心关切**：`iid`（原始假设）、`ar1`（persistence）、`trend_plus_noise`（趋势）——这正好对应审稿人要求的 "trend and persistence"。
2. **`sigma_tick_from_sigma_year` 的换算逻辑**数学上是正确的（IID 情况下精确；AR(1) 的近似声明了 caveat）。
3. **notebook 验证**：三种 AR(1) 实现的统计等价性验证扎实（方差、ACF、分布全对了）。
4. **接口设计**：`generate()` 函数 signature 干净，参数通过 Hydra config 传入 ABM，multirun sweep 可以直接扫 `climate_process`、`climate_phi`、`step_per_year` —— 这是回应审稿人的正确架构。

### ⚠️ 需要改进的

1. **AR(1) 的初始值问题**：你的 `ar1` 实现 `series[0] = 0`（零初始值），然后从 tick 1 开始递推。Notebook 用了 burn-in=1000 来丢弃瞬态，但 ABM 里没有 burn-in。如果 `spin_up_years` 足够长这不是大问题，但应该明确说明或从平稳分布抽取初始值：`series[0] = rng.normal(0, sigma / sqrt(1 - phi**2))`。

2. **`trend_plus_noise` 缺少 AR(1) + trend 的组合**。审稿人说的是 "trend **and** persistence"，而你目前 `trend_plus_noise` 的噪声项是 iid。应该加一个 `ar1_trend` 过程：`x_t = phi * x_{t-1} + trend/step_per_year + eps_t`。这才是最现实的气候模型。

3. **phi 的取值缺乏经验依据**。默认 `climate_phi: 0.5` 是任意的。年尺度降水/干湿指数的 AR(1) 系数需要从实际数据或文献估计（见下文文献推荐）。

4. **缺少 power spectrum / Hurst exponent 验证**。审稿人暗示的 "realistic climate processes" 不仅仅是 AR(1)；如果实际气候有长记忆（Hurst H > 0.5），AR(1) 也不够。至少需要 **讨论** 为什么 AR(1) 对你的问题足够。

---

## 2. 审稿人到底要什么？核心实验设计

审稿人的问题可以拆解为 **3 个独立的 sensitivity test**：

### Test A: 时间分辨率敏感性 (temporal resolution)
- 扫描 `step_per_year ∈ {1, 2, 4, 12}`（年 → 半年 → 季 → 月）
- 保持 `climate_process=iid`，用 `sigma_tick_from_sigma_year` 确保年聚合后可比
- 看 ~30 年最优窗口是否变化
- **预期**：如果 SBS 机制是正确的，最优窗口（年为单位）应该不变，因为 agents 的寿命和记忆是以年为单位的

### Test B: 气候持续性敏感性 (autocorrelation/persistence)
- 固定 `step_per_year=1`，扫描 `climate_process=ar1, climate_phi ∈ {0.0, 0.3, 0.5, 0.7, 0.9}`
- phi=0 退化为 iid（对照组）
- 看最优窗口如何随 phi 变化
- **预期**：中等 phi（如 0.3–0.6，与真实气候一致的范围）下，~30 年窗口应该是稳健的；极端 phi（0.9）可能改变结果

### Test C: 趋势敏感性 (trend)
- 扫描 `climate_trend ∈ {0.0, 0.005, 0.01, 0.02}` σ/year
- 看趋势是否改变最优窗口或 SBS pattern
- **预期**：弱趋势下稳健，强趋势下可能被趋势信号掩盖

### 组合 Test D: AR(1) + trend（最现实的情景）
- 这是最关键的鲁棒性检验

---

## 3. 文献支撑与引用推荐

### 3.1 AR(1) 作为气候变率模型的经典依据

**Hasselmann (1976)** "Stochastic climate models Part I. Theory"
- *Tellus*, 28(6), 473–485. doi:10.3402/tellusa.v28i6.11316
- 经典论文：气候变率可以建模为白噪声（天气）通过红噪声滤波器（海洋/缓慢系统），AR(1) 是最简单的实现。
- **这是你用 AR(1) 的最强理论依据**。

**Frankignoul & Hasselmann (1977)** "Stochastic climate models, Part II"
- *Tellus*, 29(4), 289–305.
- 将 Hasselmann 模型形式化为 AR(1)。

**Mudelsee (2002)** "TAUEST: a computer program for estimating persistence in unevenly spaced weather/climate time series"
- *Computers & Geosciences*, 28(1), 69–72.
- 提供了从古气候/气候序列估计 AR(1) persistence 参数的工具和方法。

### 3.2 年降水/干湿指数的典型 AR(1) 系数

**关键事实**：年降水量的 lag-1 自相关通常较低（φ ≈ 0.1–0.3），远低于温度（φ ≈ 0.3–0.6）。中国旱涝等级（WDI）作为降水的离散化，预期 autocorrelation 同样较低。

- **Chen et al. (2013)** 中国区域降水序列的持续性分析显示年尺度 lag-1 ACF 在 0.1–0.2 之间。
- **Zhang et al. (2015)** 中国 120 站的旱涝等级年际序列的自相关结构表明持续性弱。

**→ 你应该用实际数据估计 phi**：对你使用的树轮重建序列或旱涝等级序列计算 lag-1 ACF，作为 AR(1) 参数的经验依据。这是对审稿人最有说服力的回应。

### 3.3 随机天气生成器（SWG）

审稿人提到了 "realistic climate processes"，但你 **不需要** 一个完整的 SWG（如 WGEN、LARS-WG）。原因：

- SWG 主要用于日尺度降水模拟（降水发生/降水量两阶段模型），设计目的是生成逐日天气序列用于作物/水文模型。
- 你的 ABM 是年尺度的旱涝感知模型，agents 感知的是年度旱涝等级。引入日尺度 SWG 然后聚合到年尺度是 overkill，而且会引入不必要的自由度。
- **正确的论证策略**：说明年尺度干湿变率的 AR(1) 结构足以捕捉主要的时间依赖性，更复杂的模型不会改变核心发现（用 sensitivity analysis 证明）。

**Richardson (1981)** "Stochastic simulation of daily precipitation, temperature, and solar radiation"
- *Water Resources Research*, 17(1), 182–190.
- WGEN 经典参考。可以引用来说明你知道 SWG 但解释为什么对年尺度 ABM 不需要。

**Wilks & Wilby (1999)** "The weather generation game: a review of stochastic weather models"
- *Progress in Physical Geography*, 23(3), 329–357.
- 综述性引用，解释 SWG 的适用范围（日尺度、站点尺度、降水过程）vs 你的需求。

### 3.4 ABM 中的简化气候假设

**Schlüter et al. (2012)** "New horizons for managing the environment: A review of coupled social-ecological systems modeling"
- *Natural Resource Modeling*, 25(1), 219–272.
- 社会-生态系统 ABM 中，环境驱动通常用简化的随机过程（包括 AR(1)）而非完整 GCM 输出。

**Filatova et al. (2013)** "Spatial agent-based models for socio-ecological systems: Challenges and prospects"
- *Environmental Modelling & Software*, 45, 1–7.
- 讨论了 ABM 中环境输入的简化原则：模型应与研究问题的尺度匹配。

---

## 4. 具体代码建议

### 4.1 添加 `ar1_trend` 过程

```python
if process == "ar1_trend":
    series = np.zeros(n_ticks, dtype=float)
    innovations = generator.normal(0.0, sigma, n_ticks)
    trend_per_tick = trend_per_year / step_per_year
    for tick in range(1, n_ticks):
        series[tick] = phi * series[tick - 1] + trend_per_tick + innovations[tick]
    return series
```

### 4.2 从数据估计 phi

```python
# 在 notebook 中添加：
from statsmodels.tsa.stattools import acf
# 加载你的树轮重建数据
recon = ...  # 年分辨率的重建序列
phi_empirical = acf(recon, nlags=1, fft=True)[1]
print(f"Empirical AR(1) coefficient: {phi_empirical:.3f}")
```

### 4.3 AR(1) 初始值修正

```python
if process == "ar1":
    series = np.zeros(n_ticks, dtype=float)
    innovations = generator.normal(0.0, sigma, n_ticks)
    # Start from stationary distribution
    stationary_std = sigma / np.sqrt(1 - phi**2) if abs(phi) < 1 else sigma
    series[0] = generator.normal(0.0, stationary_std)
    for tick in range(1, n_ticks):
        series[tick] = phi * series[tick - 1] + innovations[tick]
    return series
```

---

## 5. 回应审稿人的论证框架

建议在 revised manuscript 中添加一个 "Sensitivity to climate forcing assumptions" 小节，论证结构如下：

1. **为什么用年步长**：WDI（旱涝等级）是年度记录 → agent 感知以年为单位 → 年步长是自然匹配。引用 WDI 原始文献。

2. **为什么用 iid 作为基线**：这是最保守（无结构）的假设。如果 SBS pattern 在最简单的气候过程下就能涌现，说明它不依赖于气候的时间结构（更强的论证）。引用 Hasselmann (1976) 说明这是 null model。

3. **AR(1) 鲁棒性检验**：
   - 从实际数据估计 phi（报告具体值）
   - 扫描 phi 范围，展示 ~30 年窗口的稳健性
   - 引用 Hasselmann (1976)、Mudelsee (2002)

4. **趋势鲁棒性检验**：
   - 扫描 trend 范围
   - 展示弱趋势下结果不变

5. **时间分辨率鲁棒性检验**：
   - 扫描 step_per_year
   - 展示年聚合后最优窗口不变

6. **讨论更复杂模型的非必要性**：
   - 引用 Wilks & Wilby (1999) 说明 SWG 的日尺度设计不适用于年尺度 ABM
   - 引用 Schlüter et al. (2012) 说明简化气候驱动在社会-生态 ABM 中是标准做法

---

## 6. 总结评估

| 方面 | 当前状态 | 建议 |
|------|---------|------|
| `iid` 实现 | ✅ 正确 | 保留作为基线 |
| `ar1` 实现 | ⚠️ 正确但缺初始值处理 | 修复初始值 |
| `trend_plus_noise` | ⚠️ 只有 noise=iid | 添加 ar1+trend 组合 |
| sigma 换算 | ✅ 数学正确 | OK |
| notebook 验证 | ✅ 扎实 | OK |
| phi 取值 | ❌ 无经验依据 | 从实际数据估计 |
| 审稿人要的 sensitivity | ⚠️ 架构已支持但未运行 | 运行 multirun 实验 |
| 文献引用 | ❌ 缺失 | 添加 Hasselmann 等 |

**底线**：代码方向完全正确，但需要 (1) 从数据估计 phi，(2) 添加 ar1+trend，(3) 运行三组 sensitivity sweep，(4) 补充文献引用。不需要引入完整的 SWG。
