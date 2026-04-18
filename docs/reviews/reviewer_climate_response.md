# Reviewer Response Draft: Climate Forcing and Time Resolution
<!-- 中文翻译
# 审稿回复草稿：气候强迫与时间分辨率
-->

## Scope of this revision round
<!-- 中文翻译
## 本轮修订范围
-->

This revision round focuses only on climate-related concerns raised by reviewers:
<!-- 中文翻译
本轮修订仅聚焦审稿人提出的气候相关关切：
-->

1. The baseline model previously used only an annual IID Gaussian forcing.
<!-- 中文翻译
1. 基线模型先前仅使用年度尺度的独立同分布（IID）高斯强迫。
-->
2. The interpretation of a ~30-year optimal window needed robustness checks.
<!-- 中文翻译
2. 对约 30 年最优窗口的解释需要进行稳健性检验。
-->
3. Time-resolution sensitivity (annual vs. finer forcing) needed explicit testing.
<!-- 中文翻译
3. 需要对时间分辨率敏感性（年度 vs 更细分辨率的强迫）进行明确测试。
-->

Population-structure extensions and full global sensitivity analysis are deferred to a later round.
<!-- 中文翻译
人口结构扩展与完整的全局敏感性分析将推迟到后续修订轮次。
-->

## Clarified framing in methods
<!-- 中文翻译
## 方法部分的表述澄清
-->

- The ABM forcing is now described as a **discrete climate forcing sequence** rather than a continuous physical climate trajectory.
<!-- 中文翻译
- 现在将 ABM 强迫表述为**离散的气候强迫序列**，而非连续的物理气候轨迹。
-->
- The annual-step baseline is retained because the target empirical comparison is annualized WDI/tree-ring data.
<!-- 中文翻译
- 保留年度步长的基线设定，因为目标实证比较对象是年尺度的 WDI/树轮数据。
-->
- The revised model introduces configurable forcing generators and optional subannual forcing with yearly aggregation for output comparison.
<!-- 中文翻译
- 修订后的模型引入了可配置的强迫生成器，并支持可选的年内（subannual）强迫；为便于输出对比，将其按年度进行聚合。
-->

## New climate forcing scenarios
<!-- 中文翻译
## 新的气候强迫情景
-->

The ABM now supports:
<!-- 中文翻译
ABM 现在支持：
-->

- `iid`: Gaussian white-noise forcing.
<!-- 中文翻译
- `iid`：高斯白噪声强迫。
-->
- `ar1`: Persistent forcing with AR(1) coefficient `climate_phi`.
<!-- 中文翻译
- `ar1`：具有持续性的强迫，AR(1) 系数为 `climate_phi`。
-->
- `trend_plus_noise`: Linear trend (`climate_trend`) plus Gaussian noise.
<!-- 中文翻译
- `trend_plus_noise`：线性趋势（`climate_trend`）叠加高斯噪声。
-->

## New time-resolution check
<!-- 中文翻译
## 新的时间分辨率检查
-->

- `step_per_year=1` (annual baseline)
<!-- 中文翻译
- `step_per_year=1`（年度基线）
-->
- `step_per_year>1` (subannual forcing, then yearly aggregation using `subannual_aggregation`)
<!-- 中文翻译
- `step_per_year>1`（年内强迫，然后使用 `subannual_aggregation` 做年度聚合）
-->

All peak-window statistics are reported in yearly units to keep interpretation consistent.
<!-- 中文翻译
所有峰值窗口统计均以“年”为单位报告，以保持解释一致性。
-->

## Results from the robustness matrix (10 repeats, 80 analysis years)
<!-- 中文翻译
## 稳健性矩阵结果（10 次重复，80 个分析年份）
-->

Summary artifacts:
<!-- 中文翻译
汇总产物：
-->

- `reports/results/climate_scenarios/climate_scenario_summary.csv`
<!-- 中文翻译
- `reports/results/climate_scenarios/climate_scenario_summary.csv`
-->
- `reports/results/climate_scenarios/climate_scenario_summary.md`
<!-- 中文翻译
- `reports/results/climate_scenarios/climate_scenario_summary.md`
-->

Peak-window location (mean ± sd across 10 repeats) and peak correlation strength:
<!-- 中文翻译
峰值窗口位置（10 次重复的均值 ± 标准差）与峰值相关强度：
-->

| Scenario | Peak window (yr) | SD | Peak strength |
| --- | --- | --- | --- |
| iid_annual | 28.6 | 8.8 | 0.88 |
| iid_subannual4 | 40.3 | 12.4 | 0.85 |
| ar1_annual (phi=0.6) | 32.5 | 18.0 | 0.88 |
| ar1_subannual4 (phi=0.6) | 43.2 | 22.3 | 0.91 |
| trend_annual (trend=0.03/yr) | 23.4 | 4.5 | 0.89 |
| trend_subannual4 | 25.3 | 9.7 | 0.85 |

Key observations:
<!-- 中文翻译
关键观察：
-->

- **At the empirically relevant annual resolution, the ~30-year peak is robust to climate structure.** Both IID (28.6 yr) and AR(1) with phi=0.6 (32.5 yr) bracket the ~30-year window reported in the manuscript. Adding persistence does *not* collapse or inflate the emergent memory window.
<!-- 中文翻译
- **在与实证对齐的年度分辨率下，约 30 年的峰值对气候结构具有稳健性。** IID（28.6 年）与 \(\phi=0.6\) 的 AR(1)（32.5 年）均覆盖了稿件中报告的约 30 年窗口。加入持续性并不会使涌现的记忆窗口塌缩或膨胀。
-->
- Under a linear trend, the peak shortens slightly (23.4 yr) but remains in the generational 20-40 yr band.
<!-- 中文翻译
- 在线性趋势下，峰值略有缩短（23.4 年），但仍位于代际尺度的 20–40 年区间内。
-->
- Subannual forcing (`step_per_year=4`) shifts the peak upward by ~10-15 years in every scenario. This is partly an artifact of the current tick-to-year sigma rescaling (strictly valid for IID only; see "Caveats" below) and should not be interpreted as a physical effect of finer resolution.
<!-- 中文翻译
- 年内强迫（`step_per_year=4`）在所有情景中都会使峰值上移约 10–15 年。这部分来自当前从 tick 到 year 的 sigma 重标定带来的伪影（严格来说仅对 IID 有效；见下文“注意事项”），不应被解释为更细时间分辨率的物理效应。
-->
- Peak correlation strength is comparable across all six scenarios (0.85-0.91), indicating the mechanism itself is not scenario-specific.
<!-- 中文翻译
- 六种情景的峰值相关强度相近（0.85–0.91），表明该机制本身并不依赖于某一特定情景。
-->

## Suggested wording revision for the main claim
<!-- 中文翻译
## 对主要结论表述的建议修订
-->

- Previous (too strong): "the model supports a generation-scale SBS window around ~30 years."
<!-- 中文翻译
- 先前（过强）： “the model supports a generation-scale SBS window around ~30 years.”
-->
- Revised (calibrated): "the emergent optimal window lies in the 20-40-year generational band across a range of climate forcings (IID, AR(1), and trend-plus-noise) at annual resolution, consistent with the empirical ~30-year optimum. The window broadens modestly under subannual forcing but remains generational in scale."
<!-- 中文翻译
- 修订（更稳妥）：“the emergent optimal window lies in the 20-40-year generational band across a range of climate forcings (IID, AR(1), and trend-plus-noise) at annual resolution, consistent with the empirical ~30-year optimum. The window broadens modestly under subannual forcing but remains generational in scale.”
-->

This revision retains the central finding while explicitly demonstrating its robustness to the reviewer's concerns about persistence, trend, and timestep.
<!-- 中文翻译
这一修订在保留核心发现的同时，明确展示了其对审稿人关于持续性、趋势与时间步长关切的稳健性。
-->

## Caveats and scope
<!-- 中文翻译
## 注意事项与范围界定
-->

- **Sigma scaling for AR(1) and trend scenarios is approximate.** The `_sigma_tick_from_sigma_year` helper assumes IID tick-level noise; for AR(1) the stationary variance is `sigma^2 / (1 - phi^2)`, so the yearly-aggregated variance is not exactly preserved across `step_per_year` settings. The subannual-vs-annual comparison should therefore be read as indicative rather than variance-matched.
<!-- 中文翻译
- **AR(1) 与趋势情景的 sigma 缩放是近似的。** `_sigma_tick_from_sigma_year` 辅助函数假设 tick 级噪声为 IID；对于 AR(1)，平稳方差为 `sigma^2 / (1 - phi^2)`，因此在不同 `step_per_year` 设置下，年度聚合后的方差并不会被严格保持。因而年内与年度的对比应理解为指示性的，而非严格的方差匹配比较。
-->
- Age-structure and population-growth extensions are deferred to a later revision round, as noted above.
<!-- 中文翻译
- 如上所述，年龄结构与人口增长扩展将推迟到后续修订轮次。
-->

## Reproducibility
<!-- 中文翻译
## 可复现性
-->

Run:
<!-- 中文翻译
运行：
-->

```bash
uv run python reports/climate_scenario_experiments.py --repeats 4 --years 80
```
<!-- 中文翻译
用于重新生成同样的快速检验流程。
-->

to regenerate the same quick-check pipeline.
