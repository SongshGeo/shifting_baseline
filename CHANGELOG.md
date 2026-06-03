# Changelog

## 自底向上代码审查（2026-06，`743560b`..`HEAD`）

对全仓活代码做了一轮自底向上（叶子模块 → 编排入口）的逐模块手动检查：每站先人工核对逻辑，
再补/修测试。共 **49 个文件改动，+4744 / −2599**；测试从 ~287 增至 **377 passed**（同时删除了
死代码自带的测试）。所有改动按下方依赖层推进，每层依赖的下层都已先行确认。

### 🐛 Bug 修复

- **compare.py — off-by-2**：头条"最优窗口"图的窗口尺寸偏移 2，导致 `sweep_max_corr_year`
  报告的峰值窗口系统性错位。已修，并补 `sweep_slices` / `sweep_max_corr_year` 回归测试。
- **filters.py — ddof 不一致**：核心 SBS 操作 `calc_std_deviation`（滑窗再标准化）在不同路径上
  混用 `ddof=0/1`。统一后机制本体的标准化口径自洽。
- **utils/plot.py — headline 配色错位**：personal/collective 头条图的颜色映射错配。
- **reports/_bug2a_compare.py — 复现脚本被重构打断**：calibration 改造给
  `_run_significance_test` 增加了 `random_seed` 形参并由 `analyze_error_patterns` 转发，
  导致 buggy 复刻（缺该形参）一跑即 `TypeError`。已为 override 补上 `random_seed` 并忠实忽略
  （pre-fix 行为本就走全局 `np.random`，由 `_run_one` 统一 seed）。

### ♻️ 重构 / 死代码清理

- **utils/anova.py（−671）+ tests/test_anova.py**：删除。单因素 ANOVA 已被 Sobol 全局敏感性
  分析取代，`comprehensive_anova_analysis` 仅剩自身测试引用。
- **utils/email.py（−124）、utils/config.py（−48 净）**：删除未使用的封装/死路径。
- **data.py（−343 净）**：删除探索性死代码（probability_weighted / bayesian 等未用分支，经确认"没用过，可归档"）。
- **utils/plot.py（−285 净）**：剥离与绘图分离的死数据变换函数。
- **calibration.py**：蒙特卡洛显著性检验改用 `np.random.default_rng(random_seed)`（可复现、
  不污染全局随机态）；`get_mean_diff` 改为所有非 NaN 单元的整体均值（每单元等权）。
- **constants.py / utils/types.py**：底层常量精简；最底层文件加 `_` 前缀语义标注"不应改动"；
  统一显著性星号约定为 `*** < 0.01 / ** < 0.05 / * < 0.1`。
- **类型导入归一**：`data.py` 的 `Region`、`calibration.py` 的 `PathLike` 等统一从 `geo_dskit` 引用。
- **shifting_baseline/__main__.py**：移除未被调用的 `batch_process_recon_data` 死 import 及其
  `__all__` 导出。
- **SPEI 孤儿清理**（源自已删除的 `shifting_baseline/ci/spei`）：删除
  `tests/utils/generate_spei_testdata.py`（−127，import 即崩）、`conftest.py` 的
  `expected_spei_dict`/`pr`/`pet` fixture（−32，无任何测试引用）及 `tests/data/{input,expected}/*.nc`。

### ✅ 测试新增 / 增强

| 新增/改名文件 | 覆盖 |
|------|------|
| `tests/test_constants.py` | 阈值、等级↔类别对称映射、STAGE/END 边界自洽性 |
| `tests/test_log.py`（原 `test_logging.py`） | logger 初始化、Hydra 集成、重复初始化去重 |
| `tests/test_calc_misc.py` | `calc_corr` / `low_pass_filter` / NaN / 空序列分支 |
| `tests/test_filters.py`（增强） | 极端窗口、全 NaN 边界 |
| `tests/test_plot_logic.py` | 纯数据准备函数 + 绘图 smoke |
| `tests/test_compare_sweep.py` | `sweep_slices` 窗口生成、`sweep_max_corr_year` 多切片（含 off-by-2 回归） |
| `tests/test_calibration_mc.py` | MC 可复现性、不污染全局态、`get_mean_diff` 整体均值 |
| `tests/test_abm_mechanism.py` | `perceive` z-score（钉死审稿人 R2 的 `(climate−baseline)/std` 优先级修复）、`write_down` 负性偏差 |
| `tests/test_abm_model.py` | `__init__` 契约校验、`_aggregate_to_yearly` 三种聚合、`archive_it` loss_rate、生命周期 min_age 门控、personal 基线整体跑通 |
| `tests/test_reports_scripts.py` | `run_sensitivity.parse_args` 默认值契约、`summarize_scenario` 峰值提取、`plot_sobol` 标注辅助 |

### 🔎 审查发现（未改动，留作记录）

- **`cfg.how` 静默无效**：`__main__._main` 硬编码 Step 1–6 流水线，从不读 `cfg.how`（仅
  `process.py` 读 `cfg.how.recon`）。故 `how=correlation` / `how=compare` 这类 override 被 Hydra
  接受但不影响主入口实际执行路径。与 `CLAUDE.md` 记录的行为一致，属潜在不一致而非 bug。
- **半死代码保留**：`process.py`（树轮 recon 预处理，独立 `@main` 从不被主流程调用）与
  `mc.py`（pymc 重建合并，仅 `recalculate_data=true` 时由 `data.py` 条件导入）经评估为
  **committed `${ds.processed}/*.csv` 缓存的生产者**——是可复现性资产，决定保留且不补重型测试。
- **data.py:115 TODO**：`standardize_both` 其实接受 `uncertainties` 参数（None→10 年滑窗自动算）；
  现状未传 recon 自带不确定性列，是方法选择而非 bug。

### 🧰 配套（同期）

- 新增 ABM 气候场景鲁棒性脚本 `scripts/climate_scenarios.py` + `scripts/climate_scenarios.slurm`
  （多气候强迫 iid/ar1/trend × 时间分辨率 step_per_year 扫描）。
- `reports/abm.ipynb`：移除旧的描述性视图，全面改为 Sobol 全局敏感性分析；驱动散点图重做为
  左 `max_age` / 右 `climate_phi` 双面板，颜色区分 personal vs collective 基线。
- 文档：新增 `docs/architecture/uml.md` 架构图、`notes/climate_forcing_review.md` 评审笔记。
