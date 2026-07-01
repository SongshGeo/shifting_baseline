# Shifting Baseline（基线偏移）

本仓库对应论文 *“Archival and palaeoenvironmental documentation of historical extreme
events reveals perceptual bias in collective memory”* 的研究代码。

它将中国历史气候档案（1470–1900 CE）与树轮水文气候重建、器测数据（1901–2000 CE）
进行对比，用以研究**基线偏移综合征（Shifting Baseline Syndrome, SBS）**，并用
**智能体模型（ABM）**复现观测到的感知偏差模式。

!!! tip "语言"
    使用顶部的语言选择器在 **English** 与 **中文** 之间切换。

## 项目做什么

- **两条旱涝指数（WDI）序列。** 受感知影响的 **H-WDI**（历史档案）与相对独立的
  **N-WDI**（自然树轮代用指标）对比，二者均分为五个序数等级（SD、MD、N、MW、SW）。
- **误配（mismatch）分析。** 在二者不一致处，档案表现出系统性的*移位比较*偏差——
  当年被相对于近期经验、而非绝对基线来判断。
- **滑动窗口再标准化。** 在约 20–40 年（最优 ≈ 30 年）窗口内对自然序列再标准化可
  提升一致性——这是核心发现。
- **智能体模型。** 一个极简 ABM 用来区分两种 SBS 机制（**H1 代际遗忘** 对
  **H2 集体错觉**）；只有代际遗忘能复现实证的代际最优。稳健性由 AR(1) 气候强迫与
  全局（Sobol）敏感性分析确立。

## 复现

我们共享分析真正使用的**两条派生序列**——历史档案 **H-WDI**（序数等级）与树轮 **N-WDI**
（z-score）——以便你复现核心结果并接入自己的数据。原始语料与数据生产步骤并不需要。
从 **[数据与复现](guide/data.md)** 开始。

## 分析模块

| 模块 | 作用 |
| --- | --- |
| [`filters`](api/filters.md) | 滑动窗口再标准化与序数分级 |
| [`compare`](api/compare.md) | 相关性计算与窗口扫描（约 30 年最优） |
| [`calibration`](api/calibration.md) | `MismatchReport` 混淆矩阵/误差模式分析 + 蒙特卡洛零分布 |
| [`abm`](api/abm.md) | `ClimateObservingModel` 及观察者智能体 |
| [`climate_forcing`](api/climate_forcing.md) | i.i.d. / AR(1) / 趋势 气候生成器 |
| [`sensitivity`](api/sensitivity.md) | 基于方差的 Sobol 敏感性分析 |
| [`constants`](api/constants.md)、[`utils`](api/utils.md) | 年份边界、映射、绘图、日志 |

## 快速上手

```bash
uv sync                                   # 安装运行时 + 开发依赖
uv run python shifting_baseline/abm.py    # 运行 ABM（自包含，无需数据）
uv run pytest                             # 运行测试
```

请参阅 **[快速开始](getting-started/installation.md)** 完成安装、**[用户指南](guide/overview.md)**
了解科学背景，并从 **[数据与复现](guide/data.md)** 用共享序列运行分析。

## 引用

```bibtex
@software{shifting_baseline,
  title  = {Shifting Baseline: analysis of Shifting Baseline Syndrome in historical climate archives},
  author = {Song, Shuang},
  url    = {https://github.com/SongshGeo/shifting_baseline},
}
```

## 参与开发

这是一个持续开发中的研究代码库。如果你有兴趣参与——扩展 ABM、分析流水线，或为模型
搭建前端——欢迎联系：

- 📧 **song[at]gea.mpg.de**
- 🌐 [cv.songshgeo.com](https://cv.songshgeo.com/)
- 🐛 [GitHub issues](https://github.com/SongshGeo/shifting_baseline/issues)
