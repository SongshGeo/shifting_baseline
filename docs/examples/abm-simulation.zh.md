# ABM 模拟

概念（2×2 基线、气候强迫、Sobol 敏感性）见 **[ABM 指南](../guide/abm.md)**；本页演示如何*运行*。

## 命令行（推荐）

ABM 是一个 Hydra 应用，通过 `abses.Experiment` 支持 `--multirun` 扫描。

```bash
# 主分析：personal 基线（H1）下的 AR(1) 强迫
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# 判别机制：扫描四种基线
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model

# 寿命扫描（Sobol 主导参数）
uv run python shifting_baseline/abm.py --multirun model.max_age=20,30,40,50,60
```

产物写入时间戳 `multirun/` 目录；每次运行都会写出其 窗口–相关 向量。

## Python

```python
from hydra import compose, initialize
from shifting_baseline.abm import ClimateObservingModel

with initialize(version_base=None, config_path="../config"):
    cfg = compose(config_name="config.yaml",
                  overrides=["model=test", "+model.climate_process=ar1"])

model = ClimateObservingModel(parameters=cfg)
model.run_model()

# 集体档案与真实气候之间的 窗口–τ 曲线
curve = model.get_corr_curve(corr_method=cfg.corr_method)
print(curve.head())

# 底层序列
df = model.climate_df          # 客观气候 + 集体记忆气候
```

!!! note
    构造器/方法的精确签名会演进——请查看生成的 [`abm` API](../api/abm.md)。在 `personal`
    基线下，窗口–τ 曲线在 20–40 年带出现峰值；在 `collective` 下单调上升。

## 敏感性分析

基于方差的 Sobol 分析位于 [`sensitivity`](../api/sensitivity.md)，通常作为批处理作业运行
（对五个参数做 Saltelli 采样、多次重复）。大规模扫描见该模块 API 及项目的 `slurm` 脚本。
