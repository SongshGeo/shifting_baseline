# 架构图 (UML)

本页用 **PlantUML** 描述 `shifting_baseline` 的架构，重点是基于
[`abses`](https://github.com/ABSESpy/ABSES) 框架的多主体模型（ABM）。它也是**为 ABM 搭建
前端 / 部署模型**的参考蓝图（类结构、参数面、运行时序一目了然）。

共四张图，组织为「总览 → ABM 核心」：

1. [组件 / 包总览图](#1-component-overview) — 分析模块与其依赖
2. [ABM 类图](#2-abm-class-diagram) — 模型与观察者主体的类结构（前端最重要的参考）
3. [Observer 生命周期状态图](#3-observer-lifecycle) — 主体的「出生 → 记录 → 死亡」状态机
4. [仿真运行时序图](#4-simulation-sequence) — 一个 tick 与收尾阶段的调用链

---

## 1. 组件 / 包总览图 {#1-component-overview}

分析消费两条[共享序列](../guide/data.md)，运行一条短链：再标准化与分级 → 相关 → 校准 →
用 ABM 验证。原始数据读取、空间聚合、重建整合属于*数据生产*，此处有意省略。

```plantuml
@startuml component_overview_zh
title shifting_baseline — Analysis & Model Components

skinparam componentStyle rectangle
skinparam shadowing false
skinparam defaultTextAlignment center
skinparam packageStyle rectangle

component "Shared data\nH-WDI levels + N-WDI z-score" as SharedData

package "shifting_baseline" {
  component "filters\nclassify / calc_std_deviation" as Filters
  component "compare\nexperiment_corr_2d / sweep_*" as Compare
  component "calibration\nMismatchReport (+ MC null)" as Calib
  component "**abm**\nClimateObservingModel / ClimateObserver" as ABM
  component "climate_forcing\ngenerate / sigma_tick_*" as Forcing
  component "sensitivity\nrun_sobol" as SA
  component "constants" as Const
  package "utils" {
    component "calc" as UCalc
    component "log" as ULog
    component "plot" as UPlot
    component "types" as UTypes
  }
}

cloud "External libs" {
  component "Hydra / OmegaConf" as Hydra
  component "abses" as Abses
  component "SALib" as SALib
  component "pandas / numpy" as Pandas
}

SharedData ..> Filters
Filters ..> Const
Compare ..> Filters
Compare ..> UCalc
Compare ..> UPlot
Calib ..> Filters
Calib ..> UPlot

ABM ..> Forcing
ABM ..> Filters
ABM ..> Calib
ABM ..> Compare
ABM ..> Const
ABM ..> Abses
ABM ..> UCalc
ABM ..> Hydra

SA ..> ABM : Hydra multirun\n(subprocess overrides)
SA ..> SALib

Compare ..> Pandas

note bottom of SharedData
  Raw archives, spatial aggregation and
  reconstruction integration are data
  production — outside this diagram.
end note

@enduml
```

---

## 2. ABM 类图 {#2-abm-class-diagram}

ABM 由两个类构成：`ClimateObservingModel`（继承 `abses.MainModel`）与 `ClimateObserver`
（继承 `abses.Actor`）。模型持有离散气候序列 `_climate` 与逐 tick 档案 `_archive`，每个 tick
生成新的观察者；观察者相对**基线**感知气候 z-score 并按「负偏好」概率记录极端事件。这是
**前端要暴露参数、读取结果**的核心结构。

```plantuml
@startuml abm_class_diagram_zh
title ABM — Class Diagram (abm.py + climate_forcing.py)

skinparam shadowing false
skinparam classAttributeIconSize 0
hide empty members

class MainModel <<abses>> {
  + agents
  + time
  + p : config params
  + running : bool
  + run_id
  + outpath
}
class Actor <<abses>> {
  + model
  + age() : int
  + die()
}

class ClimateObservingModel {
  -- climate / time --
  - _climate : np.ndarray
  - _climate_process : str
  - _climate_sigma_year : float
  - _climate_sigma_tick : float
  - _climate_phi : float
  - _climate_trend : float
  - _step_per_year : int
  - _subannual_aggregation : SubannualAggregation
  -- agents / lifecycle --
  - _max_age_years / _max_age_ticks : int
  - _min_age_years / _min_age_ticks : int
  - _new_agents : int
  - spin_up_years / spin_up_ticks : int
  - _years / _ticks : int
  -- archive / caches --
  - _archive : dict[int, list[int]]
  - _collective_cache : Optional[pd.Series]
  - _collective_baseline_cache : tuple[float,float]
  - _frozen_year_means : dict[int, float]
  --
  + step() : None
  + end() : None
  + archive_it(extreme: int) : None
  + get_corr_curve(window_length=100, min_window=2, corr_method="kendall") : pd.DataFrame
  + climate_now : float <<property>>
  + climate_series : pd.Series <<cached>>
  + collective_memory_climate : pd.Series <<property>>
  + collective_baseline_stats : tuple <<property>>
  + model_baseline_stats : tuple <<cached>>
  + climate_df : pd.DataFrame <<property>>
  + mismatch_report : MismatchReport <<property>>
  - _generate_climate_series(n_ticks) : np.ndarray
  - _aggregate_to_yearly(series) : pd.Series
}

class ClimateObserver {
  - _memory : deque  '' maxlen = max_age
  - _max_age : int
  - _min_age : int
  --
  + step() : None
  + perceive(climate: float) : float  '' z-score vs baseline
  + write_down(z_score, scale=1, f0=0.1) : bool  '' negativity bias
  + memory : np.ndarray <<property>>
}

class MismatchReport <<calibration>> {
  + pred / true : pd.Series
  + analyze_error_patterns() : pd.DataFrame
  + get_statistics_summary() : str | dict
  + generate_report_figure() : plt.Figure
}

class "climate_forcing" as Forcing <<module>> {
  + generate(process, n_ticks, *, sigma, phi, trend_per_year) : np.ndarray
  + sigma_tick_from_sigma_year(*, sigma_year, step_per_year, subannual_aggregation) : float
}

class "filters" as Filters <<module>> {
  + classify_single_value(value) : int
  + classify(series, handle_na) : pd.Series
}

enum SubannualAggregation {
  mean
  sum
  last
}
enum ClimateProcess {
  iid
  ar1
  trend_plus_noise
  ar1_trend
}
enum MemoryBaseline {
  personal
  collective
  collective_lifetime
  model
}
enum CorrFunc {
  pearson
  kendall
  spearman
}

MainModel <|-- ClimateObservingModel
Actor <|-- ClimateObserver

ClimateObservingModel "1" *-- "0..*" ClimateObserver : agents.new()\nspawns each tick
ClimateObserver --> ClimateObservingModel : self.model

ClimateObservingModel ..> MismatchReport : mismatch_report
ClimateObservingModel ..> Forcing : _generate_climate_series()
ClimateObservingModel ..> Filters : classify()
ClimateObserver ..> Filters : classify_single_value()

ClimateObservingModel ..> ClimateProcess
ClimateObservingModel ..> SubannualAggregation
ClimateObservingModel ..> CorrFunc
ClimateObserver ..> MemoryBaseline

note right of ClimateObserver
  perceive() 的四种基线：
  - personal: 自身 memory 的 mean/std
  - collective: 当前 tick 群体记忆 mean/std
  - collective_lifetime: 集体档案，但限于
    主体自身寿命窗口
  - model: 全程气候序列 mean/std (恒定, 非感知)
  NaN 兜底: baseline→0, std→1
end note

note bottom of Filters
  constants:
  LEVELS = [-2,-1,0,1,2]
  THRESHOLDS = [-1.17,-0.33,0.33,1.17]
  MAX_AGE = 40 (years)
end note

@enduml
```

---

## 3. Observer 生命周期状态图 {#3-observer-lifecycle}

每个 `ClimateObserver` 在每个 tick 都把当前气候压入个人记忆；年龄达到 `min_age` 后才开始按
概率记录事件，超过 `max_age` 即死亡。这一年龄结构正是「基线偏移」机制的载体——年轻主体与
年长主体对同一极端事件的感知不同。

```plantuml
@startuml observer_lifecycle_zh
title ClimateObserver — Lifecycle State Machine

skinparam shadowing false

[*] --> Youth : agents.new()\nage = 0

state Youth {
  Youth : age() < min_age
  Youth : append climate -> _memory
  Youth : **records nothing**
}

state Prime {
  Prime : min_age <= age() <= max_age
  Prime : append climate -> _memory
  Prime : z = perceive(climate)
  Prime : if write_down(z):\n  classify_single_value(z) -> level\n  model.archive_it(level)
}

state Dead {
  Dead : age() > max_age
  Dead : self.die()
}

Youth --> Prime : age() >= min_age
Prime --> Prime : each tick
Prime --> Dead : age() > max_age
Dead --> [*]

note right of Prime
  write_down 采用"负偏好"概率：
  prob = norm.sf(|z|, scale)
  record  <=  rand() < f0 + 0.5 - prob
  极端越强 → 记录概率越高
end note

note bottom of Dead
  archive_it 还会按 loss_rate 概率丢弃
  记录（实验默认 0.4）
end note

@enduml
```

---

## 4. 仿真运行时序图 {#4-simulation-sequence}

下图展示一次实验的调用链：初始化生成气候序列 → 每个 tick 生成主体并令其 `step` → 收尾计算
相关性曲线并输出。`abses.Experiment` 负责按 `repeats / num_process` 批量并行运行。

```plantuml
@startuml simulation_sequence_zh
title ABM — Simulation Run Sequence

skinparam shadowing false
skinparam sequenceMessageAlign center

actor "Hydra\n(repeat_run)" as Hydra
participant "Experiment\n(abses)" as Exp
participant "ClimateObservingModel" as Model
participant "agents\n(abses)" as Agents
participant "ClimateObserver" as Obs
participant "climate_forcing" as Forcing
participant "filters" as Filters

== Initialization ==
Hydra -> Exp : Experiment.new(ClimateObservingModel, cfg)
Exp -> Model : __init__()
Model -> Forcing : sigma_tick_from_sigma_year(...)
Model -> Forcing : generate(process, n_ticks, sigma=...)
Forcing --> Model : _climate : np.ndarray
Model -> Model : _archive = {tick: [] ...}

== Per tick (xN) ==
Exp -> Model : step()
Model -> Agents : new(ClimateObserver, new_agents,\nmax_age_ticks, min_age_ticks)
Model -> Agents : do("step")
loop each observer
  Agents -> Obs : step()
  Obs -> Model : climate_now
  Obs -> Obs : _memory.append(climate)
  alt age() < min_age
    Obs -->> Agents : return (too young)
  else age() >= min_age
    Obs -> Obs : perceive(climate) -> z
    note right of Obs
      baseline depends on
      p.memory_baseline
      (personal / collective /
       collective_lifetime / model)
    end note
    Obs -> Obs : write_down(z) ?
    alt recorded
      Obs -> Filters : classify_single_value(z) -> level
      Obs -> Model : archive_it(level)
      Model -> Model : if rand() >= loss_rate:\n  _archive[tick].append(level)
    end
    opt age() > max_age
      Obs -> Obs : die()
    end
  end
end

== Finalize ==
Exp -> Model : end()
Model -> Model : climate_df\n(aggregate + drop spin-up)
Model -> Model : get_corr_curve()\n-> compare_corr_2d(...)
Model -> Model : write correlations.csv

@enduml
```
