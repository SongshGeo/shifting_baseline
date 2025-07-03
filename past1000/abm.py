#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from collections import deque

import numpy as np
import pandas as pd
from abses import Actor, Experiment, MainModel
from hydra import main
from omegaconf import DictConfig
from scipy.stats import mode, norm

from past1000.compare import compare_corr_2d
from past1000.filters import calc_std_deviation, classify
from past1000.utils.config import get_output_dir

MAX_AGE = 40
THRESHOLDS = [-1.17, -0.33, 0.33, 1.17]


class Model(MainModel):
    """模型
    在世界上，有一定数量的气候观察者，他们根据自己的感知记录极端气候事件。
    我们观察他们的集体记录，并与实际的气候极端情况进行对比。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建一个气候序列，表现为正态分布，有些小概率的极端事件
        years = self.p.get("years", 100)
        self._climate = np.random.normal(0, 1, years)
        # 一个记录极端气候事件的列表，可以理解为气候观察者的集体记忆
        self._archive: dict[int, list[int]] = {i: [] for i in range(years)}
        self._years = years
        self._max_age = self.p.get("max_age", 40)
        self._new_agents = self.p.get("new_agents", 10)
        self._mode_cache = None
        self._mode_cache_tick = -1  # 记录缓存是在哪个时间点生成的
        # 新增一个缓存，用于存储每个级别最后一次出现的年份
        self._last_event_years = {}
        self.final_corr = None

    @property
    def climate(self) -> float:
        """Current climate"""
        # 返回当前时刻的气候值
        return self._climate[self.time.tick]

    @property
    def climate_series(self) -> pd.Series:
        """Climate series"""
        return pd.Series(self._climate, index=range(self._years))

    @property
    def historical_mean(self) -> pd.Series:
        """Historical mean of climate"""
        return pd.Series({k: np.mean(v) for k, v in self._archive.items()})

    @property
    def classified(self) -> pd.Series:
        """Classified climate"""
        # 对气候序列进行分类，返回一个理论分类结果的 Series
        return classify(self._climate)

    @property
    def estimation(self) -> pd.DataFrame:
        """Estimation of climate"""
        value = self.climate_series
        expect = self.mode
        classified = self.classified
        return pd.DataFrame(
            {
                "value": value,
                "expect": expect,
                "classified": classified,
            }
        )

    @property
    def mode(self) -> pd.Series:
        """Mode of climate. Results are cached per tick."""
        # 如果当前 tick 的 mode 已经计算过，直接返回缓存
        if self._mode_cache_tick == self.time.tick:
            return self._mode_cache

        # 否则，重新计算
        current_archive = {k: v for k, v in self._archive.items() if v}  # 只计算非空列表
        result = pd.Series({k: mode(v)[0] for k, v in current_archive.items()})

        # 更新缓存和时间戳
        self._mode_cache = result
        self._mode_cache_tick = self.time.tick
        return result

    def write_down(self, extreme: int) -> None:
        """记录极端气候事件"""
        loss_rate = self.p.get("loss_rate", 0.2)
        if np.random.random() < loss_rate:
            return
        self._archive[self.time.tick].append(extreme)
        # 当数据写入时，清除缓存，确保下次调用 mode 时会重新计算
        # 注意：因为我们的缓存是按 tick 记录的，所以这里其实可以不清除
        # 但如果逻辑变复杂，主动清除缓存是好习惯
        # self._mode_cache_tick = -1

    def step(self) -> None:
        # 在所有 Agent 行动之前，先更新一次全局信息
        self.update_last_event_years()

        if self.time.tick == self._years - 1:
            self.running = False
        self.agents.new(ClimateObserver, self._new_agents, max_age=self._max_age)
        self.agents.do("step")

    def update_last_event_years(self):
        """
        （在每个时间步开始时调用）
        更新并缓存每个气候级别最后一次被记录的年份。
        """
        current_mode = self.mode
        if current_mode.empty:
            return
        # 找到当前所有出现过的级别
        unique_levels = current_mode.unique()
        for level in unique_levels:
            # 找到该 level 最后一次出现的 index (年份)
            last_year = current_mode[current_mode == level].last_valid_index()
            # 更新到缓存中
            self._last_event_years[level] = last_year

    def get_last_event_year_for_level(self, level: int) -> int | None:
        """Agent 可以调用的接口，直接从缓存获取结果"""
        return self._last_event_years.get(level, None)

    def end(self) -> None:
        """保存估计结果"""
        min_period = self.p.get("min_period", 2)
        filter_side = self.p.get("filter_side", "right")
        corr_method = self.p.get("corr_method", "kendall")
        self.estimation.to_csv(self.outpath / f"estimation_{self.run_id}.csv")
        windows = np.arange(2, 100)
        min_periods = np.repeat(min_period, 98)
        rs, _, _ = compare_corr_2d(
            self.historical_mean,
            self.climate_series,
            windows=windows,
            min_periods=min_periods,
            filter_func=calc_std_deviation,
            corr_method=corr_method,
            filter_side=filter_side,
        )
        self.final_corr = pd.Series(
            data=rs,
            index=windows,
            name=f"model_{self.run_id}",
        )


class ClimateObserver(Actor):
    """气候观察者

    气候观察者会对极端气候事件有所观察、感知、记录。
    由于经验有限，他们只能基于自己的记忆，对气候事件进行粗略的分类（如：极端干旱、极端湿润等）。
    由于记异略常的特性，他们更倾向于记录极端气候事件，而不是正常气候年份。
    """

    def __init__(self, *args, max_age=MAX_AGE, **kwargs):
        super().__init__(*args, **kwargs)
        # 一个记录极端气候事件的列表，可以理解为气候观察者的个人记忆
        self._memory = deque(maxlen=max_age)
        # 记录观察者年龄，只有观察者年龄大于 10 时，才有可能记录极端气候事件
        self.age: int = 1
        self._max_age = max_age
        self._min_age = self.p.get("min_age", 10)

    @property
    def memory(self) -> np.ndarray:
        """Memory of extreme climate events"""
        # 返回一个数组，记录了观察者记忆中的极端气候事件
        return np.array(self._memory)

    @property
    def classified(self) -> pd.Series:
        """Classified climate"""
        # 对观察者记忆中的极端气候事件进行分类，返回一个理论分类结果的 Series
        return classify(self.memory)

    def write_down(self, event: float) -> bool:
        """记录极端气候事件"""
        # 计算事件的绝对值，并使用正态分布的生存函数（1-CDF）来计算事件发生的概率
        prob = norm.sf(abs(event))
        # 如果随机数小于 1 - 事件发生的概率，则记录该事件
        return np.random.random() < 1 - prob

    def perception(self, climate: float) -> int:
        """感知气候变化的 Z-score"""
        z = (climate - self.memory.mean()) / self.memory.std()
        return z

    def judge_extreme(self, z: float) -> int:
        """判断是否为极端气候事件"""
        if z > THRESHOLDS[3]:
            return 2
        if z > THRESHOLDS[2]:
            return 1
        if z < THRESHOLDS[0]:
            return -1
        if z < THRESHOLDS[1]:
            return -2
        return 0

    def rejudge_based_on_mode(self, init_judgment: int) -> int:
        """
        基于群体的集体记忆 (mode)，重新评估自己对气候事件的判断。

        逻辑：
        1. 观察者做出一个基于个人记忆的初步判断 (init_judgment)。
        2. 他去"查阅史书"（model.mode），寻找之前的与他初步判断同级别的最后一次记录。
        3. 如果找到了，他比较当前的气候与那时的气候。
        4. 如果当前气候比那时更"极端"（更冷或更热），他有一定概率
           会加剧或减弱自己的判断。

        Parameters:
        -----------
        init_judgment : int
            观察者基于个人记忆做出的初步判断。

        Returns:
        --------
        int
            经过集体记忆修正后的最终判断。
        """
        # 直接向 Model 查询结果，无需自己计算
        last_event_year = self.model.get_last_event_year_for_level(init_judgment)

        # 如果历史上没有同级别的记载，我只能相信自己的初步判断
        if last_event_year is None or last_event_year >= self.model.time.tick:
            # 增加一个判断，确保不会引用到未来的数据（虽然不太可能发生）
            return init_judgment

        # 获取那时和现在的气候数据
        climate_then = self.model.climate_series.loc[last_event_year]
        climate_now = self.model.climate  # 当前气候

        # 计算差异
        diff = climate_now - climate_then

        # 差异越大，越有可能调整判断。我们用 sigmoid 函数来平滑这个概率
        # sigmoid(0) = 0.5, diff 越大，概率越接近1
        prob_to_adjust = 1 / (1 + np.exp(-abs(diff)))

        # 如果随机数表明不需要调整，则坚持原判
        if self.random.random() > prob_to_adjust:
            return init_judgment

        # 根据差异的方向进行调整
        if diff > 0.1:
            return min(init_judgment + 1, 2)
        if diff < -0.1:
            return max(init_judgment - 1, -2)
        return init_judgment

    def step(self):
        """更新状态"""
        self.age += 1
        climate = self.model.climate
        self._memory.append(climate)
        if self.age < self._min_age:
            return
        z = self.perception(climate)
        # 如果观察者准备记录极端气候事件，则在全局记录该事件
        if self.write_down(z):
            extreme = self.judge_extreme(z)
            extreme = self.rejudge_based_on_mode(extreme)
            self.model.write_down(extreme)
        # 如果观察者年龄大于最大年龄，则死亡
        if self.age > self._max_age:
            self.die()


@main(config_path="../config", config_name="config", version_base=None)
def repeat_run(cfg: DictConfig | None = None):
    """重复运行模型"""
    assert cfg is not None, "cfg is None"
    path = get_output_dir()
    final_corr_datasets = []
    for i in range(cfg.how.repeat):
        model = Model(parameters={"model": cfg.how}, run_id=i, outpath=path)
        model.run_model()
        final_corr_datasets.append(model.final_corr)
    pd.concat(final_corr_datasets, axis=1).to_csv(path / "correlations.csv")


if __name__ == "__main__":
    repeat_run()
