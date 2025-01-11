#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from itertools import product
from typing import Any, Callable, Dict, Generic, List, Sequence, TypeVar

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from past1000.api.io import Path, PathLike
from past1000.core.models import MODELS, MULTI_VARS, VARS, _EarthSystemModel

M = TypeVar("M", bound=_EarthSystemModel)


class ModelComparisonExperiment(Generic[M]):
    """地球系统模型对比实验类

    该类用于管理和比较多个地球系统模型的数据，支持批量处理和分析。

    Attributes:
        folder (Path): 数据根目录
        subdir_pattern (str): 子目录模式，支持 {freq} 和 {model} 占位符
        _models (Dict[MODELS, _EarthSystemModel]): 模型字典
        freq (str): 数据时间频率（如 'mon', 'day'）

    Args:
        folder (PathLike): 数据根目录路径
        subdir_pattern (str, optional): 子目录模式. 默认为 ""
        freq (str, optional): 数据时间频率. 默认为 "mon"

    Example:
        >>> exp = ModelComparisonExperiment("data/", "{freq}/{model}")
        >>> exp.add_model("CESM", "MPI")
        >>> results = exp.apply(lambda m: m.calc_spei())
    """

    def __init__(self, folder: PathLike, subdir_pattern: str = "", freq: str = "mon"):
        self.folder = Path(folder)
        # {freq}/{model}
        self.subdir_pattern = subdir_pattern
        self._models: Dict[MODELS, _EarthSystemModel] = {}
        self.freq = freq

    def __iter__(self):
        return iter(self.models)

    def __next__(self):
        return next(self.models)

    @property
    def models(self) -> List[_EarthSystemModel]:
        """模型列表"""
        return list(self._models.values())

    def get_data_path(self, model_name: MODELS, freq: str) -> Path:
        """获取特定模型的数据路径

        Args:
            model_name (MODELS): 模型名称
            freq (str): 数据频率

        Returns:
            Path: 模型数据的完整路径

        Note:
            路径基于 folder 和 subdir_pattern 构建，会替换模式中的占位符
        """
        return self.folder / self.subdir_pattern.replace("{freq}", freq).replace(
            "{model}", model_name
        )

    def create_model(self, name: MODELS) -> _EarthSystemModel:
        """创建新的模型实例

        Args:
            name (MODELS): 模型名称

        Returns:
            _EarthSystemModel: 新创建的模型实例

        Note:
            使用当前设置的频率(freq)创建模型
        """
        path = self.get_data_path(
            name,
            self.freq,
        )
        return _EarthSystemModel(name, path)

    def add_model(self, *models: MODELS | _EarthSystemModel) -> List[_EarthSystemModel]:
        """添加一个或多个模型到实验中

        Args:
            *models: 要添加的模型，可以是模型名称或模型实例

        Returns:
            List[_EarthSystemModel]: 成功添加的模型列表

        Note:
            - 如果提供模型名称，会自动创建模型实例
            - 重复添加同名模型会发出警告
        """
        added_models = []
        for model in models:
            if isinstance(model, _EarthSystemModel):
                pass
            else:
                model = self.create_model(model)
            if model.name in self._models:
                logger.warning(f"模型 {model.name} 已存在")
            logger.info(f"添加模型 {model.name}")
            self._models[model.name] = model
            added_models.append(model)
        return added_models

    def apply(
        self,
        func: Callable[[_EarthSystemModel], Any],
        **kwargs,
    ) -> Dict[MODELS, Any]:
        """对所有模型应用指定函数

        Args:
            func: 要应用的函数，接收模型实例作为第一个参数
            **kwargs: 传递给函数的额外参数

        Returns:
            Dict[MODELS, Any]: 以模型名称为键的结果字典

        Example:
            >>> def process_temp(model, threshold=0):
            ...     return model.process_data('tas', threshold=threshold)
            >>> results = exp.apply(process_temp, threshold=273.15)
        """
        results: Dict[MODELS, Any] = {}
        for model in tqdm(self.models):
            logger.info(f"应用函数 {func.__name__} 到模型 {model.name}, 参数包括: {kwargs}")
            results[model.name] = func(model, **kwargs)  # 对于普通函数，传入 model
        return results

    def check_variables(self, variables: VARS | MULTI_VARS) -> pd.DataFrame:
        """检查所有模型中变量的可用性

        Args:
            variables (VARS | MULTI_VARS): 要检查的变量名称或列表

        Returns:
            pd.DataFrame: 变量可用性矩阵，行为变量，列为模型

        Note:
            返回的DataFrame中，True表示变量可用，False表示不可用
        """
        check_vars_list = []
        for model in self.models:
            check_vars = model.check_variables(variables, raise_error=False)
            check_vars.name = model.name
            check_vars_list.append(check_vars)
        return pd.concat(check_vars_list, axis=1)

    def batch_apply(
        self,
        func: Callable[[_EarthSystemModel], Any],
        iter_vars: Dict[str, Sequence[Any]],
        value_name: str = "value",
        **kwargs,
    ) -> pd.DataFrame:
        """批量对模型应用函数，支持参数迭代

        Args:
            func: 要应用的函数
            iter_vars: 要迭代的参数字典，键为参数名，值为可能的参数值序列
            value_name (str, optional): 结果值的列名. 默认为 "value"
            **kwargs: 其他固定参数

        Returns:
            pd.DataFrame: 包含所有结果的数据框，包含模型名称和参数值列

        Raises:
            ValueError: 当函数返回值类型不支持时

        Example:
            >>> results = exp.batch_apply(
            ...     lambda m: m.calc_spei(),
            ...     iter_vars={'scale': [1, 3, 6, 12]},
            ...     value_name='spei'
            ... )
        """
        results: List[pd.DataFrame] = []
        for model, var_name in product(self.models, iter_vars):
            for var_value in iter_vars[var_name]:
                kwargs[var_name] = var_value
                res = func(model, **kwargs)
                if isinstance(res, (np.ndarray, list)):
                    res = pd.DataFrame(res, columns=[value_name])
                elif isinstance(res, (pd.Series, pd.DataFrame)):
                    res = pd.DataFrame(res)
                else:
                    raise ValueError(f"函数 {func.__name__} 返回值类型不支持.")
                res["model"] = model.name
                res[var_name] = var_value
                results.append(res)
        return pd.concat(results).reset_index(drop=True)
