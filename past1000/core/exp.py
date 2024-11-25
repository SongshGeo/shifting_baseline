#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Any, Callable, Dict, Generic, List, TypeVar

import pandas as pd
from loguru import logger
from tqdm import tqdm

from past1000.api.io import Path, PathLike

from .models import MODELS, MULTI_VARS, VARS, _EarthSystemModel

M = TypeVar("M", bound=_EarthSystemModel)


class ModelComparisonExperiment(Generic[M]):
    """模型对比实验"""

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
        """获取数据路径

        Args:
            model_name: 模型名称
            freq: 频率
        """
        return self.folder / self.subdir_pattern.replace("{freq}", freq).replace(
            "{model}", model_name
        )

    def create_model(self, name: MODELS) -> _EarthSystemModel:
        """创建模型"""
        path = self.get_data_path(
            name,
            self.freq,
        )
        return _EarthSystemModel(name, path)

    def add_model(self, *models: MODELS | _EarthSystemModel) -> List[_EarthSystemModel]:
        """添加模型"""
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
        self, func: Callable[[_EarthSystemModel], Any] | str, **kwargs
    ) -> Dict[MODELS, Any]:
        """应用函数"""
        results: Dict[MODELS, Any] = {}
        for model in tqdm(self.models):
            if isinstance(func, str):
                method = getattr(model, func)
                logger.info(f"应用方法 {func} 到模型 {model.name}, 参数包括: {kwargs}")
                results[model.name] = method(**kwargs)  # 不需要传入 model 参数
            else:
                logger.info(f"应用函数 {func.__name__} 到模型 {model.name}, 参数包括: {kwargs}")
                results[model.name] = func(model, **kwargs)  # 对于普通函数，传入 model
        return results

    def check_variables(self, variables: VARS | MULTI_VARS) -> pd.DataFrame:
        """检查变量是否存在"""
        check_vars_list = []
        for model in self.models:
            check_vars = model.check_variables(variables, raise_error=False)
            check_vars.name = model.name
            check_vars_list.append(check_vars)
        return pd.concat(check_vars_list, axis=1)
