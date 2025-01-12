#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from functools import partial
from importlib import resources
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias

import geopandas as gpd
import pandas as pd
import xarray as xr
import xclim.indices as xci
import yaml  # type: ignore
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from past1000.api.io import (
    PathLike,
    XarrayData,
    check_data_dir,
    read_nc,
    search_cmip_files,
    write_nc,
)
from past1000.ci.clip import clip_data
from past1000.ci.spei import DistributionType, calc_single_spei, spei_to_level
from past1000.utils.units import (
    MONTH_DAYS,
    YEAR_DAYS,
    TimeUnit,
    _get_time_resolution,
    convert_cmip_units,
    flux_kg_to_mm,
)
from past1000.viz.plot import plot_single_time_series

VARS: TypeAlias = Literal[
    "pr",
    "tas",
    "hfls",
    "rsd",
    "rld",
    "ps",
    "sfcWind",
    "mrso",
    "mrro",
    "tasmin",
    "tasmax",
    "hurs",
]
MULTI_VARS: TypeAlias = Sequence[VARS]
MODELS: TypeAlias = Literal[
    "MRI-ESM2-0",
    "MIROC-ES2L",
    "ACCESS-ESM1-5",
]

# 读取字典
_VARS = resources.files("config") / "variables.yaml"
with open(Path(str(_VARS)), "r", encoding="utf-8") as f:
    VARS_ATTRS = yaml.safe_load(f)


class _EarthSystemModel:
    """地球系统模式类

    该类用于处理和分析地球系统模型的数据，包括数据读取、处理、单位转换和可视化等功能。

    Attributes:
        _name (MODELS): 模型名称
        _dir (Path): 数据存储目录
        _files (Dict[str, List[str]]): 变量文件映射字典
        _merged_data (Dict[VARS, XarrayData]): 已处理的数据缓存

    Args:
        model_name (MODELS): 模型名称，必须是预定义的模型之一
        data_path (PathLike): 数据存储路径
    """

    callable_attrs = ["process_data", "save_data", "calc_spei"]

    def __init__(
        self,
        model_name: MODELS,
        data_path: PathLike,
    ):
        self._name: MODELS = model_name
        self._dir: Path = check_data_dir(data_path)
        self._files: Dict[str, List[str]] = {}
        self._merged_data: Dict[VARS, XarrayData] = {}

    @property
    def name(self) -> MODELS:
        """模型名称"""
        return self._name

    @property
    def folder(self) -> Path:
        """数据目录"""
        return self._dir

    @property
    def time_resolutions(self) -> Dict[VARS, TimeUnit]:
        """获取变量时间分辨率"""
        res: Dict[VARS, TimeUnit] = {}
        for k, da in self._merged_data.items():
            res[k] = _get_time_resolution(da)
        return res

    def get_variables(
        self,
        variable: VARS,
        create: bool = False,
        **kwargs,
    ) -> XarrayData:
        """获取变量数据"""
        if variable in self._merged_data:
            return self._merged_data[variable]
        if create:
            return self.process_data(variable, **kwargs)
        error_msg = (
            f"{self.name} 模型没有缓存的变量 {variable}，请使用 `process_data` 方法处理数据，"
            "并设置 `cache=True`。"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def search_variable(
        self,
        variable: VARS,
        **kwargs,
    ) -> List[str]:
        """搜索变量"""
        if variable not in self._files:
            logger.info(f"{self.name}模型首次搜索 {variable} 变量")
            files = search_cmip_files(
                self.folder,
                model=self.name,
                variable=variable,
                **kwargs,
            )
            self._files[variable] = files
            return files
        return self._files[variable]

    def _merge_data(self, variable: VARS, **kwargs) -> XarrayData:
        """合并数据

        Args:
            variable: 变量
            **kwargs: 传递给 `read_nc` 的其他参数

        Returns:
            合并后的数据
        """
        files = self.search_variable(variable)
        if len(files) == 0:
            error_msg = f"{self.name} 模型没有找到 {variable} 变量。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(files) == 1:
            return read_nc(
                self.folder / files[0], variable=variable, use_cftime=True, **kwargs
            )
        logger.info(f"{self.name} 模型需合并 {len(files)} 个文件。")
        xda = [
            read_nc(self.folder / f, variable=variable, use_cftime=True, **kwargs)
            for f in tqdm(files, desc=f"Reading {variable} files")
        ]
        return xr.concat(xda, dim="time").sortby("time")

    def _convert_units(
        self,
        xda: XarrayData,
        variable: VARS,
        unit_to: str | bool = True,
    ) -> XarrayData:
        """转换单位"""
        if unit_to is False:
            return xda
        if unit_to is True:
            unit = VARS_ATTRS[variable]["output_units"]
        else:
            unit = unit_to
        return convert_cmip_units(xda, variable, unit)

    def _process_single_variable(
        self,
        variable: VARS,
        clip_by: Optional[str | gpd.GeoDataFrame] = None,
        unit_to: str | bool = True,
        cache: bool = True,
    ) -> XarrayData:
        if variable in self._merged_data:
            return self._merged_data[variable]
        # 合并数据
        xda = self._merge_data(variable=variable)
        # 裁剪数据
        if clip_by is not None:
            xda = clip_data(xda, clip_by)
        # 转换单位
        if unit_to:
            xda = self._convert_units(xda, variable, unit_to)
        if cache:
            self._merged_data[variable] = xda
        return xda

    def check_variables(
        self,
        variables: VARS | MULTI_VARS,
        raise_error: bool = True,
    ) -> pd.Series:
        """检查变量是否存在"""
        if isinstance(variables, str):
            variables = [variables]
        result = pd.Series(index=variables, dtype=bool)
        for v in variables:
            files = self.search_variable(v)
            if len(files) == 0:
                if raise_error:
                    msg = f"{self.name} 模型没有找到 {v} 变量。"
                    logger.error(msg)
                    raise ValueError(msg)
                result[v] = False
            else:
                result[v] = True
        return result

    def process_data(
        self,
        variable: VARS | MULTI_VARS,
        clip_by: Optional[str | gpd.GeoDataFrame] = None,
        unit_to: str | Dict[VARS, str | bool] | bool = True,
        cache: bool = True,
    ) -> Optional[XarrayData]:
        """处理模型变量数据

        该方法用于处理单个或多个变量的数据，包括数据合并、裁剪和单位转换等操作。

        Args:
            variable: 需要处理的变量名称或变量列表
            clip_by: 用于裁剪数据的地理范围，可以是GeoJSON文件路径或GeoDataFrame对象
            unit_to: 单位转换设置
                - 当为str时，指定转换的目标单位
                - 当为dict时，为每个变量指定转换单位
                - 当为True时，使用默认的输出单位
                - 当为False时，不进行单位转换
            cache: 是否缓存处理后的数据

        Returns:
            处理后的数据数组，如果处理多个变量则返回None

        Raises:
            ValueError: 当请求的变量不存在时
        """
        check_vars = self.check_variables(variable, raise_error=False)
        if not check_vars.all():
            msg = (
                f"{self.name} 模型缺少变量: {check_vars[check_vars is False].index.tolist()}"
            )
            logger.warning(msg)
        # 如果变量是列表，则递归处理每个变量
        if isinstance(variable, str):
            if isinstance(unit_to, dict):
                unit = unit_to[variable]
            else:
                unit = unit_to
            return self._process_single_variable(
                variable,
                clip_by,
                unit,
                cache,
            )
        for v in variable:
            self.process_data(v, clip_by, unit_to, cache)
        return None

    def plot_series(self, variable: VARS | MULTI_VARS, **kwargs) -> None:
        """绘制变量的时间序列图

        Args:
            variable: 要绘制的变量名称或变量列表
            **kwargs: 传递给plot_single_time_series的其他参数

        Note:
            - 单个变量时绘制单张图
            - 多个变量时绘制多子图
            - 图表标题会使用模型名称
        """
        if isinstance(variable, str):
            variable = [variable]
        len_var = len(variable)
        fig, axes = plt.subplots(
            nrows=len_var,
            ncols=1,
            figsize=(12, 3 * len_var),
            tight_layout=True,
        )
        # 如果只有一个变量，则将 axes 转换为列表
        if len_var == 1:
            axes = [axes]
        # 绘制每个变量
        for ax, var in zip(axes, variable):
            attrs = VARS_ATTRS[var]
            data = self.get_variables(var)
            plot_single_time_series(data, ax=ax, attrs=attrs, **kwargs)
        fig.suptitle(self.name)
        plt.show()

    def calc_spei(
        self,
        to_level: bool = True,
        scale: int = 1,
        distribution: DistributionType = "pearson",
        years: Tuple[int, int, int] = (850, 850, 1850),
    ) -> xr.DataArray:
        """计算标准化降水蒸发指数 (SPEI)

        Args:
            to_level: 是否将SPEI值转换为干旱等级
            scale: SPEI计算的时间尺度（月）
            distribution: 概率分布类型，可选 "pearson"、"gamma" 等
            years: 用于拟合分布的年份范围，格式为(起始年,校准起始年,校准结束年)

        Returns:
            包含SPEI值的数据数组

        Note:
            计算需要降水量(pr)和最高最低温度(tasmax, tasmin)数据
            如果to_level=True，返回的是干旱等级而不是SPEI值
        """
        pr = self.get_variables("pr").resample(time="ME").mean()
        # 计算 PET
        pet = (
            self.calc_pet(
                input_freq=self.time_resolutions["pr"],
                output_freq="month",
            )
            .resample(time="ME")
            .mean()
        )
        # 对齐时间，删除不匹配的时间
        pr, pet = xr.align(pr, pet, join="inner")
        # 计算 SPEI
        spei = xr.apply_ufunc(
            partial(
                calc_single_spei,
                scale=scale,
                distribution=distribution,
                years=years,
            ),
            pr,
            pet,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        if to_level:
            return spei_to_level(spei)
        return spei

    def calc_pet(
        self,
        input_freq: TimeUnit = "month",
        output_freq: TimeUnit = "month",
    ) -> xr.DataArray:
        """计算潜在蒸发蒸腾量 (PET)

        使用Hargreaves方法计算潜在蒸发蒸腾量。

        Args:
            input_freq: 输入数据的时间频率 ('day', 'month', 'year')
            output_freq: 输出数据的时间频率 ('day', 'month', 'year')

        Returns:
            包含PET值的数据数组，单位为毫米

        Note:
            计算基于最高温度(tasmax)和最低温度(tasmin)
            结果会根据指定的时间频率进行单位转换
        """
        days = {
            "day": 1,
            "month": MONTH_DAYS,
            "year": YEAR_DAYS,
        }
        tasmin = self.get_variables("tasmin").pint.dequantify(format="unit")
        tasmax = self.get_variables("tasmax").pint.dequantify(format="unit")
        pet = (
            xci.potential_evapotranspiration(
                tasmin=tasmin,
                tasmax=tasmax,
                method="HG85",
            )
            * days[input_freq]
        )
        return flux_kg_to_mm(pet, flux_frequency=output_freq)

    def save_data(self, path: PathLike) -> None:
        """保存数据

        将模型中所有缓存的数据保存到指定路径。
        命名规则为：{model_name}_{variable}_{frequency}_{time_range}.nc

        Args:
            path: 保存路径
        """
        path = check_data_dir(path, create=True)
        for var, data in self._merged_data.items():
            data = data.pint.dequantify()
            write_nc(
                data=data,
                path=path,
                variable=var,
                model=self.name,
                frequency=self.time_resolutions[var],
            )
