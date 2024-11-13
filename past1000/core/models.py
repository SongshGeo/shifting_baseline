#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, TypeAlias

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
from past1000.ci.spei import calc_single_spei
from past1000.utils.units import (
    MONTH_DAYS,
    YEAR_DAYS,
    TimeUnit,
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
    """地球系统模式类"""

    def __init__(
        self,
        model_name: MODELS,
        data_path: PathLike,
        out_path: PathLike = "output",
        **kwargs,
    ):
        self._name: MODELS = model_name
        self._dir: Path = check_data_dir(data_path)
        self._files: Dict[str, List[str]] = {}
        self.attrs: Dict[str, Any] = kwargs
        self._merged_data: Dict[VARS, XarrayData] = {}
        self._out_dir: Path = check_data_dir(
            self.folder / out_path,
            create=True,
        )

    @property
    def name(self) -> MODELS:
        """模型名称"""
        return self._name

    @property
    def folder(self) -> Path:
        """数据目录"""
        return self._dir

    @property
    def out_dir(self) -> Path:
        """输出目录"""
        return self._out_dir

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

    def _clip_data(
        self, xda: XarrayData, clip_by: Optional[str | gpd.GeoDataFrame] = None
    ) -> XarrayData:
        """裁剪数据"""
        if isinstance(clip_by, str):
            clip_by = gpd.read_file(clip_by)
        if clip_by is not None:
            return clip_data(xda, clip_by)
        logger.warning("未提供裁剪数据，返回原始数据。")
        return xda

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
        save: bool = False,
        cache: bool = True,
    ) -> XarrayData:
        if variable in self._merged_data:
            return self._merged_data[variable]
        # 合并数据
        xda = self._merge_data(variable=variable)
        # 裁剪数据
        xda = self._clip_data(xda, clip_by)
        # 转换单位
        if unit_to:
            xda = self._convert_units(xda, variable, unit_to)
        if save:
            write_nc(xda, self.out_dir, variable, self.name)
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
        save: bool = False,
        cache: bool = True,
    ) -> Optional[XarrayData]:
        """处理数据"""
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
                save,
                cache,
            )
        for v in variable:
            self.process_data(v, clip_by, unit_to, save, cache)
        return None

    def plot_series(self, variable: VARS | MULTI_VARS, **kwargs):
        """绘制时间序列图"""
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

    def calc_spei(self, pet_freq: TimeUnit = "month") -> xr.DataArray:
        """计算 SPEI"""
        return xr.apply_ufunc(
            calc_single_spei,
            self.calc_pet(input_freq=pet_freq, output_freq="month")
            .resample(time="ME")
            .mean(),
            self.get_variables("pr").resample(time="ME").mean(),
            input_core_dims=[["time"], ["time"]],  # 输入数组的核心维度
            output_core_dims=[["time"]],  # 输出数组的核心维度
            vectorize=True,  # 自动向量化
            dask="parallelized",  # 并行计算
            output_dtypes=[float],  # 输出数据类型
        )

    def calc_pet(
        self,
        input_freq: TimeUnit = "month",
        output_freq: TimeUnit = "month",
    ) -> xr.DataArray:
        """计算PET"""
        days = {
            "day": 1,
            "month": MONTH_DAYS,
            "year": YEAR_DAYS,
        }
        tasmin = self.get_variables("tasmin").pint.dequantify(format="unit")
        tasmax = self.get_variables("tasmax").pint.dequantify(format="unit")
        tas = self.get_variables("tas").pint.dequantify(format="unit")
        pet = (
            xci.potential_evapotranspiration(
                tasmin=tasmin,
                tasmax=tasmax,
                tas=tas,
                method="HG85",
            )
            * days[input_freq]
        )
        return flux_kg_to_mm(pet, flux_frequency=output_freq)
