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
import xarray as xr
import yaml  # type: ignore
from loguru import logger
from matplotlib import pyplot as plt

from past1000.api.io import (
    PathLike,
    XarrayData,
    check_data_dir,
    read_nc,
    search_cmip_files,
)
from past1000.ci.clip import clip_data
from past1000.ci.spei import calc_single_spei
from past1000.utils.units import convert_cmip_units
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
]
MULTI_VARS: TypeAlias = Sequence[VARS]

# 读取字典
_VARS = resources.files("config") / "variables.yaml"
with open(Path(str(_VARS)), "r", encoding="utf-8") as f:
    VARS_ATTRS = yaml.safe_load(f)


class _EarthSystemModel:
    """地球系统模式类"""

    def __init__(
        self,
        model_name: str,
        data_path: PathLike,
        **kwargs,
    ):
        self._name: str = model_name
        self._dir: Path = check_data_dir(data_path)
        self._files: Dict[str, List[str]] = {}
        self.attrs: Dict[str, Any] = kwargs
        self._merged_data: Dict[VARS, XarrayData] = {}

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    @property
    def dir(self) -> Path:
        """数据目录"""
        return self._dir

    def get_variables(
        self, variable: VARS, create: bool = False, **kwargs
    ) -> XarrayData:
        """获取变量数据"""
        if variable in self._merged_data:
            return self._merged_data[variable]
        if create:
            return self.process_data(variable, **kwargs)
        error_msg = f"{self.name} 模型还没有处理变量 {variable} 数据，请使用 `process_data` 方法处理数据。"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def search_variable(self, variable: VARS, **kwargs) -> List[str]:
        """搜索变量"""
        if variable not in self._files:
            logger.info(f"{self.name}模型首次搜索 {variable} 变量")
            files = search_cmip_files(
                self.dir, model=self.name, variable=variable, **kwargs
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
                self.dir / files[0], variable=variable, use_cftime=True, **kwargs
            )
        logger.info(f"{self.name} 模型需合并 {len(files)} 个文件。")
        xda = [
            read_nc(self.dir / f, variable=variable, use_cftime=True, **kwargs)
            for f in files
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
        convert_units: Optional[str] = None,
    ) -> XarrayData:
        """转换单位"""
        if convert_units is None:
            convert_units = VARS_ATTRS[variable]["output_units"]
        xda = convert_cmip_units(xda, variable, convert_units)
        return xda

    def _process_single_variable(
        self,
        variable: VARS,
        clip_by: Optional[str | gpd.GeoDataFrame] = None,
        convert_units: Optional[str] = None,
    ) -> XarrayData:
        if variable in self._merged_data:
            return self._merged_data[variable]
        # 合并数据
        xda = self._merge_data(variable=variable)
        # 裁剪数据
        xda = self._clip_data(xda, clip_by)
        # 转换单位
        xda = self._convert_units(xda, variable, convert_units)
        self._merged_data[variable] = xda
        return xda

    def process_data(
        self,
        variable: VARS | MULTI_VARS,
        clip_by: Optional[str | gpd.GeoDataFrame] = None,
        unit_to: Optional[str] = None,
    ) -> Optional[XarrayData]:
        """处理数据"""
        # 如果变量是列表，则递归处理每个变量
        if isinstance(variable, str):
            return self._process_single_variable(variable, clip_by, unit_to)
        for v in variable:
            self._process_single_variable(v, clip_by, unit_to)
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

    def compute_spei(self, **kwargs) -> xr.DataArray:
        """计算 SPEI"""
        return xr.apply_ufunc(
            calc_single_spei,
            self.get_variables("hfls").resample(time="ME").mean(),
            self.get_variables("pr").resample(time="ME").mean(),
            input_core_dims=[["time"], ["time"]],  # 输入数组的核心维度
            output_core_dims=[["time"]],  # 输出数组的核心维度
            vectorize=True,  # 自动向量化
            dask="parallelized",  # 并行计算
            output_dtypes=[float],  # 输出数据类型
        )


class MRIESM20(_EarthSystemModel):
    """MRI-ESM2-0 模型"""

    def __init__(self, path: PathLike, **kwargs):
        super().__init__("MRI-ESM2-0", path, **kwargs)


class MIROCES2L(_EarthSystemModel):
    """MIROC-ES2L 模型"""

    def __init__(self, path: PathLike, **kwargs):
        super().__init__("MIROC-ES2L", path, **kwargs)


class ACCESSESM15(_EarthSystemModel):
    """ACCESS-ESM1-5 模型"""

    def __init__(self, path: PathLike, **kwargs):
        super().__init__("ACCESS-ESM1-5", path, **kwargs)
