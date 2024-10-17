#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import geopandas as gpd
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
from past1000.utils.units import convert_cmip_units
from past1000.viz.plot import plot_single_time_series

VARS = Literal["pr", "tas"]

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
        shp: Optional[PathLike] = None,
        **kwargs,
    ):
        self._name: str = model_name
        self._dir: Path = check_data_dir(data_path)
        self._files: Dict[str, List[str]] = {}
        self.attrs: Dict[str, Any] = kwargs
        if shp:
            self.shp: gpd.GeoDataFrame = gpd.read_file(shp)

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    @property
    def dir(self) -> Path:
        """数据目录"""
        return self._dir

    def read_file(self, file_name: str, variable: VARS, **kwargs) -> XarrayData:
        """读取文件"""
        return read_nc(self.dir / file_name, variable=variable, **kwargs)

    def clip_data(
        self,
        data: XarrayData,
        shp: Optional[str | gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> XarrayData:
        """裁剪数据"""
        if shp is None:
            shp = self.shp
        if isinstance(shp, str):
            shp = gpd.read_file(shp)
        return clip_data(data, shp, **kwargs)

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


class MRIESM20(_EarthSystemModel):
    """MRI-ESM2-0 模型"""

    def __init__(self, path: PathLike, **kwargs):
        super().__init__("MRI-ESM2-0", path, **kwargs)

    def plot_series(self, variable: List[VARS], **kwargs):
        """绘制时间序列图"""
        len_var = len(variable)
        _, axes = plt.subplots(
            nrows=len_var,
            ncols=1,
            figsize=(12, 3 * len_var),
            tight_layout=True,
        )
        for ax, var in zip(axes, variable):
            files = self.search_variable(var)
            attrs = VARS_ATTRS[var]
            # TODO 目前只支持单个文件
            if len(files) > 1:
                raise ValueError(f"{self.name} 模型有多个 {var} 变量文件")
            xda = self.read_file(files[0], variable=var, use_cftime=True)
            clipped = self.clip_data(xda, self.shp)
            converted = convert_cmip_units(clipped, var, attrs["output_units"])
            plot_single_time_series(converted, ax=ax, attrs=attrs, **kwargs)
        plt.show()
