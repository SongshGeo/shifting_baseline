#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
        self._merged_data: Dict[str, XarrayData] = {}

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    @property
    def dir(self) -> Path:
        """数据目录"""
        return self._dir

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

    def merge_data(self, variable: VARS, **kwargs) -> XarrayData:
        """合并数据

        Args:
            variable: 变量
            **kwargs: 传递给 `read_nc` 的其他参数

        Returns:
            合并后的数据
        """
        files = self.search_variable(variable)
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

    @lru_cache
    def merge_and_clip(
        self,
        variable: VARS,
        shp: Optional[str | gpd.GeoDataFrame] = None,
    ) -> XarrayData:
        """合并并裁剪数据"""
        xda = self.merge_data(variable)
        if isinstance(shp, str):
            shp = gpd.read_file(shp)
        if shp is not None:
            xda = clip_data(xda, shp)
        return xda

    def plot_series(self, variable: List[VARS], **kwargs):
        """绘制时间序列图"""
        len_var = len(variable)
        fig, axes = plt.subplots(
            nrows=len_var,
            ncols=1,
            figsize=(12, 3 * len_var),
            tight_layout=True,
        )
        for ax, var in zip(axes, variable):
            attrs = VARS_ATTRS[var]
            xda = self.merge_and_clip(variable=var)
            converted = convert_cmip_units(xda, var, attrs["output_units"])
            plot_single_time_series(converted, ax=ax, attrs=attrs, **kwargs)
        fig.suptitle(self.name)
        plt.show()


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
