#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""处理任何历史重建数据的管道
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import xarray as xr
from loguru import logger
from omegaconf import DictConfig
from pandas import DataFrame, Series, read_csv
from xarray import DataArray, open_dataarray

if TYPE_CHECKING:
    from geo_dskit.utils.types import PathLike


__all__ = [
    "open_dataarray",
    "read_csv",
    "convert_time_axis",
]


def convert_time_axis(ds: xr.Dataset, begin_year: int = 1470) -> xr.Dataset:
    """转换时间轴为实际年份，并保留时间属性信息

    Args:
        ds: 包含时间轴的 xarray Dataset
    Returns:
        更新了时间轴的 Dataset
    """
    # 获取时间单位和参考时间
    actual_years = begin_year + ds.time.values.astype(int)

    # 更新数据集的时间坐标
    ds = ds.assign_coords(time=actual_years)
    # 保留时间属性信息
    ds.time.attrs["units"] = "year"
    ds.time.attrs["original_units"] = f"years since {begin_year}-1-1"
    return ds


class ProcessRecon:
    """处理历史重建数据"""

    # 处理函数列表
    process_list: List[str] = __all__

    def __init__(
        self, name: str, path: str, processes: Dict[str, Dict[str, Any]]
    ) -> None:
        self.name: str = name
        self._process_list: Dict[str, Callable] = {}
        self._args: Dict[str, Dict[str, Any]] = {}
        for process, kwargs in processes.items():
            self._add_process(process, **kwargs)
        self.data_path: Path = Path(path)

    def __str__(self):
        return f"<{self.name}: {self.data_path}>"

    def __repr__(self):
        return str(self)

    @property
    def data_path(self) -> Path:
        """数据路径"""
        path = Path(self._path)
        return path

    @data_path.setter
    def data_path(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        self._path = path

    @property
    def pipeline(self):
        """处理函数列表"""
        return self._process_list.keys()

    def _add_process(self, process: str, **kwargs):
        """添加处理函数

        Args:
            process: 处理函数名
            **kwargs: 处理函数的参数

        Raises:
            ValueError: 如果提供了未知参数
        """
        if process not in self.process_list:
            raise KeyError(f"未知函数: {process}，请检查处理函数列表 {self.process_list}")
        process_func = globals()[process]
        # 获取函数的参数信息
        sig = inspect.signature(process_func)
        # 检查提供的参数是否都在函数参数列表中
        unknown_params = [param for param in kwargs if param not in sig.parameters]
        if unknown_params:
            raise ValueError(f"函数 {process} 收到未知参数: {', '.join(unknown_params)}")

        self._process_list[process] = process_func
        self._args[process] = kwargs

    def process(self) -> Any:
        """处理数据"""
        data = self.data_path
        for name in self.pipeline:
            func = self._process_list[name]
            kwargs = self._args[name]
            data = func(data, **kwargs)
        return data

    @staticmethod
    def export(data, path: PathLike) -> None:
        """导出数据"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, DataArray):
            data.drop_vars("spatial_ref", errors="ignore").to_netcdf(path)
        elif isinstance(data, (DataFrame, Series)):
            data.to_csv(path, index=False)
        else:
            raise ValueError(f"未知数据类型: {type(data)}")


def batch_process_recon_data(cfg: DictConfig):
    """批量处理历史重建数据"""
    for name, values in cfg.how.recon.items():
        logger.info(f"Processing {name}...")
        # 获取配置
        path = values["path"]
        out_path = values["out"]
        processes = values["process"]
        # 创建处理对象
        recon = ProcessRecon(name, path, processes)
        # 处理数据
        data = recon.process()
        # 导出数据
        recon.export(data, path=out_path)
        logger.success(f"Processed {name} successfully!")


if __name__ == "__main__":
    pass
