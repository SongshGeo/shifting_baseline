#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import re
from pathlib import Path
from typing import List, Optional, TypeAlias

import click
import xarray as xr
from loguru import logger

from past1000.api.log import setup_logger

PathLike: TypeAlias = str | Path
XarrayData: TypeAlias = xr.Dataset | xr.DataArray


def check_data_dir(
    path: Optional[PathLike] = None,
    create: bool = False,
) -> Path:
    """检查数据目录，如果不存在则创建。

    Args:
        path: 要检查的目录。
        create: 如果目录不存在，是否创建。

    Returns:
        检查的目录。
    """
    # 如果 path 为 None，则使用当前工作目录
    if path is None:
        path = Path.cwd()
    path = Path(path)
    # 检查路径是否为目录
    if not path.is_dir():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            msg = f"创建目录: {path}"
            logger.info(msg)
        else:
            msg = f"目录不存在: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
    logger.info(f"目录 {path} 检查完毕。")
    return path


def get_matching_files(
    directory: PathLike,
    pattern: str,
) -> List[str]:
    """
    获取指定目录下所有符合正则表达式的文件名列表。

    Args:
        directory: 要搜索的目录路径。
        pattern: 正则表达式模式。

    Returns:
        符合正则表达式的文件名列表。
    """
    directory = Path(directory)
    logger.info(f"搜索目录: {directory}")
    logger.debug(f"正则表达式: {pattern}")
    regex = re.compile(pattern)
    matching_files = [
        file.name
        for file in directory.iterdir()
        if file.is_file() and regex.match(file.name)
    ]
    logger.info(f"找到 {len(matching_files)} 个文件。")
    return matching_files


def create_cmip_regex(
    variable: Optional[str] = None,
    model: Optional[str] = None,
    frequency: Optional[str] = None,
    experiment: Optional[str] = None,
    ensemble: Optional[str] = None,
) -> str:
    """创建用于匹配CMIP文件名的正则表达式。

    Args:
        variable: 变量名。
        model: 模型名。
        frequency: 频率。
        experiment: 实验名。
        ensemble: 集合名。

    Returns:
        用于匹配CMIP文件名的正则表达式。
    """
    logger.debug(f"变量名: {variable}")
    logger.debug(f"模型名: {model}")
    logger.debug(f"频率: {frequency}")
    logger.debug(f"实验名: {experiment}")
    logger.debug(f"集合名: {ensemble}")
    pattern = [
        r"(?P<variable>{})".format(variable or r"[\w]+"),
        r"(?P<frequency>{})".format(frequency or r"[\w]+"),
        r"(?P<model>{})".format(re.escape(model) if model else r"[\w-]+"),
        r"(?P<experiment>{})".format(experiment or r"[\w]+"),
        r"(?P<ensemble>{})".format(ensemble or r"[\w\d]+"),
        r"[\d]{4,}-[\d]{4,}",
    ]
    return r"_".join(pattern) + r"\.nc$"


def write_geo_attrs(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """写入地理属性"""
    dataset.lat.attrs["units"] = "degree"
    dataset.lon.attrs["units"] = "degree"
    logger.debug(f"写入地理属性: {dataset}")
    return dataset


def read_nc(
    path: PathLike,
    verbose: bool = False,
    variable: Optional[str] = None,
    **kwargs,
) -> XarrayData:
    """读取nc文件"""
    logger.debug(f"读取nc文件: {path}")
    dataset = xr.open_dataset(path, **kwargs)
    if variable:
        dataset = dataset[variable]
    dataset = write_geo_attrs(dataset)
    if verbose:
        logger.debug(f"Info: {dataset.info()}")
    return dataset


def write_nc(
    data: XarrayData,
    path: PathLike,
    variable: str,
    model: str,
    frequency: str = "mon",
    experiment: str = "past1000",
    ensemble: str = "r1i1p1f1",
    time_range: Optional[str] = None,
    encoding: Optional[dict] = None,
    **kwargs,
) -> None:
    """写入nc文件，保持CMIP6命名格式

    Args:
        data: 要写入的数据（Dataset 或 DataArray）
        path: 写入目录
        variable: 变量名（如 'pet', 'pr' 等）
        model: 模型名
        frequency: 时间频率，默认'mon'
        experiment: 实验名，默认'past1000'
        ensemble: 集合名，默认'r1i1p1f1'
        time_range: 时间范围，如'0850-1850'。如果为None则自动从数据中提取
        encoding: 编码设置
        **kwargs: 传递给to_netcdf的其他参数
    """
    path = Path(path)

    # 确保目录存在
    path.mkdir(parents=True, exist_ok=True)

    # 如果没有提供时间范围，从数据中提取
    if time_range is None:
        start_year = int(data.time.dt.year.min().values)
        end_year = int(data.time.dt.year.max().values)
        time_range = f"{start_year:04d}-{end_year:04d}"

    # 构建CMIP格式的文件名
    filename = f"{variable}_{frequency}_{model}_{experiment}_{ensemble}_{time_range}.nc"
    filepath = path / filename

    # 如果是DataArray，转换为Dataset
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=variable)

    # 设置默认编码
    if encoding is None:
        encoding = {var: {"zlib": True, "complevel": 4} for var in data.variables}

    # 写入文件
    logger.debug(f"写入nc文件: {filepath}")
    data.to_netcdf(filepath, encoding=encoding, **kwargs)
    logger.info(f"文件已保存: {filepath}")


def search_cmip_files(
    path: PathLike,
    model: Optional[str] = None,
    variable: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """搜索CMIP文件"""
    regex = create_cmip_regex(model=model, variable=variable, **kwargs)
    files = get_matching_files(path, regex)
    if not files:
        logger.warning("未找到匹配的文件。")
    elif len(files) > 1:
        logger.info(f"找到 {len(files)} 个匹配的文件。")
    return files


@click.command()
@click.argument("path", type=click.Path(exists=False), default=None)
@click.option("--model", "-m", type=str, help="模型名")
@click.option("--variable", "-v", type=str, help="变量名")
@click.option("--frequency", "-f", type=str, help="频率")
@click.option("--experiment", "-e", type=str, help="实验名")
@click.option("--ensemble", "-n", type=str, help="集合名")
@click.option("--verbose", is_flag=True, help="是否显示详细信息")
def cli(
    path: PathLike = Path.cwd(),
    model: Optional[str] = None,
    variable: Optional[str] = None,
    frequency: Optional[str] = None,
    experiment: Optional[str] = None,
    ensemble: Optional[str] = None,
    verbose: bool = False,
):
    """搜索CMIP文件。

    PATH: 要搜索的目录路径
    """
    params = {
        "模型": model,
        "变量": variable,
        "频率": frequency,
        "实验": experiment,
        "集合": ensemble,
    }
    for name, value in params.items():
        if value:
            logger.info(f"搜索{name}: {value}")
    files = search_cmip_files(
        path,
        model=model,
        variable=variable,
        frequency=frequency,
        experiment=experiment,
        ensemble=ensemble,
    )
    if verbose:
        for file in files:
            logger.info(file)


if __name__ == "__main__":
    setup_logger(std_level="INFO")
    cli()
