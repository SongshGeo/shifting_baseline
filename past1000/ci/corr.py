#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
计算两个数据集之间的相关系数
"""
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from rasterio.enums import Resampling

from past1000.api.series import HistoricalRecords, classify_by_std
from past1000.core.models import _EarthSystemModel


def summer_precipitation(model: _EarthSystemModel):
    """获取夏季（6-8月）降水量

    Args:
        model: 模型对象

    Returns:
        xr.DataArray: 夏季降水量数据，每年都是 3 个月数据的和
    """
    pr = model["pr"]
    basin_mask = pr.isel(time=0).notnull()
    # 选择 6-8 月的数据
    return (
        pr.sel(time=pr.time.dt.month.isin([6, 7, 8]))
        .groupby("time.year")
        .sum(dim="time", skipna=True)
        .where(basin_mask)
        .reset_coords(names="time", drop=True)
        .rename({"year": "time"})
    )


def corr(x, y) -> np.ndarray:
    """计算两个数组之间的相关系数"""
    return np.corrcoef(x, y)[0, 1]


def reproject_match(
    da: xr.DataArray,
    target: xr.DataArray,
    resampling: str = "bilinear",
    dims_name: Tuple[str, str] = ("lat", "lon"),
) -> xr.DataArray:
    """重采样数据集以匹配目标数据集

    Args:
        da: 需要重采样的数据集，维度为 (time, lat, lon)
        target: 目标数据集，维度为 (time, lat, lon)
        resampling: 重采样方法
    """
    y_dim, x_dim = dims_name
    # 重命名维度并确保顺序正确
    return da.rio.reproject_match(
        target,
        resampling=Resampling[resampling],
    ).rename(
        {
            "y": y_dim,
            "x": x_dim,
        }
    )


def xarray_corr(
    da1: xr.DataArray,
    da2: xr.DataArray,
    resampling: str = "nearest",
    reproject_to: Optional[Literal["da1", "da2"]] = None,
    time_dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    """计算两个数据集之间的相关系数"""
    # 选择共同年份
    da1, da2 = xr.align(da1, da2, join="inner")
    # 重投影
    if reproject_to is not None:
        # pass
        if reproject_to == "da1":
            da2 = reproject_match(da2, da1, resampling)
        elif reproject_to == "da2":
            da1 = reproject_match(da1, da2, resampling)
        else:
            raise ValueError(f"Invalid reproject value: {reproject_to}")

    if da1.shape != da2.shape:
        raise ValueError("The two datasets have different shapes.")
    if da1[time_dim].size != da2[time_dim].size:
        raise ValueError("The two datasets have different time dimensions.")

    return xr.apply_ufunc(
        corr,
        da1,
        da2,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[]],  # 输出没有时间维度
        vectorize=True,
        **kwargs,
    )


def merge_comparable_data(
    *data: pd.DataFrame | pd.Series, index: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """合并可比较的数据"""
    if index is None:
        index = np.arange(850, 2000)
    df = pd.DataFrame(index=index)
    for d in data:
        df = pd.merge(
            df,
            right=d,
            how="left",
            left_index=True,
            right_index=True,
        )
    return df


def plot_corr(df: pd.DataFrame) -> None:
    """绘制相关系数"""
    dataplot = sns.heatmap(
        df.corr(numeric_only=True),
        cmap="YlGnBu",
        annot=True,
    )
    dataplot.set_title("Correlation Coefficient")
    return dataplot


def process_exp_data(
    datasets: Dict[str, xr.DataArray],
    var_name: str,
) -> List[pd.DataFrame]:
    """处理CMIP数据，转化为标准差等级"""
    cmip_summer_pr = []
    for model, da in datasets.items():
        tmp_df = da.mean(dim=["lat", "lon"]).to_dataframe()
        tmp_df = tmp_df.drop("spatial_ref", axis=1).rename({var_name: model}, axis=1)
        ser = classify_by_std(tmp_df)
        cmip_summer_pr.append(ser)
    return cmip_summer_pr


def calc_corr(
    data: xr.DataArray,
    models: List[_EarthSystemModel],
    vars: List[str],
) -> pd.DataFrame:
    """计算相关系数"""
    pass


def process_historical_data(
    shp_path: str,
    data_path: str,
    how: Literal["mean", "mode"] = "mean",
) -> pd.DataFrame:
    """处理历史数据"""
    records = HistoricalRecords(shp_path=shp_path, data_path=data_path)
    # TODO 这里需要处理缺失值？这里目前画图的时候不显示
    if how == "mean":
        data = np.mean(records, axis=1)
    elif how == "mode":
        data = records.mode(axis=1)[0]
    else:
        raise ValueError(f"Invalid how value: {how}")
    data.name = f"historical_{how}"
    return data


def process_nc_data(
    path: str,
) -> pd.DataFrame:
    """处理nc数据"""
    # TODO 这里变成除了 time 之外的维度
    data = xr.open_dataarray(path).mean(dim=["lat", "lon"])
    return data.to_dataframe()


def process_csv_data(
    path: str,
) -> pd.DataFrame:
    """处理csv数据"""
    return pd.read_csv(path, index_col=0)


def process_recon_data(
    path: str,
    dtype: Literal["nc", "csv"] = "nc",
    var_names_map: Dict[str, str] | None = None,
    to_level: bool = True,
) -> pd.DataFrame:
    """处理重建数据"""
    if var_names_map is None:
        var_names_map = {}
    if dtype == "nc":
        data = process_nc_data(path)
    elif dtype == "csv":
        data = process_csv_data(path)
    else:
        raise ValueError(f"Invalid dtype value: {dtype}")
    data = data.rename(var_names_map, axis=1)
    if to_level:
        return classify_by_std(data)
    return data
