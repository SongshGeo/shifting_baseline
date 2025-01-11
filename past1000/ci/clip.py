#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import geopandas as gpd
import rioxarray
import xarray as xr

from past1000.api.io import write_geo_attrs


def clip_data(
    data: xr.DataArray,
    shp: gpd.GeoDataFrame,
    crs: str = "epsg:4326",
    **kwargs,
) -> xr.DataArray:
    """
    裁剪空间数据.

    Args:
        data (xr.DataArray): 数据
        shp (gpd.GeoDataFrame): 矢量数据
        crs (str, optional): 目标坐标系. Defaults to "epsg:4326".
        **kwargs: 其他参数

    Returns:
        xr.DataArray: 裁剪后的数据
    """
    data.rio.write_crs(crs, inplace=True)
    xda = data.rio.clip(
        shp.geometry.values,
        crs=data.rio.crs,
        **kwargs,
    )
    return write_geo_attrs(xda)


if __name__ == "__main__":
    rioxarray.show_versions()
