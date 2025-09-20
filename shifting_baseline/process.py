#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""处理任何历史重建数据的管道
"""

from __future__ import annotations

import gc
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import pandas as pd
import psutil
import xarray as xr
from hydra import main
from omegaconf import DictConfig
from pandas import DataFrame, Series, read_csv
from xarray import DataArray, open_dataarray

if TYPE_CHECKING:
    from geo_dskit.core.types import PathLike


from shifting_baseline.utils.log import get_logger

log = get_logger(__name__)

__all__ = [
    "open_dataarray",
    "read_csv",
    "convert_time_axis",
]


def log_memory_usage(stage: str = ""):
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    log.info("Memory usage %s: %.1f MB", stage, memory_mb)


def force_garbage_collection():
    """Force garbage collection to free memory"""
    collected = gc.collect()
    log.debug("Garbage collection freed %d objects", collected)


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
        log.info("Processing %s...", name)
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
        log.info("Processed %s successfully!", name)


def load_and_combine_datasets_chunked(extract_dir, chunk_size=5):
    """Load netCDF files in chunks to reduce memory usage"""
    extract_path = Path(extract_dir)
    nc_files = sorted(extract_path.glob("pre_*.nc"))

    if not nc_files:
        raise FileNotFoundError("No netCDF files found in extract directory")

    log.info("Found %d netCDF files to process", len(nc_files))
    log.info("Processing in chunks of %d files to optimize memory usage", chunk_size)

    # Process files in chunks
    all_datasets = []

    for i in range(0, len(nc_files), chunk_size):
        chunk_files = nc_files[i : i + chunk_size]
        log.info(
            "Processing chunk %d/%d (%d files)...",
            i // chunk_size + 1,
            (len(nc_files) + chunk_size - 1) // chunk_size,
            len(chunk_files),
        )

        chunk_datasets = []
        for nc_file in chunk_files:
            log.info("Loading %s...", nc_file.name)
            try:
                # Use chunks parameter to enable dask arrays for lazy loading
                ds = xr.open_dataset(nc_file, chunks={"time": 100})
                chunk_datasets.append(ds)
            except Exception as e:
                log.error(f"Error loading {nc_file.name}: {e}")
                continue

        if chunk_datasets:
            log.info("Combining chunk datasets...")
            chunk_combined = xr.concat(chunk_datasets, dim="time")
            all_datasets.append(chunk_combined)

            # Clear chunk datasets from memory
            del chunk_datasets
            del chunk_combined

    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")

    log.info("Combining all chunks along time dimension...")
    combined_ds = xr.concat(all_datasets, dim="time")

    # Sort by time
    combined_ds = combined_ds.sortby("time")

    return combined_ds


def load_and_combine_datasets_lazy(extract_dir):
    """Load netCDF files using lazy loading with dask for memory efficiency"""
    extract_path = Path(extract_dir)
    nc_files = sorted(extract_path.glob("pre_*.nc"))

    if not nc_files:
        raise FileNotFoundError("No netCDF files found in extract directory")

    log.info("Found %d netCDF files to process", len(nc_files))
    log.info("Using lazy loading with dask for memory efficiency")

    # Use xarray's open_mfdataset for efficient lazy loading
    try:
        log.info("Opening files with lazy loading...")
        # Use chunks to enable dask arrays
        combined_ds = xr.open_mfdataset(
            nc_files,
            chunks={"time": 100},  # Chunk along time dimension
            combine="nested",  # Use nested concatenation instead of by_coords
            concat_dim="time",  # Explicitly specify concatenation dimension
            parallel=True,  # Enable parallel reading
        )

        log.info("Successfully opened %d files with lazy loading", len(nc_files))
        return combined_ds

    except Exception as e:
        log.warning("Lazy loading failed, falling back to chunked processing: %s", e)
        return load_and_combine_datasets_chunked(extract_dir)


def load_and_combine_datasets(extract_dir):
    """Load all netCDF files and combine them - optimized version"""
    return load_and_combine_datasets_lazy(extract_dir)


def extract_summer_precipitation(dataset, aggregation="sum", agg_months=None):
    if agg_months is None:
        agg_months = [7, 8, 9]
    """Extract summer (JAS) precipitation and aggregate by year with memory optimization"""
    log.info("Extracting summer months (July, August, September)...")

    # First, we need to convert time coordinate to proper datetime
    log.info("Converting time coordinate to datetime...")
    if not hasattr(dataset.time, "dt"):
        # Time coordinate is not datetime, need to convert
        # Based on the data structure, time seems to be month numbers within years
        # We need to create proper datetime coordinates
        log.info("Time coordinate is not datetime, creating proper time axis...")

        # Get the time values and create datetime coordinates
        time_values = dataset.time.values
        # Assuming time values represent months from start of the dataset
        # We need to create proper datetime based on the file naming pattern
        # For now, let's use a simple approach based on the time values

        # Create datetime coordinates based on time values
        # This is a simplified approach - in practice you'd need to parse the file names
        # or use the time attributes to determine the actual dates
        # Create a simple datetime range starting from 1901
        # This is a workaround - ideally you'd parse the actual dates from file names
        start_date = pd.Timestamp("1901-01-01")
        time_coords = pd.date_range(
            start=start_date, periods=len(time_values), freq="MS"
        )

        # Assign the new time coordinates
        dataset = dataset.assign_coords(time=time_coords)
        log.info(
            "Created datetime coordinates from %s to %s",
            time_coords[0],
            time_coords[-1],
        )

    # Select summer months (7=July, 8=August, 9=September)
    log.info("Selecting summer months from dataset...")
    summer_ds = dataset.sel(time=dataset.time.dt.month.isin(agg_months))

    # Handle both Dataset and DataArray
    if hasattr(summer_ds, "time"):
        time_coord = summer_ds.time
    else:
        # If it's a DataArray, get time from coordinates
        time_coord = summer_ds.coords["time"]

    start_year = time_coord.dt.year.min().item()
    end_year = time_coord.dt.year.max().item()
    log.info("Summer data covers %d to %d", start_year, end_year)

    # Aggregate by year with memory optimization
    log.info("Calculating seasonal %s...", aggregation)
    if aggregation == "sum":
        summer_agg = summer_ds.groupby("time.year").sum()
    elif aggregation == "mean":
        summer_agg = summer_ds.groupby("time.year").mean()
    else:
        raise ValueError("Aggregation must be 'sum' or 'mean'")

    # Compute the result to trigger actual calculation
    log.info("Computing aggregated results...")
    summer_agg = summer_agg.compute()

    # Add metadata
    summer_agg.attrs["title"] = f"Summer (JAS) Precipitation - {aggregation.title()}"
    summer_agg.attrs[
        "description"
    ] = f"Summer (July-August-September) precipitation {aggregation} by year"
    summer_agg.attrs["created"] = pd.Timestamp.now().isoformat()
    return summer_agg


@main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig | None = None):
    assert cfg is not None, "cfg is None"
    data_dir = cfg.ds.instrumental.input
    output_file = cfg.ds.instrumental.output
    agg_months = cfg.ds.instrumental.agg_months
    agg_method = cfg.ds.instrumental.agg_method

    log_memory_usage("at start")

    try:
        # Step 1: Load and combine datasets
        log.info("Step 1: Loading and combining datasets...")
        combined_ds = load_and_combine_datasets(data_dir)
        log_memory_usage("after loading datasets")

        log.info("Combined dataset info:")
        # Handle both Dataset and DataArray
        if hasattr(combined_ds, "time"):
            time_coord = combined_ds.time
        else:
            time_coord = combined_ds.coords["time"]

        # Check if time coordinate has dt attribute
        if hasattr(time_coord, "dt"):
            log.info(
                "- Time range: %d to %d",
                time_coord.dt.year.min().item(),
                time_coord.dt.year.max().item(),
            )
        else:
            log.info(
                "- Time range: %s to %s",
                time_coord.min().item(),
                time_coord.max().item(),
            )

        if hasattr(combined_ds, "data_vars"):
            log.info("- Variables: %s", list(combined_ds.data_vars))
        else:
            log.info("- Variable: %s", combined_ds.name)
        log.info("- Dimensions: %s", dict(combined_ds.dims))

        # Step 2: Extract summer precipitation
        log.info("Step 2: Extracting summer precipitation...")
        summer_precip = extract_summer_precipitation(
            combined_ds,
            aggregation=agg_method,
            agg_months=agg_months,
        )
        log_memory_usage("after extracting summer precipitation")

        # Clear the large combined dataset from memory
        del combined_ds
        force_garbage_collection()
        log_memory_usage("after clearing combined dataset")

        # Step 3: Save to netCDF
        log.info("Step 3: Saving to %s...", output_file)
        summer_precip.to_netcdf(output_file)
        log_memory_usage("after saving")

        log.info("Success! Summer precipitation dataset saved to: %s", output_file)

        # Print summary statistics
        if hasattr(summer_precip, "data_vars"):
            precip_var = list(summer_precip.data_vars)[
                0
            ]  # Assume first variable is precipitation
            precip_data = summer_precip[precip_var]
        else:
            precip_data = summer_precip

        log.info("Dataset summary:")
        log.info(
            "- Years: %d (%d - %d)",
            len(summer_precip.year),
            summer_precip.year.min().item(),
            summer_precip.year.max().item(),
        )
        log.info("- Spatial extent: %s", summer_precip.dims)
        log.info("- Mean summer precipitation: %.2f mm", float(precip_data.mean()))
        log.info("- Max summer precipitation: %.2f mm", float(precip_data.max()))

    except Exception as e:
        log.error("Error: %s", e)
        log_memory_usage("at error")
        sys.exit(1)


if __name__ == "__main__":
    _main()
