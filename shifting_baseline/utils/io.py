#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""Helpers for writing manuscript tables into the shared workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shifting_baseline.utils.log import get_logger

log = get_logger(__name__)

# 手工维护、各脚本共享的表格工作簿（位于 ``cfg.ds.figs``）
TABLES_WORKBOOK = "CollMemo_Tables.xlsx"


def write_table_sheet(figs_dir: str | Path, table: pd.DataFrame, sheet: str) -> None:
    """把表格写入 ``<figs_dir>/CollMemo_Tables.xlsx`` 的指定 sheet,保留其它手工维护的表。"""
    xlsx_path = Path(figs_dir) / TABLES_WORKBOOK
    if not xlsx_path.exists():
        log.warning("工作簿不存在,跳过写入 xlsx: %s", xlsx_path)
        return
    # mode="a" + if_sheet_exists="replace":仅新增/替换该 sheet,不动其它 sheet
    with pd.ExcelWriter(
        xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        table.to_excel(writer, sheet_name=sheet, index=False)
    log.info("已写入表格到 %s [%s]", xlsx_path.name, sheet)
