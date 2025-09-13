#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
测试数据，value 是自然记录，expect 是历史记录，classified 是分类结果


value	classified	expect	exact	last	diff
0	0.11	0	0	True	NaN	NaN
1	0.34	1	1	True	NaN	NaN
2	1.10	2	1	False	NaN	NaN
3	0.10	-1	0	False	NaN	NaN
4	0.32	2	2	True	1.10	-0.78
5	0.33	0	2	False	0.11	0.22
6	3.00	2	2	True	0.32	2.68
7	0.11	0	0	True	0.33	-0.22
"""

import pandas as pd

from past1000.calibration import check_estimation

test_df = pd.DataFrame(
    {
        "value": [0.11, 0.34, 1.10, 0.10, 0.32, 0.33, 3.0, 0.11],
        "classified": [0, 1, 2, -1, 2, 0, 2, 0],
        "expect": [0, 1, 1, 0, 2, 2, 2, 0],
    }
)

check_estimation(test_df, "value", "expect", "classified")
