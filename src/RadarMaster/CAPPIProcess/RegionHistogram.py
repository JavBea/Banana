#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RegionHistogram.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/14 11:37 
"""
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def show_region_histograms(npy_path, top_left, bottom_right, frame_index=None, bins=50):
    """
    查看 (N, H, W) 雷达数据某区域内数值的直方图。

    坐标体系:
        - 左上角为原点 (0,0)
        - 坐标格式为 (height, width)

    参数：
        npy_path : str
            NPY 文件路径
        top_left : (y1, x1)
            左上角坐标 (height, width)
        bottom_right : (y2, x2)
            右下角坐标 (height, width)
        frame_index : int or None
            指定帧号。None = 遍历所有帧
        bins : int
            直方图 bin 数
    """
    data = np.load(npy_path)  # (N, H, W)
    N, H, W = data.shape

    y1, x1 = top_left
    y2, x2 = bottom_right

    # 区域提取（忽略 NaN）
    def extract_region(frame):
        region = frame[y1:y2, x1:x2]
        return region[~np.isnan(region)]

    # === 单帧模式 ===
    if frame_index is not None:
        values = extract_region(data[frame_index])

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=bins)
        plt.title(f"Frame {frame_index} Histogram  Region {top_left} → {bottom_right}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        return

    # === 多帧模式 ===
    for i in range(N):
        values = extract_region(data[i])

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=bins)
        plt.title(f"Frame {i} Histogram  Region {top_left} → {bottom_right}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


show_region_histograms(
    r"E:\MyFiles\data\CAPPI0408_images_gray.npy",
    top_left=(634, 483),
    bottom_right=(672, 557),
    # frame_index=10,   # 第10帧
    bins=60
)
