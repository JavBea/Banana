#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：EstimateGranularity.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/2 20:00 
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def haversine(lat1, lon1, lat2, lon2):
    """Haversine公式计算地球表面两点之间的距离（米）"""
    R = 6371000  # 地球半径，米
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def estimate_granularity_from_csv(filepath, plot=False):
    """
    从CSV文件读取经纬度点，计算点间距的统计信息
    文件需包含列: 经度, 纬度
    """
    df = pd.read_csv(filepath, sep=r"\s+")  # 如果是逗号分隔，可改 sep=","
    lon = df["经度"].values
    lat = df["纬度"].values

    if len(lat) < 2:
        raise ValueError("CSV文件至少需要两行经纬度数据")

    distances = [
        haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
        for i in range(len(lat) - 1)
    ]
    distances = np.array(distances)

    stats = {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "median": np.median(distances),
        "25%": np.percentile(distances, 25),
        "75%": np.percentile(distances, 75)
    }

    if plot:
        plt.hist(distances, bins=30, edgecolor="black")
        plt.xlabel("距离 (米)")
        plt.ylabel("频数")
        plt.title("相邻点间距分布")
        plt.show()

    return stats, distances


# 示例调用
if __name__ == "__main__":
    filepath = "../../static/balloon_data.txt"  # 替换为你的csv路径
    stats, distances = estimate_granularity_from_csv(filepath, plot=True)
    print("粒度统计：", stats)
