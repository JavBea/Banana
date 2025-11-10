#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：EBPPI_Mapping.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/12 20:36 
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

padding_value = 32768
colors = None
bounds = None
new_min = 0
new_max = 0

# 放缩函数
def rescale(matrix):
    old_min = np.nanmin(matrix)
    old_max = np.nanmax(matrix)

    return (matrix - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def get_from_json(data_type="EBPPI"):
    global colors
    global bounds
    global new_min
    global new_max

    with open("colormaps.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    colors = data[data_type]["colors"]  # 一维数组
    bounds = data[data_type]["bounds"]  # 一维数组
    new_min = data[data_type]["min"]
    new_max = data[data_type]["max"]

    print(colors)
    print(bounds)

def plot_histogram(arr, bins=50, title="Value Distribution"):
    """
    绘制二维数组数值分布的直方图

    :param arr: 二维 numpy 数组
    :param ignore_value: 要忽略的值（默认32768）
    :param bins: 直方图的分箱数量
    :param title: 图表标题
    """
    # 转为一维并过滤无效值
    values = arr.ravel()

    if values.size == 0:
        print("⚠️ 数组中没有有效值")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("数值")
    plt.ylabel("频数")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def filter(arr):
    arr_float = arr.astype(float)  # 转成 float 才能放 NaN
    arr_float[arr_float == padding_value] = np.nan  #将填充值置为NAN

    min_val = np.nanmin(arr_float)
    max_val = np.nanmax(arr_float)

    print("最小值:", min_val)
    print("最大值:", max_val)
    return arr_float,min_val,max_val

def visualize_radar_data(file_path,
                         encoding='uint8',
                         row_len=None,
                         certain_width=True,
                         filtered_value=None):
    """
    读取二进制文件，跳过前1036个字节，处理后面的数据并可视化为图像。固定长度

    :param file_path: 二进制文件路径
    :param encoding: 数据编码方式（默认为uint8），可以是其他类型如 'uint16'等
    :param row_len: 每行数据的长度（可以计算得出，或者手动设置）
    :param certain_width: 是否固定宽度（默认为True）
    :param filtered_value: 过滤缺测填充值（默认为None）
    """
    # 读取文件并跳过前1036个字节
    with open(file_path, 'rb') as f:
        f.seek(1036)  # 跳过前1036个字节
        raw_data = f.read()

    # 将二进制数据解码为指定类型的数组
    data = np.frombuffer(raw_data, dtype=encoding)

    # 根据数据的总长度和指定的行长度计算图像的宽度
    total_pixels = len(data)
    if row_len is None:
        row_len = int(np.sqrt(total_pixels))  # 默认宽度为总像素数的平方根，求近似
    num_rows = (total_pixels // row_len) + (1 if total_pixels % row_len else 0)

    # 将数据reshape为二维数组，并填充不足的部分
    if certain_width:
        data_reshaped = np.resize(data, (num_rows, row_len))
        # # 手动将不足的行填充padding_value
        # expected_size = num_rows * row_len
        #
        # if len(data) < expected_size:
        #     # 不够 → 在尾部补 32768
        #     data = np.pad(data, (0, expected_size - len(data)), constant_values=padding_value)
        # else:
        #     # 太多 → 截断
        #     data = data[:expected_size]
        #
        # data_reshaped = data.reshape(num_rows, row_len)
    else:
        data_reshaped = np.resize(data, (row_len, num_rows))

    # 应用九宫格平均滤波，使得每个像素点的颜色是其周围九宫格的平均颜色
    # smoothed_data = uniform_filter(data_reshaped, size=3)

    # 仅保留有效数据
    filtered_data, min_v, max_v=filter(data_reshaped)

    # 有效数据的分布可视化
    # plot_histogram(filtered_data)

    # 放缩数据
    final_data = rescale(filtered_data)
    plot_histogram(final_data)

    # 如果用户提供了 bounds 和 colors，就构建自定义 colormap
    if bounds is not None and colors is not None:
        if len(colors) != (len(bounds) - 1):
            raise ValueError("colors 数量必须比 bounds 少 1")
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    else:
        cmap = "viridis"
        norm = None

    # 可视化数据
    plt.imshow(final_data, cmap=cmap, norm=norm, interpolation='none')
    plt.colorbar()  # 显示色条
    plt.title(f"Radar Data Visualization size: {data_reshaped.shape}")
    plt.show()


file_path = r"C:\Users\Me\Desktop\雷达数据\20250228\EBPPI\D20250228090622.EBF"

get_from_json()

# EBPPI最佳长宽：（300，300） 编码方式：uint16
visualize_radar_data(file_path,row_len=300,encoding='uint16',certain_width=True)

# 固定宽度
# for length in range(128,512,1):
#     visualize_radar_data(file_path,row_len=length,encoding='uint16',certain_width=True)

# 固定长度
# for length in range(604,702,1):
#     visualize_radar_data(file_path,row_len=length,encoding='uint16',certain_width=False)
