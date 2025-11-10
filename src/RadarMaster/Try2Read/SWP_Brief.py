#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：SWP_Brief.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/9 10:30 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def visualize_radar_data(file_path, encoding='uint8', row_len=None, cmap='viridis', certain_width=True):
    """
    读取二进制文件，跳过前1036个字节，处理后面的数据并可视化为图像。固定长度

    :param file_path: 二进制文件路径
    :param encoding: 数据编码方式（默认为uint8），可以是其他类型如'uint16'等
    :param row_len: 每行数据的长度（可以计算得出，或者手动设置）
    :param cmap: 颜色映射，默认为viridis
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
    else:
        data_reshaped = np.resize(data, (row_len, num_rows))

    # 应用九宫格平均滤波，使得每个像素点的颜色是其周围九宫格的平均颜色
    smoothed_data = uniform_filter(data_reshaped, size=3)

    # 可视化数据
    plt.imshow(smoothed_data, cmap=cmap, interpolation='none')
    plt.colorbar()  # 显示色条
    plt.title(f"Radar Data Visualization size: {data_reshaped.shape}")
    plt.show()


file_path = r"G:TY\20250228\SWP\D20250228090622.SPF"

# SWP最佳长宽：（340，300） 编码方式：uint16
visualize_radar_data(file_path,row_len=300,encoding='uint16',certain_width=True)

# 固定宽度
# for length in range(128,512,1):
#     visualize_radar_data(file_path,row_len=length,encoding='uint16',certain_width=True)

# 固定长度
# for length in range(604,702,1):
#     visualize_radar_data(file_path,row_len=length,encoding='uint16',certain_width=False)