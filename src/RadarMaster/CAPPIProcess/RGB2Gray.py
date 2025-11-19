#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RGB2Gray.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/10 16:29 
"""
import numpy as np

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def rgb_to_gray(r, g, b):
    """计算灰度值（加权平均法）"""
    return 0.299 * r + 0.587 * g + 0.114 * b


def print_gray_values(hex_colors):
    """依次计算并输出每个颜色的灰度值"""
    for hex_color in hex_colors:
        r, g, b = hex_to_rgb(hex_color)
        gray = rgb_to_gray(r, g, b)
        print(f"{hex_color}: r:{r},g:{g},b:{b}灰度值 = {gray:.2f}")


def rgb_to_gray_npy(input_path, output_path):
    # 加载原始 RGB npy 文件
    rgb_data = np.load(input_path)   # shape: (239,1000,1000,3)

    # 检查形状
    if rgb_data.ndim != 4 or rgb_data.shape[-1] != 3:
        raise ValueError("输入文件形状必须为 [N, H, W, 3]")

    # 使用加权公式转换为灰度
    # 结果 shape: (239,1000,1000)
    gray_data = np.dot(rgb_data[...,:3], [0.299, 0.587, 0.114])

    # 保存新的灰度 npy 文件
    np.save(output_path, gray_data)
    print("灰度文件已保存到:", output_path)


if __name__ == "__main__":
    rgb_to_gray_npy(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_origin_v2.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_gray.npy"
    )
