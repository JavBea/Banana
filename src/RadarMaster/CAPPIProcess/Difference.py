#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：Difference.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/21 19:13 
"""
import numpy as np

def compute_frame_diff(input_path, output_path):
    """
    对 [T, H, W] 形状的雷达数据逐帧求差，并保存为新 npy 文件。

    参数:
        input_path:  输入 npy 文件路径
        output_path: 输出 npy 文件路径
    """
    # 读取数据
    data = np.load(input_path)  # shape: [T, H, W]

    if data.ndim != 3:
        raise ValueError("数据必须是 [T, H, W] 三维数组")

    # 逐帧差分
    # result[t] = data[t+1] - data[t]
    diff = data[1:] - data[:-1]  # shape: [T-1, H, W]

    # 保存
    np.save(output_path, diff)
    print(f"完成！差分数据已保存到: {output_path}")

if __name__ == "__main__":
    compute_frame_diff(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_single_difference.npy"
    )
