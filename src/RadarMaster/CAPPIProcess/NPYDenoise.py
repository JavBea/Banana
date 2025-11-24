#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：NPYDenoise.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/14 15:45 
"""

import numpy as np

# step0.5
def replace_negative_zero(input_path, output_path):
    """
    读取 npy 文件，将所有 负数 替换为 NaN，不处理 NaN，并保存到指定位置。
    """
    # 读取数据
    data = np.load(input_path)

    # 将负数置 0
    data[data < 0] = np.nan

    # 保存处理后的数据
    np.save(output_path, data)

    print(f"处理完成：保存到 {output_path}")

    return data

# step1.v1
def replace_with_zero(input_path, output_path):
    """
    读取 npy 文件，将所有 NaN 替换为 0，并保存到指定位置。
    """
    # 读取数据
    data = np.load(input_path)

    # 将 NaN 替换为 0
    data = np.nan_to_num(data, nan=0.0)

    # 将负数置 0
    data[data < 0] = 0

    # 保存处理后的数据
    np.save(output_path, data)

    print(f"处理完成：保存到 {output_path}")

    return data

# step1.v2
def replace_with_zero_in_circle(input_path, output_path, r):
    """
    读取 npy 文件，仅在每一帧以中心为圆心、半径 r 的圆内将负值替换为 0。
    其它区域保持原样。
    """
    # 读取数据: shape = (frames, H, W)
    data = np.load(input_path)
    frames, H, W = data.shape

    # 将 NaN 替换为 0
    data = np.nan_to_num(data, nan=0.0)

    # === 构建圆形 mask（只需构建一次） ===
    cy, cx = H // 2, W // 2  # 圆心
    y, x = np.ogrid[:H, :W]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2   # True 表示在圆内

    # === 逐帧处理 ===
    for i in range(frames):
        frame = data[i]

        # 仅对圆内区域做操作：frame[mask]
        neg_mask = (frame < 0) & mask
        frame[neg_mask] = 0

        data[i] = frame

    # 保存处理后的数据
    np.save(output_path, data)
    print(f"处理完成：保存到 {output_path}")

    return data



if __name__ == "__main__":

    replace_negative_zero(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_single_denoise0.5.npy"
    )

    replace_with_zero(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_single_denoise1.v1.npy"
    )

    replace_with_zero_in_circle(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_single_denoise1.v2.npy",
        r=128
    )

    # from VisualizeNPY import show_npy_frame, show_npy_frame_ignore_percent
    # for i in range(0,239):
    #     show_npy_frame_ignore_percent(
    #         r"E:\MyFiles\data\CAPPI0408_images_single_denoise1.npy",
    #         i
    #     )

