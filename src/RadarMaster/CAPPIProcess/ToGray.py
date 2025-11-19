#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ToGray.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/18 16:45 
"""
import os
import numpy as np
from PIL import Image

import numpy as np

def convert_rgb_npy_to_gray_npy(input_npy_path, output_npy_path):
    """
    将存储 RGB 图像序列 (T,H,W,3) 的 NPY 文件转换为 (T,H,W) 的灰度 NPY 文件。

    参数:
        input_npy_path: str
            输入 RGB 序列 .npy 文件路径
        output_npy_path: str
            输出灰度序列 .npy 文件路径
    """

    # 读取原始RGB序列
    rgb_array = np.load(input_npy_path)
    assert rgb_array.ndim == 4 and rgb_array.shape[-1] == 3, \
        "输入 NPY 必须为 [T,H,W,3] 的 RGB 图像序列"

    # RGB 转灰度: 采用标准加权方式
    gray_array = (
        0.299 * rgb_array[..., 0] +
        0.587 * rgb_array[..., 1] +
        0.114 * rgb_array[..., 2]
    ).astype(np.uint8)

    # 保存为新的 NPY 文件
    np.save(output_npy_path, gray_array)

    print(f"转换完成！灰度 NPY 已保存到：{output_npy_path}")


if __name__ == "__main__":
    convert_rgb_npy_to_gray_npy(
        input_npy_path=r"E:\MyFiles\data\CAPPI0408_images_origin_v2.npy",
        output_npy_path=r"E:\MyFiles\data\CAPPI0408_images_gray.npy",
    )