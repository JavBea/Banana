#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ToVideoNPY.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/18 17:00 
"""
import numpy as np
import cv2

def gray_npy_to_video(input_npy_path, output_video_path, fps=25):
    """
    将存储灰度图序列 (T, H, W) 的 NPY 文件可视化为视频 MP4。

    参数:
        input_npy_path : str
            输入灰度序列 .npy 文件路径，形状应为 (T, H, W)
        output_video_path : str
            输出视频文件路径，如 output.mp4
        fps : int
            视频帧率
    """

    # 加载灰度图序列
    gray_array = np.load(input_npy_path)
    assert gray_array.ndim == 3, "输入 NPY 必须为 (T,H,W) 的单通道灰度序列"

    T, H, W = gray_array.shape

    # 初始化视频写入器（mp4）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H), isColor=False)

    for i in range(T):
        frame = gray_array[i]

        # cv2 要求 uint8 类型
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # 写入灰度帧
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已生成：{output_video_path}")

if __name__ == '__main__':
    gray_npy_to_video(r"E:\MyFiles\data\CAPPI0408_images_single_denoise2.npy", r"E:\MyFiles\data\CAPPI0408_images_single_denoise2.mp4", fps=10)
