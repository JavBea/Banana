#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：将指定文件夹下的所有图片合成为一个视频，并可指定帧率
@File    ：ToVideoPic.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/13 19:09 
"""
import cv2
import os
from natsort import natsorted  # 用于按自然顺序排序文件名（可选）

def ToVideo(input_path, output_path, fps=10):


    # ==== 获取图片文件 ====
    # 支持常见图片格式，可根据需要扩展
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(input_path) if f.lower().endswith(extensions)]

    # 对文件名进行排序（确保视频帧顺序正确）
    images = natsorted(images)

    if not images:
        raise ValueError("文件夹中未找到图片！")

    # ==== 读取第一张图片以确定尺寸 ====
    first_image = cv2.imread(os.path.join(input_path, images[0]))
    if first_image is None:
        raise ValueError("无法读取第一张图片，请检查文件路径或格式。")

    height, width, _ = first_image.shape

    # ==== 初始化视频写入对象 ====
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ==== 将图片逐帧写入视频 ====
    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️ 跳过无法读取的图片: {img_name}")
            continue

        # 若尺寸不一致，可自动调整到第一张图的大小
        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    # ==== 释放资源 ====
    out.release()
    print(f"✅ 视频合成完成：{output_path}")


if __name__ == "__main__":
    ToVideo(input_path=r"E:\MyFiles\data\CAPPI0408_single_denoise1", output_path=r"E:\MyFiles\data\20250408_single_denoise1.mp4")