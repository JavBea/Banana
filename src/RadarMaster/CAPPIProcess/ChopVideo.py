#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ChopVideo.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/13 19:07 
"""

import cv2

def chop(input_path,output_path,r1, c1, r2, c2):

    # ==== 打开视频 ====
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取帧率、尺寸等信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 输出视频大小为裁剪区域的高宽
    height, width = r2 - r1, c2 - c1
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ==== 逐帧读取并裁剪 ====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按矩阵坐标裁剪（注意：frame[row_start:row_end, col_start:col_end]）
        cropped = frame[r1:r2, c1:c2]

        # 写入输出视频
        out.write(cropped)

    # ==== 释放资源 ====
    cap.release()
    out.release()
    print("裁剪完成，输出视频已保存到：", output_path)

points=[[215,513],
        [219,536],
        [197,532],
        [481,503],
        [473,405],
        [673,489],
        [706,482],
        [636,760],
        [360,806],]
size = 7
for r,c in points:
    chop(input_path=r"E:\MyFiles\data\20250408.mp4",
         output_path=rf"E:\MyFiles\data\CAPPI0010\20250408_({r},{c})_{size}size.mp4",
         r1=r,
         c1=c,
         r2=r+size,
         c2=c+size)