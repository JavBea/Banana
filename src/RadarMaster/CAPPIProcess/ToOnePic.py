#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ToOnePic.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/21 15:40 
"""
import os
import cv2
import numpy as np


def merge_frames_range(
    folder,
    start_frame,
    end_frame,
    rows,
    cols,
    output_path,
    margin=10,
    bg_color=(20, 20, 20)
):
    """
    按文件名字典序排序后，从第 start_frame 到 end_frame 读取图片，
    按 rows × cols 拼成一张大图。

    字典序示例：1.png < 10.png < 2.png < 20.png < 3.png
    """

    # -------------------------------
    # 1. 获取图片文件（严格字典序）
    # -------------------------------
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    files = sorted(files)   # **关键：字典序排序**

    total = len(files)
    if start_frame < 1 or end_frame > total or start_frame > end_frame:
        raise ValueError(f"帧编号不合法，文件夹内共有 {total} 张图片")

    # -------------------------------
    # 2. 选取 start~end 区间
    # -------------------------------
    selected_files = files[start_frame - 1 : end_frame]

    if len(selected_files) == 0:
        raise RuntimeError("选中范围为空")

    target_count = rows * cols

    # -------------------------------
    # 3. 读取图片
    # -------------------------------
    imgs = []
    for fname in selected_files[:target_count]:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            raise RuntimeError(f"无法读取文件: {fname}")
        imgs.append(img)

    # 若不足 rows*cols，用背景色补齐
    if len(imgs) < target_count:
        h, w = imgs[0].shape[:2]
        blank = np.zeros((h, w, 3), np.uint8)
        blank[:] = bg_color
        while len(imgs) < target_count:
            imgs.append(blank.copy())

    # -------------------------------
    # 4. 统一尺寸
    # -------------------------------
    h0, w0 = imgs[0].shape[:2]
    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (w0, h0), interpolation=cv2.INTER_AREA)

    # -------------------------------
    # 5. 创建画布
    # -------------------------------
    H = rows * h0 + (rows + 1) * margin
    W = cols * w0 + (cols + 1) * margin

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = bg_color

    # -------------------------------
    # 6. 放置每张图
    # -------------------------------
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols

        y = margin + r * (h0 + margin)
        x = margin + c * (w0 + margin)

        canvas[y:y+h0, x:x+w0] = img

    # -------------------------------
    # 7. 保存
    # -------------------------------
    cv2.imwrite(output_path, canvas)
    print(f"拼接完成 -> {output_path}")

merge_frames_range(
    folder=r"E:\MyFiles\data\Histograms",
    start_frame=110,
    end_frame=130,
    rows=4,
    cols=5,
    margin=20,
    bg_color=(30, 30, 30),  # 深灰
    output_path=r"E:\MyFiles\data/Histograms_merged1.png"
)

merge_frames_range(
    folder=r"E:\MyFiles\data\Histograms",
    start_frame=136,
    end_frame=156,
    rows=4,
    cols=5,
    margin=20,
    bg_color=(30, 30, 30),  # 深灰
    output_path=r"E:\MyFiles\data/Histograms_merged2.png"
)
