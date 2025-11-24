#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ChopPictures.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/21 15:27 
"""
import os
import cv2
import numpy as np
from math import ceil


def crop_and_merge_frames(
    folder,
    i,
    j,
    top_left,
    bottom_right,
    rows,
    cols,
    output_path,
    margin=10,
    pixel_scale=10,
    font_scale=1.0,
    thickness=2
):
    """
    批量裁剪并拼接图片（每个像素可放大为 pixel_scale × pixel_scale 块，裁剪块间加入 margin 留白）

    参数:
        folder: 图片文件夹路径
        i, j: 处理第 i~j 帧（按排序从 1 开始）
        top_left:    [H, W] 裁剪区左上
        bottom_right:[H, W] 裁剪区右下
        rows, cols: 网格布局（rows 行 × cols 列）
        margin: 拼接时各块之间的留白（像素）
        pixel_scale: 每个像素放大倍数（1 表示不放大）
        output_path: 输出保存路径
    """
    i+=1
    j+=1

    # --------------------------
    # 1. 获取文件列表
    # --------------------------
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    )

    selected = files[i-1:j]

    if len(selected) == 0:
        raise ValueError("未选中任何图片，请检查 i, j")

    # --------------------------
    # 2. 逐张裁剪 + 放大像素
    # --------------------------
    crops = []
    for idx, fname in enumerate(selected, start=i):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"无法读取: {path}")

        h1, w1 = top_left
        h2, w2 = bottom_right
        crop = img[h1:h2, w1:w2]

        # 插入标题
        cv2.putText(
            crop,
            f"Frame {idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # 将每个像素放大为 pixel_scale × pixel_scale
        if pixel_scale > 1:
            crop = cv2.resize(
                crop,
                (crop.shape[1] * pixel_scale, crop.shape[0] * pixel_scale),
                interpolation=cv2.INTER_NEAREST
            )

        crops.append(crop)

    # --------------------------
    # 3. 创建画布（考虑 margin）
    # --------------------------
    h_crop, w_crop = crops[0].shape[:2]

    canvas_h = rows * h_crop + (rows + 1) * margin
    canvas_w = cols * w_crop + (cols + 1) * margin

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 20  # 背景可略微偏亮

    # --------------------------
    # 4. 逐个贴到画布
    # --------------------------
    for k, crop in enumerate(crops):
        r = k // cols
        c = k % cols
        if r >= rows:
            break

        y = margin + r * (h_crop + margin)
        x = margin + c * (w_crop + margin)

        canvas[y:y+h_crop, x:x+w_crop] = crop

    # --------------------------
    # 5. 保存输出
    # --------------------------
    cv2.imwrite(output_path, canvas)
    print(f"已保存到: {output_path}")

if __name__ == "__main__":
    crop_and_merge_frames(
        folder=r"E:\MyFiles\data\CAPPI0010\20250408",
        i=110,
        j=130,
        top_left=[634, 483],
        bottom_right=[641,490],
        rows=4,
        cols=5,
        output_path=r"E:\MyFiles\data/merged1.png"
    )
    crop_and_merge_frames(
        folder=r"E:\MyFiles\data\CAPPI0010\20250408",
        i=136,
        j=156,
        top_left=[634, 483],
        bottom_right=[641,490],
        rows=4,
        cols=5,
        output_path=r"E:\MyFiles\data/merged2.png"
    )
