#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：DifferenceBased_denoise.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/10/1 16:26 
"""

import numpy as np
import cv2
import os
from glob import glob


def load_png_sequence(folder_path, ext="png"):
    """
    从文件夹中读取一批PNG图像，并按文件名排序
    返回 numpy array shape (N, H, W)
    """
    files = sorted(glob(os.path.join(folder_path, f"*.{ext}")))
    if not files:
        raise FileNotFoundError(f"No {ext} files found in {folder_path}")

    frames = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # 雷达图通常单通道
        if img is None:
            raise IOError(f"Failed to read {f}")
        frames.append(img)

    data = np.stack(frames, axis=0)  # (N, H, W)
    return data, files


def detect_static_noise_by_diff(data, th_percentile=10, min_area=200):
    """
    data: numpy array shape (N, H, W)
    return: mask (H, W), suspected static noise region
    """
    N = data.shape[0]
    # compute abs differences between adjacent frames
    diffs = np.abs(data[1:] - data[:-1])  # shape (N-1, H, W)

    stat = np.median(diffs, axis=0)  # (H,W)

    th = np.percentile(stat.ravel(), th_percentile)
    mask = (stat < th).astype(np.uint8) * 255

    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out_mask[labels == i] = 255
    return out_mask  # 255 = suspected static noise


if __name__ == "__main__":
    # 指定你的PNG文件夹路径
    folder = r"E:\RadarData\images"  # 修改为你的路径

    data, files = load_png_sequence(folder)

    mask = detect_static_noise_by_diff(data, th_percentile=10, min_area=500)

    # 保存mask结果
    out_mask_path = os.path.join(folder, "detected_noise_mask.png")
    cv2.imwrite(out_mask_path, mask)

    print(f"噪声mask已保存到: {out_mask_path}")
