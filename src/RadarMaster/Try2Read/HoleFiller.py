#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：HoleFiller.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/10/23 20:04 
"""
import numpy as np
import cv2
from skimage.measure import label, regionprops

def detect_holes(frame, low_ratio=0.3, min_size=100):
    """
    检测包括半开放空洞的区域
    :param frame: 2D numpy array
    :param low_ratio: 空洞像素与局部均值的比值阈
    """
    # 局部平滑估计背景
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    diff = blurred - frame

    # 找出低于背景的区域
    mask = (diff > blurred * (1 - low_ratio)).astype(np.uint8)

    # 去除太小的区域
    labeled = label(mask)
    hole_mask = np.zeros_like(mask)
    for region in regionprops(labeled):
        if region.area >= min_size:
            for (x, y) in region.coords:
                hole_mask[x, y] = 1
    return hole_mask

def close_open_holes(mask, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

def smart_fill(frame, mask, area_thresh=5000):
    labeled = label(mask)
    filled = frame.copy()

    large_mask = np.zeros_like(mask)
    for region in regionprops(labeled):
        if region.area < area_thresh:
            coords = region.coords
            bbox = region.bbox
            pad = 5
            x1 = max(bbox[0]-pad, 0)
            y1 = max(bbox[1]-pad, 0)
            x2 = min(bbox[2]+pad, frame.shape[0])
            y2 = min(bbox[3]+pad, frame.shape[1])
            local = frame[x1:x2, y1:y2]
            fill_value = np.median(local[local > 0]) if np.any(local > 0) else 0
            filled[coords[:, 0], coords[:, 1]] = fill_value
        else:
            coords = region.coords
            large_mask[coords[:, 0], coords[:, 1]] = 1

    # 使用 inpaint 处理大空洞
    norm = cv2.normalize(filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inpainted = cv2.inpaint(norm, large_mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    result = inpainted.astype(float) / 255 * frame.max()
    return result

def adaptive_hole_repair(frame):
    # Step 1: 检测空洞（包括半开放）
    hole_mask = detect_holes(frame, low_ratio=0.4, min_size=200)

    # Step 2: 形态学封闭
    closed_mask = close_open_holes(hole_mask, iterations=3)

    # Step 3: 智能填补
    repaired = smart_fill(frame, closed_mask, area_thresh=4000)
    return repaired



# 1️⃣ 读取图片（注意：JPG不能用 np.load 加载！）
# 如果是普通图片文件：
image = cv2.imread(r"C:\Users\Me\Desktop\CAPPI\aaa.png", cv2.IMREAD_GRAYSCALE)

# 2️⃣ 调用修复函数
repaired = adaptive_hole_repair(image)

# 3️⃣ 保存修复后的图片
save_path = r"C:\Users\Me\Desktop\CAPPI\aaa-filled.png"
cv2.imwrite(save_path, repaired)

print(f"修复后的图片已保存到：{save_path}")

