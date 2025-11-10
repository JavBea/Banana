#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** `
@File    ：Bin.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/10/1 16:27 
"""

import cv2
import os

def crop_radar_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)

            # 裁剪核心区域 [y1:y2, x1:x2]
            cropped = img[12:1012, 12:1012]

            cv2.imwrite(os.path.join(output_dir, file), cropped)
            print(f"已处理: {file}")

# 示例调用
crop_radar_images(r"C:\Users\Me\Desktop\CAPPI\20250408", r"C:\Users\Me\Desktop\CAPPI\output")
