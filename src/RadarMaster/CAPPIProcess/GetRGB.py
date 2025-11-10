#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：GetRGB.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/10 16:17 
"""
from PIL import Image

def get_pixel_hex_color(image_path, width, height):
    """
    获取图片指定坐标的RGB值（16进制格式）
    :param image_path: 图片路径
    :param width: 横坐标
    :param height: 纵坐标
    :return: 16进制RGB颜色字符串，例如 '#A1B2C3'
    """
    # 打开图片
    img = Image.open(image_path).convert('RGB')

    # 获取像素值 (R, G, B)
    r, g, b = img.getpixel((width, height))

    # 转换为16进制格式
    hex_color = f'#{r:02X}{g:02X}{b:02X}'
    return hex_color


# 示例调用
if __name__ == "__main__":
    path = r"E:\MyFiles\data\CAPPI0010\20250408\CAPPI-202504080000-0010-150-Z.JPG"   # 图片路径
    x, y = 539, 131      # 坐标点
    color = get_pixel_hex_color(path, x, y)
    print(f"坐标({x}, {y})的颜色是：{color}")
