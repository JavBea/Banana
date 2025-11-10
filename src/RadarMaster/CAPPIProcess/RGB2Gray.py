#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RGB2Gray.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/10 16:29 
"""
def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def rgb_to_gray(r, g, b):
    """计算灰度值（加权平均法）"""
    return 0.299 * r + 0.587 * g + 0.114 * b


def print_gray_values(hex_colors):
    """依次计算并输出每个颜色的灰度值"""
    for hex_color in hex_colors:
        r, g, b = hex_to_rgb(hex_color)
        gray = rgb_to_gray(r, g, b)
        print(f"{hex_color}: r:{r},g:{g},b:{b}灰度值 = {gray:.2f}")


# 示例颜色列表
hex_colors = [
    "#9c9c9c","#767676","#aaaaff","#8c8cee","#7070c9","#00ffff","#0096ff","#0000ff",
    "#00ff00","#00c800","#009600","#ffff00","#ffc800","#ff7800","#ff0000",
    "#c80000","#960000","#ff00ff","#9600fa"
]

# 执行函数
print_gray_values(hex_colors)
