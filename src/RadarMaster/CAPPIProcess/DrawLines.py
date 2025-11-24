#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：DrawLines.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/19 11:22 
"""
import cv2

def draw_line_on_image(
    img_path,
    output_path,
    p1,
    p2,
    color=(0, 0, 255),
    thickness=2
):
    """
    在一张彩色图像上画一条直线，并保存到指定位置。

    参数:
        img_path: str      输入图片路径
        output_path: str   输出图片路径
        p1: (h1, w1)       直线起点（注意：h 是行，w 是列）
        p2: (h2, w2)       直线终点
        color: (B, G, R)   线的颜色（默认红色）
        thickness: int     线宽
    """

    # 读取图像（保持彩色）
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")

    # h 是 y、w 是 x，但 OpenCV 的坐标格式是 (x, y)
    (h1, w1) = p1
    (h2, w2) = p2

    # 画线
    cv2.line(img, (w1, h1), (w2, h2), color, thickness)

    # 保存结果
    cv2.imwrite(output_path, img)

    return output_path

if __name__ == "__main__":
    # draw_line_on_image(
    #     img_path=r"E:\MyFiles\data\static\CAPPI-202504080012-0010-150-Z.JPG",
    #     output_path=r"E:\MyFiles\data\bbb.JPG",
    #     p1=(500, 500),
    #     p2=(116, 514),
    #     color=(30, 30, 255),
    #     thickness=1
    # )
    draw_line_on_image(
        img_path=r"E:\MyFiles\data\bbb.JPG",
        output_path=r"E:\MyFiles\data\bbb.JPG",
        p1=(500, 500),
        p2=(122, 568),
        color=(30, 30, 255),
        thickness=1
    )
