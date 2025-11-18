#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：DrawCircle.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/17 14:20 
"""
import math


def generate_circle_points(center, r, method="regular", **kwargs):
    """
    生成圆周点，用直线连接即可画圆。

    参数:
        center: (cx, cy)
        r: 半径
        method: "regular", "parametric", "midpoint", "adaptive"
        **kwargs: 不同方法的额外参数
            - regular: n=64
            - parametric: n=360
            - midpoint: 无
            - adaptive: eps=0.5

    返回:
        点列表 [(x,y), ... , (x0,y0)] 形成闭合圆
    """
    cx, cy = center

    # ------------------------------
    # 1. regular polygon (等角正多边形)
    # ------------------------------
    if method == "regular":
        n = kwargs.get("n", 64)
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            x = cx + r * math.cos(t)
            y = cy + r * math.sin(t)
            pts.append((x, y))
        pts.append(pts[0])
        return pts

    # ------------------------------
    # 2. parametric (等角参数采样)
    # ------------------------------
    if method == "parametric":
        n = kwargs.get("n", 360)
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
        pts.append(pts[0])
        return pts

    # ------------------------------
    # 3. Midpoint / Bresenham integer circle
    # ------------------------------
    if method == "midpoint":
        x = 0
        y = int(round(r))
        d = 1 - y
        pts = []

        def add_sym(px, py):
            for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                pts.append((cx + sx * px, cy + sy * py))
                pts.append((cx + sx * py, cy + sy * px))

        add_sym(x, y)
        while x < y:
            x += 1
            if d < 0:
                d += 2 * x + 1
            else:
                y -= 1
                d += 2 * (x - y) + 1
            add_sym(x, y)

        # 排序 & 闭合
        pts_unique = list({(int(round(px)), int(round(py))) for px, py in pts})
        pts_sorted = sorted(pts_unique, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        pts_sorted.append(pts_sorted[0])
        return pts_sorted

    # ------------------------------
    # 4. 自适应细分 (Adaptive Subdivision)
    # ------------------------------
    if method == "adaptive":
        eps = kwargs.get("eps", 0.5)

        def point(t):
            return (cx + r * math.cos(t), cy + r * math.sin(t))

        pts = []

        def subdiv(a, b):
            pa = point(a)
            pb = point(b)
            mid = (a + b) / 2
            pm = point(mid)

            # 计算中点到线段的距离
            ax, ay = pa;
            bx, by = pb;
            mx, my = pm
            dx, dy = bx - ax, by - ay
            if dx == dy == 0:
                dist = math.hypot(mx - ax, my - ay)
            else:
                t = ((mx - ax) * dx + (my - ay) * dy) / (dx * dx + dy * dy)
                t = max(0, min(1, t))
                proj = (ax + t * dx, ay + t * dy)
                dist = math.hypot(mx - proj[0], my - proj[1])

            if dist <= eps:
                pts.append(pa)
            else:
                subdiv(a, mid)
                subdiv(mid, b)

        subdiv(0, 2 * math.pi)
        pts.append(point(2 * math.pi))
        return pts

    raise ValueError(f"Unknown method: {method}")
import matplotlib.pyplot as plt

def visualize_pts_matplotlib(pts, figsize=(6,6)):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.figure(figsize=figsize)
    plt.plot(xs, ys)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()   # 若你使用图像坐标系 (y 向下)
    plt.show()




pts = generate_circle_points((500,500), 67, method="regular", n=64)
# pts = generate_circle_points((500,500), 300, method="midpoint")
# pts = generate_circle_points((500,500), 300, method="adaptive", eps=0.3)

visualize_pts_matplotlib(pts)