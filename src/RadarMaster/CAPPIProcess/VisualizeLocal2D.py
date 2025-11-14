#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VisualizeLocal2D.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/13 20:49 
"""
import numpy as np
import matplotlib.pyplot as plt


# === 中文显示支持 ===
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_neighborhood_timeseries_grid(data_path, center_x, center_y, size=3, frame_range=None):
    """
    用二维网格方式可视化邻域区域的时序变化。

    参数:
        data_path: str, .npy 文件路径 (形状为 T×H×W)
        center_x, center_y: int, 中心像素坐标
        size: int, 邻域大小（奇数）
        frame_range: (start, end)，帧范围
    """
    data = np.load(data_path)
    T, H, W = data.shape

    if frame_range is None:
        frame_range = (0, T)
    start, end = frame_range
    data = data[start:end]
    frames = np.arange(start, end)

    if size % 2 == 0 or size < 1:
        raise ValueError("size 必须为奇数，例如 3、5、7。")

    half = size // 2

    # === 创建网格子图 ===
    fig, axes = plt.subplots(size, size, figsize=(size * 2.2, size * 2.2), sharex=True, sharey=True)
    fig.suptitle(f"{size}×{size} 区域的时序变化（中心点: ({center_y}, {center_x})）", fontsize=14)

    for i, dy in enumerate(range(-half, half + 1)):
        for j, dx in enumerate(range(-half, half + 1)):
            ax = axes[i, j]
            x = center_x + dx
            y = center_y + dy

            if 0 <= x < W and 0 <= y < H:
                values = data[:, y, x]
                ax.plot(frames, values, color='steelblue', linewidth=1)
                ax.set_title(f"({y},{x})", fontsize=8)
            else:
                ax.set_visible(False)  # 超出边界则隐藏

            if i == size - 1:
                ax.set_xlabel("帧索引", fontsize=8)
            if j == 0:
                ax.set_ylabel("值", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 中文支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_neighborhood_timeseries_combined(data_path, center_x, center_y, size=3, frame_range=None, cmap='viridis'):
    """
    将邻域区域内所有像素的时序曲线绘制在同一坐标系中。

    参数:
        data_path: str, .npy 文件路径 (形状为 T×H×W)
        center_x, center_y: int, 中心像素坐标
        size: int, 邻域大小（奇数）
        frame_range: (start, end)，可选，限制帧范围
        cmap: str, matplotlib 内置颜色映射（如 'viridis', 'plasma', 'rainbow'）
    """
    data = np.load(data_path)
    T, H, W = data.shape

    if frame_range is None:
        frame_range = (0, T)
    start, end = frame_range
    data = data[start:end]
    frames = np.arange(start, end)

    if size % 2 == 0 or size < 1:
        raise ValueError("size 必须为奇数，例如 3、5、7。")

    half = size // 2

    # 构造邻域坐标
    coords = [(center_x + dx, center_y + dy)
              for dy in range(-half, half + 1)
              for dx in range(-half, half + 1)
              if 0 <= center_x + dx < W and 0 <= center_y + dy < H]

    # 颜色映射
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i / len(coords)) for i in range(len(coords))]

    # === 绘图 ===
    plt.figure(figsize=(8, 5))
    for i, (x, y) in enumerate(coords):
        values = data[:, y, x]
        plt.plot(frames, values, color=colors[i], linewidth=1.5, label=f'({y},{x})')

    plt.title(f"{size}×{size} 区域的时序曲线对比\n中心点: ({center_y}, {center_x})")
    plt.xlabel("帧索引")
    plt.ylabel("值")
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data_path = r"E:\MyFiles\data\CAPPI0408_images_single.npy"

    y,x = 268,517

    size = 3

    visualize_neighborhood_timeseries_grid(
        data_path=data_path,
        center_y=y,
        center_x=x,
        size=size,
        frame_range=[2,40]
    )

    # visualize_neighborhood_timeseries_combined(
    #     data_path=data_path,
    #     center_y=y,
    #     center_x=x,
    #     size=size
    # )
    #
    # from VisualizeLocal3D import visualize_neighborhood_timeseries_interactive
    # visualize_neighborhood_timeseries_interactive(
    #     data_path=data_path,
    #     center_y=y,
    #     center_x=x,
    #     size=size
    # )