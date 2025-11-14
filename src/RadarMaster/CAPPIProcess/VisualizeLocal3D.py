#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VisualizeLocal3D.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/13 20:36 
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 不可删

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 中文显示支持 ===
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def visualize_neighborhood_timeseries(data_path, center_x, center_y, size=3, frame_range=None, scale=0.02):
    """
    在三维空间中展示某坐标邻域区域的时序变化曲线。

    参数:
        data_path: str, .npy 文件路径 (形状为 T×H×W)
        center_x, center_y: int, 中心像素坐标
        size: int, 邻域尺寸（奇数），如3表示3×3区域，5表示5×5区域
        frame_range: (start, end)，可选，限定帧范围
        scale: float, 值的垂直缩放系数（控制线条高度）
    """
    # === 读取数据 ===
    data = np.load(data_path)
    T, H, W = data.shape

    # 帧范围
    if frame_range is None:
        frame_range = (0, T)
    start, end = frame_range
    data = data[start:end]
    frames = np.arange(start, end)

    # === 校验 size ===
    if size % 2 == 0 or size < 1:
        raise ValueError("size 必须是奇数，如 3、5、7 等。")

    half = size // 2

    # === 构造邻域坐标 ===
    offsets = [(dx, dy) for dx in range(-half, half + 1) for dy in range(-half, half + 1)]

    # === 绘图 ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for dx, dy in offsets:
        x = center_x + dx
        y = center_y + dy
        if 0 <= x < W and 0 <= y < H:
            values = data[:, y, x]

            # 三维坐标数据
            X = np.ones_like(frames) * dx
            Y = frames
            Z = np.ones_like(frames) * dy + values * scale  # 值映射到高度

            ax.plot(X, Y, Z, label=f'({y},{x})')

    # === 坐标轴 & 样式 ===
    ax.set_xlabel('ΔX (相对中心)')
    ax.set_ylabel('帧索引')
    ax.set_zlabel('ΔY + 值变化')
    ax.set_title(f"{size}×{size} 区域的时序变化（中心点: ({center_y}, {center_x})）")

    plt.tight_layout()
    plt.show()

import numpy as np
import plotly.graph_objects as go

def visualize_neighborhood_timeseries_interactive(data_path, center_x, center_y, size=3, frame_range=None, scale=0.02):
    data = np.load(data_path)
    T, H, W = data.shape

    if frame_range is None:
        frame_range = (0, T)
    start, end = frame_range
    data = data[start:end]
    frames = np.arange(start, end)

    if size % 2 == 0 or size < 1:
        raise ValueError("size 必须是奇数")

    half = size // 2
    offsets = [(dx, dy) for dx in range(-half, half + 1) for dy in range(-half, half + 1)]

    fig = go.Figure()

    for dx, dy in offsets:
        x = center_x + dx
        y = center_y + dy
        if 0 <= x < W and 0 <= y < H:
            values = data[:, y, x]
            X = np.ones_like(frames) * dx
            Y = frames
            Z = np.ones_like(frames) * dy + values * scale

            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='lines',
                name=f'({y},{x})',
                line=dict(width=3)
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="ΔX (相对中心)",
            yaxis_title="帧索引",
            zaxis_title="ΔY + 值变化"
        ),
        title=f"{size}×{size} 区域的时序变化 (中心点: ({center_y}, {center_x}))",
        width=900, height=700
    )

    fig.show()


# visualize_neighborhood_timeseries(
#     data_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
#     center_x=522,
#     center_y=176,
# )
if __name__ == '__main__':
    visualize_neighborhood_timeseries_interactive(
        data_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        center_y=316,
        center_x=246
    )