#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VisualizeNegative.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/17 11:13 
"""

import numpy as np
import plotly.graph_objects as go

def visualize_negative_count_3d(
    npy_path,
    freq_min=None,
    freq_max=None
):
    """
    使用 Plotly 绘制 npy 文件中每个像素负数出现次数的 3D Surface 图。
    可选：freq_min, freq_max 指定展示频率范围，仅显示满足的区域。
    """

    # === 加载数据 ===
    data = np.load(npy_path)   # shape = (T, H, W)
    T, H, W = data.shape

    # === 统计负数次数 ===
    negative_count = (data < 0).sum(axis=0)  # int array

    # !!! 必须转成 float 才能写入 NaN !!!
    filtered_count = negative_count.astype(float)

    # === 如果指定了频率区间，则过滤 ===
    if freq_min is not None:
        filtered_count[filtered_count < freq_min] = np.nan
    if freq_max is not None:
        filtered_count[filtered_count > freq_max] = np.nan

    # === 构建坐标网格 ===
    x = np.arange(W)
    y = np.arange(H)

    # === Plotly Surface 图 ===
    fig = go.Figure(data=[
        go.Surface(
            z=filtered_count,
            x=x,
            y=y,
            colorscale='Jet',
            colorbar=dict(title="Negative Count"),
            showscale=True
        )
    ])

    # === 图像布局 ===
    title = "3D Negative Count per Pixel"
    if freq_min is not None or freq_max is not None:
        title += f" (Filtered: {freq_min} ≤ freq ≤ {freq_max})"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Width)',
            yaxis_title='Y (Height)',
            zaxis_title='Negative Count'
        ),
        width=900,
        height=700
    )

    fig.show()

    return negative_count, filtered_count




if __name__ == '__main__':
    neg_count = visualize_negative_count_3d(
        r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        freq_min=0,
        freq_max=230
    )
