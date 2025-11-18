#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RegionVisualize.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/14 14:55 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def lighten_cmap(cmap_name="jet", factor=0.5):
    """
    让 colormap 更浅一些。
    factor 越大越接近白色
    """
    base = plt.get_cmap(cmap_name)
    new_cmap = base(np.linspace(0, 1, 256))
    new_cmap = factor + (1 - factor) * new_cmap  # 向白色偏移
    return cm.colors.LinearSegmentedColormap.from_list("light_" + cmap_name, new_cmap)


def visualize_region_grid(
        npy_path,
        top_left,
        bottom_right,
        frame_range,       # (start, end)
        grid_shape,        # (rows, cols)
        mode="image",      # "image" | "number" | "blend"
        fontsize=7
    ):
    """
    可视化矩形区域并以 a*b 格式排列。
    -> vmin/vmax 自动基于当前截取帧的区域值计算
    -> 全局共享色条，并且不会遮挡图像
    """

    data = np.load(npy_path)  # (N, H, W)
    N, H, W = data.shape

    y1, x1 = top_left
    y2, x2 = bottom_right
    start, end = frame_range

    frames = range(start, min(end, N))
    num_frames = len(frames)

    rows, cols = grid_shape
    if rows * cols < num_frames:
        raise ValueError(f"a*b={rows*cols} 不够放下 {num_frames} 张图。")

    # ==================================================
    # 1. 基于所有截取帧的 region 数据计算 vmin/vmax
    # ==================================================
    region_values = []
    for frame_idx in frames:
        region = data[frame_idx, y1:y2, x1:x2]
        region_values.append(region)

    region_values = np.array(region_values)
    vmin = np.nanmin(region_values)
    vmax = np.nanmax(region_values)

    # ==================================================
    # 2. 创建大画布（关闭自动布局避免遮盖）
    # ==================================================
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(rows, cols)

    light_cmap = lighten_cmap("jet", factor=0.55)
    first_im = None

    # ==================================================
    # 3. 绘制所有子图
    # ==================================================
    for idx, frame_idx in enumerate(frames):
        ax = axes[idx // cols, idx % cols]
        region = data[frame_idx, y1:y2, x1:x2]

        if mode == "image":
            im = ax.imshow(region, cmap="jet", vmin=vmin, vmax=vmax)
            if first_im is None:
                first_im = im
            ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

        elif mode == "number":
            ax.imshow(region, alpha=0.0)  # 保持坐标
            ax.set_title(f"Frame {frame_idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            h, w = region.shape
            for i in range(h):
                for j in range(w):
                    v = region[i, j]
                    txt = "NaN" if np.isnan(v) else f"{int(v)}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize)

        elif mode == "blend":
            im = ax.imshow(region, cmap=light_cmap, vmin=vmin, vmax=vmax)
            if first_im is None:
                first_im = im
            ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

            h, w = region.shape
            for i in range(h):
                for j in range(w):
                    v = region[i, j]
                    txt = "NaN" if np.isnan(v) else f"{int(v)}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=fontsize, color="black")
        else:
            raise ValueError("mode 必须是 image/number/blend")

    # 隐藏多余子图
    for idx in range(num_frames, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    # ==================================================
    # 4. 添加共享色条（不会覆盖图像）
    # ==================================================
    if mode in ["image", "blend"]:
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(first_im, cbar_ax)

    plt.subplots_adjust(right=0.88)  # 给色条留空间
    plt.show()


visualize_region_grid(
    npy_path=r"E:\MyFiles\data\CAPPI0408_images_gray.npy",
    top_left=(634, 483),
    bottom_right=(641,490),
    frame_range=(110, 130),    # [10, 22)
    grid_shape=(4, 5),        # 3 行 4 列
    mode="blend",
    fontsize=9
)
