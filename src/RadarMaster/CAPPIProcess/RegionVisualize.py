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
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter


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

# Obsolete
def visualize_region_3d(arr, top_left, bottom_right):
    """
    使用 Plotly 3D 可视化 2D 数组的指定矩形区域。

    参数:
        arr: 2D numpy 数组，形状 (H, W)
        top_left:       (h1, w1) 左上角
        bottom_right:   (h2, w2) 右下角
    """

    h1, w1 = top_left
    h2, w2 = bottom_right

    # 裁剪区域
    region = arr[h1:h2, w1:w2]

    # 构建坐标网格
    H, W = region.shape
    x = np.arange(w1, w2)  # width
    y = np.arange(h1, h2)  # height
    X, Y = np.meshgrid(x, y)

    # 创建 3D surface
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=region,
            coloraxis=None
        )
    ])

    fig.update_layout(
        title="3D Region Visualization",
        scene=dict(
            xaxis_title="W",
            yaxis_title="H",
            zaxis_title="Value"
        ),
        width=900,
        height=700
    )

    fig.show()
def plotly_3d_small_patches(Z, patch_size=1, title="3D Patches"):
    """
    将二维数组 Z 的每个值绘制为一个独立的小平面（Quad Patch），彼此不连通。

    参数：
        Z : 2D numpy array (H, W)
        patch_size : 每个小平面的大小
    """

    H, W = Z.shape
    patches = []

    for y in range(H):
        for x in range(W):

            z = Z[y, x]
            s = patch_size / 2  # 以中心画方片

            # 小方片的四个顶点（平面）
            verts = np.array([
                [x - s, y - s, z],
                [x + s, y - s, z],
                [x + s, y + s, z],
                [x - s, y + s, z],
            ])

            patches.append(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    opacity=0.9,
                    color="royalblue",
                    flatshading=True
                )
            )

    fig = go.Figure(data=patches)
    fig.update_layout(
        title=title,
        width=900,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
            aspectmode="data"
        )
    )
    fig.show()
def plotly_3d_surface_two(
    Z1,
    Z2,
    smooth_sigma=1.0,
    color1="red",
    color2="blue",
    title="Two Surfaces",
    opacity1=0.9, opacity2=0.9
):
    """
    在同一张 3D 图中绘制两个平滑连接曲面，分别使用不同的单一颜色。

    参数：
        Z1, Z2 : 两个二维 numpy 数组，形状必须一致
        smooth_sigma : 平滑强度 (0 表示不平滑)
        color1 : 第一个曲面的单色
        color2 : 第二个曲面的单色
    """

    # --- 1. 检查尺寸 ---
    if Z1.shape != Z2.shape:
        raise ValueError("Z1 和 Z2 的形状必须一致！")

    # --- 2. 复制并处理 NaN/inf ---
    Z1p = np.nan_to_num(Z1.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    Z2p = np.nan_to_num(Z2.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    # --- 3. 可选平滑 ---
    if smooth_sigma > 0:
        Z1p = gaussian_filter(Z1p, sigma=smooth_sigma)
        Z2p = gaussian_filter(Z2p, sigma=smooth_sigma)

    # --- 4. 生成网格坐标 ---
    H, W = Z1.shape
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # --- 5. 创建绘图 ---
    fig = go.Figure()

    # --- Surface 1 ---
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=Z1p,
            colorscale=[[0, color1], [1, color1]],  # 单色
            showscale=False,
            opacity=opacity1,
            name="Surface 1"
        )
    )

    # --- Surface 2 ---
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=Z2p,
            colorscale=[[0, color2], [1, color2]],  # 单色
            showscale=False,
            opacity=opacity2,
            name="Surface 2"
        )
    )

    # --- Layout ---
    fig.update_layout(
        title=title,
        width=1000,
        height=900,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
            aspectmode="data"
        )
    )

    fig.show()


# Final
def plotly_3d_surface(Z, smooth_sigma=1.0, title="Smoothed 3D Surface"):
    """
    将二维数组平滑处理后，渲染为平滑连续的 3D 曲面。

    参数：
        Z : 2D numpy 数组
        smooth_sigma : 高斯平滑强度 (0 = 不平滑)
    """

    # ---- 1. 保留原始数据副本 ----
    Z_proc = Z.astype(float).copy()

    # ---- 2. 替换 NaN / Inf 防止 Surface 断裂 ----
    Z_proc = np.nan_to_num(Z_proc, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 3. 高斯平滑（关键步骤，可调）----
    if smooth_sigma > 0:
        Z_proc = gaussian_filter(Z_proc, sigma=smooth_sigma)

    # ---- 4. 生成坐标网格 ----
    H, W = Z_proc.shape
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # ---- 5. Plotly Surface 渲染 ----
    fig = go.Figure(
        data=[
            go.Surface(
                x=xx,
                y=yy,
                z=Z_proc,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )

    fig.update_layout(
        title=title,
        width=900,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
            aspectmode="data"
        )
    )

    fig.show()
def plotly_dual_surface(
    Z1, Z2,
    sigma1=1.0, sigma2=1.0,
    colorscale1="Viridis", colorscale2="Turbo",
    opacity1=0.9, opacity2=0.9,
    z_offset2=0,  # 第二个数组整体向上偏移
    title="Dual 3D Surfaces"
):
    """
    在同一张图中绘制两个二维数组的平滑 3D 表面，使用不同色标区分。

    参数：
        Z1, Z2       : 输入的两个二维数组，形状必须相同
        sigma1/2     : 高斯平滑强度 (0 = 不平滑)
        colorscale1/2: 两个表面的色标
        opacity1/2   : 透明度
        z_offset2    : 第二个曲面整体向上抬高的高度（避免遮挡）
    """

    assert Z1.shape == Z2.shape, "两个数组的形状必须一致！"

    # ---------- 处理 Z1 ----------
    Z1p = np.nan_to_num(Z1.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if sigma1 > 0:
        Z1p = gaussian_filter(Z1p, sigma=sigma1)

    # ---------- 处理 Z2 ----------
    Z2p = np.nan_to_num(Z2.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if sigma2 > 0:
        Z2p = gaussian_filter(Z2p, sigma=sigma2)

    # 给第二个表面加高度偏移，不会遮挡第一层
    Z2p = Z2p + z_offset2

    # ---------- 坐标网格 ----------
    H, W = Z1.shape
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # ---------- 绘图 ----------
    fig = go.Figure()

    # 第一层曲面
    fig.add_trace(
        go.Surface(
            x=xx, y=yy, z=Z1p,
            colorscale=colorscale1,
            opacity=opacity1,
            name="Surface 1",
            showscale=True
        )
    )

    # 第二层曲面
    fig.add_trace(
        go.Surface(
            x=xx, y=yy, z=Z2p,
            colorscale=colorscale2,
            opacity=opacity2,
            name="Surface 2",
            showscale=True
        )
    )

    # 布局设置
    fig.update_layout(
        title=title,
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
            aspectmode="data"
        )
    )

    fig.show()

if __name__ == '__main__':

    singles = np.load(r"E:\MyFiles\data\CAPPI0408_images_single.npy")
    single_differences = np.load(r"E:\MyFiles\data\CAPPI0408_images_single_difference.npy")
    grays = np.load(r"E:\MyFiles\data\CAPPI0408_images_gray.npy")
    gray_differences = np.load(r"E:\MyFiles\data\CAPPI0408_images_gray_difference.npy")


    # visualize_region_3d(
    #     arr=arrs[146],
    #     top_left=(634, 483),
    #     bottom_right=(641, 490),
    # )

    # plotly_3d_small_patches(
    #     Z=arrs[146,634:641,483:490]
    # )
    # plotly_3d_surface_smooth(
    #     Z=arrs[146,634:684,483:533],
    #     smooth_sigma=1
    # )
    # plot_dual_surface_with_wireframe(
    #     Z1=arrs[146,634:684,483:533],
    #     Z2=arrs[147,634:684,483:533],
    # )
    # plotly_3d_surface_two(
    #     Z1=arrs[146,634:684,483:533],
    #     Z2=arrs[147,634:684,483:533],
    #     opacity1=1, opacity2=1,
    # )

    # Blues & Reds
    # Greens & Purples
    plotly_dual_surface(
        Z1=singles[146, 634:684, 483:533],
        Z2=singles[147, 634:684, 483:533],
        colorscale1="Blues",
        colorscale2="Reds",
        opacity1=0.5, opacity2=1,
    )


    # visualize_region_grid(
    #     npy_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
    #     top_left=(634, 483),
    #     bottom_right=(641,490),
    #     frame_range=(110, 130),    # [10, 22)
    #     grid_shape=(4, 5),        # 3 行 4 列
    #     mode="blend",
    #     fontsize=9
    # )
    #
    # visualize_region_grid(
    #     npy_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
    #     top_left=(634, 483),
    #     bottom_right=(641,490),
    #     frame_range=(130, 150),    # [10, 22)
    #     grid_shape=(4, 5),        # 3 行 4 列
    #     mode="blend",
    #     fontsize=9
    # )
    #
    # visualize_region_grid(
    #     npy_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
    #     top_left=(634, 483),
    #     bottom_right=(641,490),
    #     frame_range=(136, 156),    # [10, 22)
    #     grid_shape=(4, 5),        # 3 行 4 列
    #     mode="blend",
    #     fontsize=9
    # )


