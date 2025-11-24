#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RegionHistogram.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/14 11:37 
"""
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def show_region_histograms(
    npy_path,
    top_left,
    bottom_right,
    frame_index=None,
    bins=50,
    save_folder=None
):
    """
    查看 (N, H, W) 雷达数据某区域内数值的直方图，并保存到指定文件夹。

    新增:
        save_folder : 指定保存直方图的文件夹（文件名为 1.png, 2.png, ...）
    """
    data = np.load(npy_path)  # (N, H, W)
    N, H, W = data.shape

    y1, x1 = top_left
    y2, x2 = bottom_right

    # Create folder
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    # 区域提取（忽略 NaN）
    def extract_region(frame):
        region = frame[y1:y2, x1:x2]
        return region[~np.isnan(region)]

    # ======================================================
    #                   单帧模式
    # ======================================================
    if frame_index is not None:
        values = extract_region(data[frame_index])

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=bins)
        plt.title(f"Frame {frame_index}  Region {top_left} → {bottom_right}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        if save_folder:
            save_path = os.path.join(save_folder, "1.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"已保存: {save_path}")
        else:
            plt.show()

        return

    # ======================================================
    #                   多帧模式
    # ======================================================
    for i in range(N):
        values = extract_region(data[i])

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=bins)
        plt.title(f"Frame {i}  Region {top_left} → {bottom_right}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        if save_folder:
            save_path = os.path.join(save_folder, f"{i+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"已保存: {save_path}")
        else:
            plt.show()


def show_region_histograms_3d(
    npy_path,
    top_left,
    bottom_right,
    bins=50,
    start_frame=0,
    end_frame=None   # 不包含 end_frame
):
    """
    使用 plotly 将某矩形区域的指定帧区间的直方图可视化为 3D 网格曲面。

    参数:
        npy_path: str            - npy 文件路径 (T,H,W)
        top_left: (h1,w1)        - 区域左上角
        bottom_right: (h2,w2)    - 区域右下角
        bins: int                - 直方图柱数
        start_frame: int         - 起始帧 (包含)
        end_frame: int or None   - 结束帧 (不包含)，None 表示直到最后一帧
    """

    data = np.load(npy_path)  # (T,H,W)
    T = data.shape[0]

    # 默认 end_frame 为 T
    if end_frame is None:
        end_frame = T
    if end_frame > T:
        end_frame = T

    # 截取帧区间
    data = data[start_frame:end_frame]
    T_sel = data.shape[0]

    h1, w1 = top_left
    h2, w2 = bottom_right

    # 先计算全局 min/max 用于统一 bins
    subregion = data[:, h1:h2, w1:w2]
    global_min = np.nanmin(subregion)
    global_max = np.nanmax(subregion)

    # bin edges
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 计算每一帧直方图
    all_hist = []
    for t in range(T_sel):
        vals = data[t, h1:h2, w1:w2]
        vals = vals[np.isfinite(vals)]

        hist, _ = np.histogram(vals, bins=bin_edges)
        all_hist.append(hist)

    all_hist = np.array(all_hist)  # shape (T_sel, bins)

    # 3D surface 绘制
    fig = go.Figure(data=[
        go.Surface(
            z=all_hist,
            x=bin_centers,               # 数值区间
            y=np.arange(start_frame, end_frame),  # 帧号
            colorscale="Viridis"
        )
    ])

    fig.update_layout(
        title=f"3D Histogram Surface (Frames {start_frame}–{end_frame})",
        scene=dict(
            xaxis_title="Value",
            yaxis_title="Frame Index",
            zaxis_title="Count"
        ),
        width=900,
        height=700
    )

    fig.show()

# def plot_3d_hist_surface(
#     npy_path,
#     top_left,
#     bottom_right,
#     bins=50,
#     start_frame=0,
#     end_frame=None
# ):
#     """
#     使用 Plotly 将 (start_frame, end_frame) 范围内的各帧直方图作为连续曲线，
#     并在帧轴方向连接这些曲线，生成 3D 曲面（连续折面）。
#     """
#
#     data = np.load(npy_path)
#     T = data.shape[0]
#
#     if end_frame is None:
#         end_frame = T
#
#     # 取选择的帧
#     data = data[start_frame:end_frame]
#     T_sel = data.shape[0]
#
#     h1, w1 = top_left
#     h2, w2 = bottom_right
#
#     # 统一 bins边界
#     subregion = data[:, h1:h2, w1:w2]
#     global_min = np.nanmin(subregion)
#     global_max = np.nanmax(subregion)
#
#     bin_edges = np.linspace(global_min, global_max, bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     # 每一帧计算直方曲线
#     hist_matrix = []
#     for t in range(T_sel):
#         vals = data[t, h1:h2, w1:w2]
#         vals = vals[np.isfinite(vals)]
#
#         hist, _ = np.histogram(vals, bins=bin_edges)
#         hist_matrix.append(hist)
#
#     hist_matrix = np.array(hist_matrix)  # shape (T_sel, bins)
#
#     # X, Y 网格
#     X, Y = np.meshgrid(bin_centers, np.arange(start_frame, end_frame))
#
#     # Z 为直方高度
#     Z = hist_matrix
#
#     # 绘制连续曲面（折面由帧之间的曲线连接而成）
#     fig = go.Figure(data=[
#         go.Surface(
#             x=X,
#             y=Y,
#             z=Z,
#             colorscale="Viridis",
#             showscale=True
#         )
#     ])
#
#     fig.update_layout(
#         title="Continuous 3D Histogram Surface",
#         scene=dict(
#             xaxis_title="Value (bin center)",
#             yaxis_title="Frame Index",
#             zaxis_title="Count"
#         ),
#         width=900,
#         height=700
#     )
#
#     fig.show()

# def plot_3d_hist_surface(
#     npy_path,
#     top_left,
#     bottom_right,
#     start_frame=0,
#     end_frame=None,
#     max_bins=200       # 限制最大 bins 数，避免 unique 过多
# ):
#     """
#     使用 Plotly 将帧区间内的直方图曲线连接成 3D 曲面。
#     bins 数量根据区域内不同值的数量自动确定。
#     """
#
#     data = np.load(npy_path)
#     T = data.shape[0]
#
#     if end_frame is None:
#         end_frame = T
#
#     # 取选择帧
#     data = data[start_frame:end_frame]
#     T_sel = data.shape[0]
#
#     h1, w1 = top_left
#     h2, w2 = bottom_right
#
#     # 获取子区域全部值
#     subregion = data[:, h1:h2, w1:w2]
#     flat_vals = subregion[np.isfinite(subregion)]
#
#     # 自动确定 bins 数量（unique 数量）
#     unique_vals = np.unique(flat_vals)
#     bins = min(len(unique_vals), max_bins)
#
#     # 设置统一 bins 边界
#     global_min = flat_vals.min()
#     global_max = flat_vals.max()
#
#     bin_edges = np.linspace(global_min, global_max, bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     # 每帧计算直方图
#     hist_matrix = []
#     for t in range(T_sel):
#         vals = data[t, h1:h2, w1:w2]
#         vals = vals[np.isfinite(vals)]
#
#         hist, _ = np.histogram(vals, bins=bin_edges)
#         hist_matrix.append(hist)
#
#     hist_matrix = np.array(hist_matrix)   # (T_sel, bins)
#
#     # 构网格
#     X, Y = np.meshgrid(bin_centers, np.arange(start_frame, end_frame))
#     Z = hist_matrix
#
#     # 绘制连续 3D Surface
#     fig = go.Figure(data=[
#         go.Surface(
#             x=X, y=Y, z=Z,
#             colorscale="Viridis",
#             showscale=True
#         )
#     ])
#
#     fig.update_layout(
#         title=f"Continuous 3D Histogram Surface (bins={bins})",
#         scene=dict(
#             xaxis_title="Value (bin center)",
#             yaxis_title="Frame Index",
#             zaxis_title="Count"
#         ),
#         width=900,
#         height=700
#     )
#
#     fig.show()

import numpy as np
import plotly.graph_objects as go

def plot_3d_hist_surface(
    npy_path,
    top_left,
    bottom_right,
    start_frame=0,
    end_frame=None,
    center_min=-60,   # bin 中心最小值
    center_max=60,    # bin 中心最大值
    step=5            # bin 中心的步长
):
    """
    3D Surface 直方曲面，其中 bin 中心严格分布在 [center_min, center_max]。
    """

    data = np.load(npy_path)
    T = data.shape[0]

    if end_frame is None:
        end_frame = T

    # 取选择的帧
    data = data[start_frame:end_frame]
    T_sel = data.shape[0]

    h1, w1 = top_left
    h2, w2 = bottom_right

    # -----------------------------
    # 1) 固定 bin 中心
    # -----------------------------
    bin_centers = np.arange(center_min, center_max + step, step)

    # bin 宽度
    bin_width = step

    # 由中心反推 edges
    bin_edges = np.arange(
        center_min - bin_width/2,
        center_max + bin_width/2 + 1e-9,
        bin_width
    )

    bins = len(bin_centers)

    # -----------------------------
    # 2) 每帧计算直方图
    # -----------------------------
    hist_matrix = []
    for t in range(T_sel):
        vals = data[t, h1:h2, w1:w2]
        vals = vals[np.isfinite(vals)]

        hist, _ = np.histogram(vals, bins=bin_edges)
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix)  # [T_sel, bins]

    # -----------------------------
    # 3) 构建 Surface 网格
    # -----------------------------
    X, Y = np.meshgrid(bin_centers, np.arange(start_frame, end_frame))
    Z = hist_matrix

    # -----------------------------
    # 4) 绘制 Plotly 曲面
    # -----------------------------
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="jet",
            showscale=True
        )
    ])

    fig.update_layout(
        title=f"3D Histogram Surface (Centers from {center_min} to {center_max})",
        scene=dict(
            xaxis_title="Value (bin center)",
            yaxis_title="Frame Index",
            zaxis_title="Count"
        ),
        width=900,
        height=700
    )

    fig.show()

def calculate_hist_matrix(
    npy_path,
    top_left,
    bottom_right,
    bins=50,
    start_frame=0,
    end_frame=None
):
    """
    计算指定帧范围内的每帧直方图，并返回一个二维数组，
    其中每一行表示一个帧的直方图（形状为 [T_sel, bins]）。

    参数：
        npy_path: 输入的 npy 文件路径
        top_left: 左上角坐标 (h1, w1)
        bottom_right: 右下角坐标 (h2, w2)
        bins: 直方图的箱数
        start_frame: 起始帧索引
        end_frame: 结束帧索引，如果为 None，则使用数据的最后一帧

    返回：
        hist_matrix: 直方图矩阵，形状为 [T_sel, bins]，每一行是一个帧的直方图
    """
    # 读取数据
    data = np.load(npy_path)
    T = data.shape[0]

    if end_frame is None:
        end_frame = T

    # 取选择的帧
    data = data[start_frame:end_frame]
    T_sel = data.shape[0]

    h1, w1 = top_left
    h2, w2 = bottom_right

    # 统一 bins 边界
    subregion = data[:, h1:h2, w1:w2]
    global_min = np.nanmin(subregion)
    global_max = np.nanmax(subregion)

    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 每一帧计算直方图
    hist_matrix = []
    for t in range(T_sel):
        vals = data[t, h1:h2, w1:w2]
        vals = vals[np.isfinite(vals)]  # 过滤 NaN 值

        hist, _ = np.histogram(vals, bins=bin_edges)
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix)  # 形状为 (T_sel, bins)

    return hist_matrix
def plot_2d_array_surface(data, title="2D Array Surface", surface_type="surface"):
    """
    使用 Plotly 绘制二维数组数据作为曲面图或折面图。

    参数：
        data: 输入的二维数组（形状为 [m, n]）
        title: 图表标题
        surface_type: 选择图表类型，'surface' 为曲面图，'mesh' 为折面图

    返回：
        fig: Plotly 图形对象
    """
    # 获取数据的形状
    m, n = data.shape

    # 创建 x 和 y 网格
    x = np.arange(n)
    y = np.arange(m)

    # 创建网格坐标
    X, Y = np.meshgrid(x, y)

    # 根据选择的图表类型绘制
    if surface_type == "surface":
        fig = go.Figure(data=[go.Surface(
            z=data,
            x=X,
            y=Y,
            colorscale="Viridis",  # 可以根据需要选择不同的颜色
            colorbar=dict(title="Value")  # 添加颜色条
        )])
    elif surface_type == "mesh":
        fig = go.Figure(data=[go.Mesh3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=data.flatten(),
            color=data.flatten(),
            opacity=0.5,
            colorscale="Viridis"
        )])
    else:
        raise ValueError("Invalid surface_type. Choose either 'surface' or 'mesh'.")

    # 更新布局
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Value"
        ),
        width=900,
        height=700
    )

    # 显示图形
    fig.show()

    return fig



if __name__ == "__main__":
    show_region_histograms(
        r"E:\MyFiles\data\CAPPI0408_images_single.npy",
        top_left=(634, 483),
        bottom_right=(672, 557),
        bins=60,
        save_folder=r"E:\MyFiles\data\histogram_single"
    )



    # plot_3d_hist_surface(
    #     r"E:\MyFiles\data\CAPPI0408_images_single.npy",
    #     top_left=(634, 483),
    #     bottom_right=(672, 557),
    #     # bins=60,
    #     start_frame=0,
    #     end_frame=None,
    #     center_min=-30,   # bin 中心最小值
    #     center_max=50,    # bin 中心最大值
    #     step=5            # bin 中心的步长
    # )


    # data = np.load(r"E:\MyFiles\data\CAPPI0408_images_single_difference.npy")
    # print(data[139][550])
    #
    # histograms = calculate_hist_matrix(
    #     r"E:\MyFiles\data\CAPPI0408_images_single_difference.npy",
    #     top_left=(634, 483),
    #     bottom_right=(672, 557),
    # )
    #
    # print("每一帧的直方图结果：")
    # print(histograms[0])

    # hist_matrix = calculate_hist_matrix(
    #     r"E:\MyFiles\data\CAPPI0408_images_single_difference.npy",
    #     top_left=(634, 483),
    #     bottom_right=(672, 557),
    #     bins=50,
    #     start_frame=110,
    #     end_frame=180
    # )
    #
    # print(hist_matrix.shape)
    #
    # # 绘制曲面图
    # plot_2d_array_surface(hist_matrix, title="Random 2D Surface", surface_type="surface")
    #
    # # 绘制折面图
    # plot_2d_array_surface(hist_matrix, title="Random 2D Mesh", surface_type="mesh")