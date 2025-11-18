#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VisualizeNPY.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/14 15:54 
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO


def show_npy_frame(npy_path, frame_index=0, cmap='jet'):
    """
    可视化 .npy 文件中的某一帧图像
    npy_path: npy 文件路径
    frame_index: 要展示的帧编号
    cmap: 颜色映射（默认 jet)
    """
    data = np.load(npy_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(data[frame_index], cmap=cmap)
    plt.title(f"Frame {frame_index}")
    plt.colorbar()
    plt.show()

def show_npy_frame_ignore_percent(npy_path, frame_index=0, ignore_percent=1, cmap='jet'):
    """
    可视化 .npy 文件中的某一帧图像。
    自动忽略上下 ignore_percent% 的值（默认 1%），使色彩更易观察。

    参数:
        npy_path: npy 文件路径
        frame_index: 要显示的帧编号
        ignore_percent: 忽略的百分比（上下各 ignore_percent）
        cmap: 色图
    """
    data = np.load(npy_path)
    frame = data[frame_index]

    # 忽略 NaN
    valid_values = frame[np.isfinite(frame)]

    # 计算上下1%分位数
    low = np.percentile(valid_values, ignore_percent)
    high = np.percentile(valid_values, 100 - ignore_percent)

    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap=cmap, vmin=low, vmax=high)
    plt.title(f"Frame {frame_index}  (ignore top/bottom {ignore_percent}%)")
    plt.colorbar()
    plt.show()



def npy_to_video_with_uniform_colorbar(
    npy_path,
    output_video_path,
    fps=10,
    ignore_percent=1,
    cmap='jet',
    dpi=100
):
    data = np.load(npy_path)  # shape = (T, H, W)
    frames, H, W = data.shape

    # === 计算全局色条范围（忽略 NaN） ===
    valid_values = data[np.isfinite(data)]
    low = np.percentile(valid_values, ignore_percent)
    high = np.percentile(valid_values, 100 - ignore_percent)
    print(f"统一色条范围: vmin={low:.3f}, vmax={high:.3f}")

    # === 自动调整画布尺寸，使生成图像像素可被16整除 ===
    # 初始 6x6 → pixel size = figsize * dpi
    fig_w, fig_h = 6, 6
    px_w = int(fig_w * dpi)
    px_h = int(fig_h * dpi)

    # 调整成16倍整数
    px_w_aligned = (px_w + 15) // 16 * 16
    px_h_aligned = (px_h + 15) // 16 * 16

    # 换算回 figsize
    fig_w = px_w_aligned / dpi
    fig_h = px_h_aligned / dpi

    print(f"自动调整图像尺寸：{px_w}×{px_h} → {px_w_aligned}×{px_h_aligned}")

    writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264')

    for i in range(frames):
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        plt.imshow(data[i], cmap=cmap, vmin=low, vmax=high)
        plt.title(f"Frame {i}")
        plt.colorbar()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)

        frame_img = imageio.v2.imread(buf)
        writer.append_data(frame_img)

        plt.close()

    writer.close()
    print(f"视频已保存到: {output_video_path}")


if __name__ == '__main__':

    for index in range(239):
        show_npy_frame_ignore_percent(
            npy_path=r"E:\MyFiles\data\CAPPI0408_images_origin_denoise0.npy",
            frame_index=index
        )

    # npy_to_video_with_uniform_colorbar(
    #     npy_path=r"E:\MyFiles\data\CAPPI0408_images_single.npy",
    #     output_video_path=r"E:\MyFiles\data\20250408_single.mp4",
    #     fps=10,
    #     ignore_percent=1,
    #     cmap="jet"
    # )
