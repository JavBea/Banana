#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：CAPPI_Denoise.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/10/1 9:48 
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2

from matplotlib import animation


def build_noise_template(data, method="median", q=0.8):
    """
    构造噪声模板

    :param data: 3D numpy array, 形状 (M, 512, N)，M 帧雷达图
    :param method: "mean" 或 "median"
    :param q: 分位数
    :return: 2D numpy array, 噪声模板

    """
    if method == "quantile":
        template = np.nanquantile(data, q, axis=0)
    elif method == "median":
        template = np.nanmedian(data, axis=0)
    elif method == "mean":
        template = np.nanmean(data, axis=0)
    return template


def denoise_frame(frame, template, alpha=1.0, mode="subtract"):
    """
    利用噪声模板对单帧去噪

    :param frame: 2D numpy array, 单帧雷达数据
    :param template: 2D numpy array, 噪声模板
    :param alpha: 模板权重，仅在 mode="subtract" 时使用
    :param mode: "subtract" 或 "mask"
                  - "subtract": 原有策略，frame - alpha*template，再截断为非负
                  - "mask": 新策略，小于模板的置 0，大于等于模板的保留原值
    :return: 去噪后的图像
    """
    if mode == "subtract":
        denoised = frame - alpha * template
        denoised[denoised < 0] = 0
    elif mode == "mask":
        denoised = np.where(frame >= template, frame, 0)
    else:
        raise ValueError("mode must be 'subtract' or 'mask'")


    return denoised


def apply_morphological_close(data, kernel_size=3):
    """
    对一个三维 numpy 数组（帧序列）进行闭操作
    :param data: numpy array, 形状为 (帧数, 高, 宽)
    :param kernel_size: int, 闭操作的结构元素大小，越大平滑/去噪效果越明显
    :return: numpy array, 闭操作后的数组，形状同原始数组
    """
    # 创建一个矩形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 初始化输出数组
    result = np.zeros_like(data)

    # 对每一帧进行闭操作
    for i in range(data.shape[0]):
        # 注意确保输入为uint8类型，如果原始数据不是，可以根据需要归一化/转换
        frame = data[i].astype(np.uint8)
        closed_frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        result[i] = closed_frame

    return result

import numpy as np
from sklearn.decomposition import TruncatedSVD

def temporal_pca_denoise(data, n_components=5, mode='subtract'):
    """
    data: ndarray, shape = (T, H, W)  (与你的 data 一致)
    n_components: 保留的主成分个数（k）
    mode: 'subtract' -> frame - background; 'reconstruct' -> use reconstructed frames
    返回: denoised ndarray shape (T, H, W)
    """
    T, H, W = data.shape
    X = data.reshape(T, H*W).astype(float)  # shape (T, HW)

    # 去均值（对时间序列去均值有助SVD）
    mean_time = X.mean(axis=0, keepdims=True)
    Xc = X - mean_time

    # Truncated SVD 在样本数 T 比较小或 HW 比较大时效果好
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    U = svd.fit_transform(Xc)                # shape (T, k)
    Vt = svd.components_                     # shape (k, HW)

    # 重构低秩近似（背景随时间的低秩变化）
    X_lowrank = (U @ Vt) + mean_time        # shape (T, HW)
    X_lowrank = X_lowrank.reshape(T, H, W)

    if mode == 'reconstruct':
        return X_lowrank
    elif mode == 'subtract':
        denoised = data - X_lowrank
        denoised[denoised < 0] = 0
        return denoised
    else:
        raise ValueError("mode must be 'subtract' or 'reconstruct'")

def save_animation(data, out_file, fps=10, cmap="jet", vmin=None, vmax=None):
    """
    保存雷达图序列为 gif 或 mp4 动画

    :param data: 3D numpy array, shape = (num_frames, H, W)
    :param out_file: 输出文件路径 (.gif 或 .mp4)
    :param fps: 帧率
    :param cmap: 颜色映射
    :param vmin: 颜色最小值 (默认自动)
    :param vmax: 颜色最大值 (默认自动)
    """
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    num_frames, H, W = data.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("雷达图序列")

    def update(frame_idx):
        im.set_array(data[frame_idx])
        ax.set_title(f"雷达图 (第 {frame_idx} 帧)")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=True)

    # 根据扩展名保存
    if out_file.endswith(".gif"):
        ani.save(out_file, writer="pillow", fps=fps)
    elif out_file.endswith(".mp4"):
        ani.save(out_file, writer="ffmpeg", fps=fps)
    else:
        raise ValueError("输出文件必须是 .gif 或 .mp4")

    plt.close(fig)
    print(f"✅ 已保存动画到 {out_file}")

def save_radar_video(data, denoised_all, template=None, output_path="radar.mp4", fps=5):
    """
    将原始雷达图、噪声模板、去噪后雷达图逐帧保存为 MP4 视频

    :param data: ndarray, shape (N, H, W)，原始雷达图序列
    :param template: ndarray, shape (H, W)，噪声模板
    :param denoised_all: ndarray, shape (N, H, W)，去噪后雷达图序列
    :param output_path: str，输出 MP4 文件路径
    :param fps: int，视频帧率
    """

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ims = []

    # 初始化三幅图
    im1 = axes[0].imshow(data[0], cmap="jet", aspect="auto")
    axes[0].set_title("原始雷达图 (第0帧)")


    im2 = axes[1].imshow(denoised_all[0], cmap="jet", aspect="auto")
    axes[1].set_title("去噪后雷达图 (第0帧)")


    if template is not None:
        im3 = axes[2].imshow(template, cmap="jet", aspect="auto")
        axes[2].set_title("噪声模板")

        # 保存 axes[2] 的内容
        extent = axes[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(r"E:\MyFiles\data\template.png", bbox_inches=extent, dpi=300)

    else:
        im3 = None

    plt.tight_layout()

    def update(frame):
        im1.set_array(data[frame])
        axes[0].set_title(f"原始雷达图 (第{frame}帧)")

        im2.set_array(denoised_all[frame])
        axes[1].set_title(f"去噪后雷达图 (第{frame}帧)")

        return im1, im2, im3

    ani = animation.FuncAnimation(
        fig, update, frames=data.shape[0], blit=False, interval=1000/fps
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"视频已保存到 {output_path}")

def denoise_by_dot():

    method = "mean"
    mode = "subtract"

    # 1. 读取数据
    data = np.load(r"E:\MyFiles\data\CAPPI0408_images_gray.npy")
    print("数据维度:", data.shape)

    # 如果需要：将 0 替换为 NaN（表示缺测）
    # data = data.astype(float)
    # data[data == 0] = np.nan

    # 2. 构造噪声模板
    template = build_noise_template(data, method=method)  # (512, 480)
    # plt.imsave(r"E:\MyFiles\Projects\Banana\output/template.png", template, cmap='viridis')

    # 3. 去噪所有帧
    denoised_all = np.empty_like(data, dtype=float)
    for i in range(data.shape[0]):
        denoised_all[i] = denoise_frame(data[i], template, alpha=1.0,mode=mode)
        if i % 50 == 0:
            print(f"已去噪 {i}/{data.shape[0]} 帧")

    print("✅ 所有帧已去噪完成，结果 shape:", denoised_all.shape)

    # # 进行闭运算
    # denoised_all = apply_morphological_close(denoised_all, kernel_size=5)

    # 进行主成分分析
    temporal_pca_denoise(denoised_all,n_components=10)

    # 4. 保存结果
    output_file = r"E:\MyFiles\Projects\Banana\output/CAPPI0408_denoised_byDot.npy"
    np.save(output_file, denoised_all)
    print(f"已保存去噪结果到 {output_file}")

    # save_animation(data, r"E:\MyFiles\Projects\Banana\output\CAPPI0408_origin.mp4")
    # save_animation(denoised_all, r"E:\MyFiles\Projects\Banana\output\CAPPI0408_denoised.mp4")
    save_radar_video(data = data,
                     denoised_all = denoised_all,
                     template = template,
                     output_path = rf"E:\MyFiles\Projects\Banana\output\CAPPI0408_{method}_{mode}.mp4",
                     fps = 10)

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

def remove_stripe_noise_simple(frame, threshold=10, aspect_ratio_thresh=10, min_area=20):
    """
    去除雷达单帧中的长条噪声
    :param frame: 2D numpy array, 单帧 (512x480)
    :param threshold: 阈值，小于该值认为是噪声背景
    :param aspect_ratio_thresh: 长宽比阈值，大于此认为是条带噪声
    :param min_area: 连通域最小面积，小于此认为是噪声
    :return: 去噪后的单帧
    """
    img = frame.astype(np.uint8)  # 转为 uint8

    # 阈值化
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[img > threshold] = 255

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        aspect_ratio = max(w / (h+1e-5), h / (w+1e-5))
        # 保留较大 or 近似方块的区域
        if area >= min_area and aspect_ratio < aspect_ratio_thresh:
            clean_mask[labels == i] = 255

    denoised = img.copy()
    denoised[clean_mask == 0] = 0
    return denoised

def remove_stripe_noise_morph(frame, threshold=10, min_len=30):
    """
    使用形态学滤波去除细长条带
    """
    mask = (frame > threshold).astype(np.uint8) * 255

    # 横向/纵向结构元素（检测长条）
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, min_len))

    # 提取横条、竖条
    stripes_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h)
    stripes_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v)

    stripes = cv2.bitwise_or(stripes_h, stripes_v)

    # 去除条带
    cleaned = frame.copy()
    cleaned[stripes > 0] = 0
    return cleaned


def remove_horizontal_stripe_noise_morph(frame, threshold=10, min_len=30):
    """
    使用形态学滤波只去除横向细长条带
    """
    # 生成二值掩码
    mask = (frame > threshold).astype(np.uint8) * 255

    # 横向结构元素（检测横条）
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 3))

    # 提取横条
    stripes_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h)

    # 去除横条
    cleaned = frame.copy()
    cleaned[stripes_h > 0] = 0
    return cleaned


def remove_stripe_noise_projection(frame, threshold=10, strip_ratio=0.5):
    """
    使用投影检测条带噪声
    :param frame: 2D numpy array
    :param threshold: 强度阈值
    :param strip_ratio: 占比阈值（某一行/列大于多少比例的像素超过阈值，判定为噪声）
    """
    h, w = frame.shape
    binary = (frame > threshold).astype(np.uint8)

    # 行、列投影
    row_sum = binary.sum(axis=1) / w  # 每行占比
    col_sum = binary.sum(axis=0) / h  # 每列占比

    # 检测异常行/列
    noisy_rows = np.where(row_sum > strip_ratio)[0]
    noisy_cols = np.where(col_sum > strip_ratio)[0]

    cleaned = frame.copy()
    if len(noisy_rows) > 0:
        cleaned[noisy_rows, :] = 0
    if len(noisy_cols) > 0:
        cleaned[:, noisy_cols] = 0

    return cleaned


def process_npy(input_path, output_path, threshold=10, aspect_ratio_thresh=10, min_area=20, method="simple"):
    """
    读取 .npy 文件，逐帧去噪，并保存结果
    """
    data = np.load(input_path)   # (239, 512, 480)
    print(f"载入数据: {data.shape}, dtype={data.dtype}")

    denoised_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        if method == "simple":
            denoised_data[i] = remove_stripe_noise_simple(
                data[i],
                threshold=threshold,
                aspect_ratio_thresh=aspect_ratio_thresh,
                min_area=min_area
            )

        elif method == "morph":
            # denoised_data[i] = remove_stripe_noise_morph(data[i])
            denoised_data[i] = remove_horizontal_stripe_noise_morph(data[i])


        elif method == "projection":
            denoised_data[i] = remove_stripe_noise_projection(data[i])

        if i % 20 == 0:
            print(f"处理进度: {i}/{data.shape[0]}")

    np.save(output_path, denoised_data)
    save_radar_video(data, denoised_data, output_path=rf"E:\MyFiles\Projects\Banana/output/CAPPI0408_denoised_byStripe_{method}.mp4", fps=10)
    print(f"去噪完成，已保存到 {output_path}")


def denoise_by_stripe():
    input_file = r"E:\MyFiles\Projects\Banana\output\CAPPI0408.npy"  # 原始数据文件
    # output_file = r"E:\MyFiles\Projects\Banana\output\CAPPI0408_denoised_byStripe_projection.npy"  # 输出结果
    # process_npy(input_file, output_file,method="projection")
    output_file = r"E:\MyFiles\Projects\Banana\output\CAPPI0408_denoised_byStripe_morph.npy"  # 输出结果
    process_npy(input_file, output_file,method="morph")


if __name__ == "__main__":
    denoise_by_dot()
    # denoise_by_stripe()

