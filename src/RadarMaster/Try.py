#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：Try.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/15 16:33 
"""

from tqdm import tqdm


# def denoise_radar_images(input_dir, output_dir):
#     """
#     基于帧间像素差分的雷达图像去噪
#     :param input_dir: 输入图像文件夹路径
#     :param output_dir: 输出图像文件夹路径
#     """
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 按文件名排序加载所有图片
#     filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     images = [cv2.imread(os.path.join(input_dir, f)) for f in filenames]
#
#     if len(images) < 2:
#         print("图片数量不足，至少需要两张。")
#         return
#
#     print(f"共加载 {len(images)} 张图片。")
#
#     # 计算相邻帧的像素差值
#     diffs = []
#     for i in range(1, len(images)):
#         diff = cv2.absdiff(images[i], images[i - 1])
#         diff_mean = np.mean(diff)  # 整张图像的平均差值
#         diffs.append((diff, diff_mean))
#
#     # 去噪：若该像素与前一帧的差值小于平均差值，则认为是噪声
#     for i in tqdm(range(1, len(images))):
#         diff, mean_val = diffs[i - 1]
#         mask = np.all(diff < mean_val, axis=-1)  # 每个像素三个通道都小于平均差
#         denoised = images[i].copy()
#         denoised[mask] = [0, 0, 0]  # 替换噪声
#         out_path = os.path.join(output_dir, filenames[i])
#         cv2.imwrite(out_path, denoised)
#
#     print("去噪完成，结果已保存到：", output_dir)

def denoise_radar_images(input_dir, output_dir):
    """
    基于帧间差分的像素级去噪：
    每个像素都有独立的平均差阈值
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有图片（RGB）
    filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [cv2.imread(os.path.join(input_dir, f)) for f in filenames]

    if len(images) < 2:
        print("图片数量不足，至少需要两张。")
        return

    print(f"共加载 {len(images)} 张图片。")

    # ---------- 计算每个像素的平均差值 ----------
    diffs = []
    for i in range(1, len(images)):
        diff = cv2.absdiff(images[i], images[i - 1])
        diff_scalar = np.linalg.norm(diff, axis=-1)  # RGB统一为单通道差
        diffs.append(diff_scalar)

    mean_diff_map = np.mean(np.stack(diffs, axis=0), axis=0)  # shape (H, W)
    print("像素级平均差值计算完成。")

    # ---------- 去噪 ----------
    for i in tqdm(range(1, len(images))):
        diff = cv2.absdiff(images[i], images[i - 1])
        diff_scalar = np.linalg.norm(diff, axis=-1)

        # 每个像素都有自己的平均差阈
        mask = diff_scalar < mean_diff_map

        denoised = images[i].copy()
        denoised[mask] = [0, 0, 0]

        out_path = os.path.join(output_dir, filenames[i])
        cv2.imwrite(out_path, denoised)

    print("✅ 去噪完成，结果已保存到：", output_dir)

def denoise_radar_images_gray(input_dir, output_dir):
    """
    基于帧间差分的灰度雷达图像去噪
    :param input_dir: 输入图像文件夹路径
    :param output_dir: 输出图像文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按文件名排序读取所有灰度图
    filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE) for f in filenames]

    if len(images) < 2:
        print("图片数量不足，至少需要两张。")
        return

    print(f"共加载 {len(images)} 张灰度图。")

    # 计算相邻帧的像素差
    diffs = []
    for i in range(1, len(images)):
        diff = cv2.absdiff(images[i], images[i - 1])  # 单通道差值
        diff_mean = np.mean(diff)  # 全图平均差
        diffs.append((diff, diff_mean))

    # 去噪：若当前像素与前一帧差值小于平均值，则认为是静态噪声
    for i in tqdm(range(1, len(images))):
        diff, mean_val = diffs[i - 1]
        mask = diff < mean_val  # 噪声掩码
        denoised = images[i].copy()
        denoised[mask] = 0  # 将噪声点置为
        out_path = os.path.join(output_dir, filenames[i])
        cv2.imwrite(out_path, denoised)

    print("灰度图去噪完成，结果已保存到：", output_dir)


import cv2
import os
import numpy as np
from natsort import natsorted


def make_side_by_side_video(folder1, folder2, output_path="output.mp4", fps=24):
    """
    将两个文件夹中同名图像左右拼接生成视频。
    若某个文件名在任一文件夹中缺失，则跳过该帧。

    参数:
        folder1 (str): 第一批图像所在文件夹路径
        folder2 (str): 第二批图像所在文件夹路径
        output_path (str): 输出视频文件路径
        fps (int): 视频帧率
    """
    # 获取图像文件列表（自然排序）
    files1 = natsorted([f for f in os.listdir(folder1) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    files2 = natsorted([f for f in os.listdir(folder2) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # 求交集文件
    common_files = sorted(list(set(files1) & set(files2)))

    if not common_files:
        print("❌ 没有找到相同文件名的图片，无法生成视频。")
        return

    print(f"🧩 找到 {len(common_files)} 个匹配文件，将生成视频...")

    # 尝试读取第一张有效图片确定视频尺寸
    for name in common_files:
        img1 = cv2.imread(os.path.join(folder1, name))
        img2 = cv2.imread(os.path.join(folder2, name))
        if img1 is not None and img2 is not None:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            frame_height, frame_width = img1.shape[0], img1.shape[1] * 2
            break
    else:
        print("❌ 所有匹配文件都无法读取。")
        return

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    skipped = 0

    for name in common_files:
        path1 = os.path.join(folder1, name)
        path2 = os.path.join(folder2, name)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"⚠️ 跳过无法读取的文件: {name}")
            skipped += 1
            continue

        # 尺寸对齐
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 左右拼接
        combined = np.hstack((img1, img2))
        out.write(combined)

    out.release()
    print(f"✅ 视频已生成: {output_path}")
    if skipped > 0:
        print(f"⚠️ 共跳过 {skipped} 个无法匹配或读取的文件。")



if __name__ == "__main__":
    # src=r"C:\Users\Me\Desktop\CAPPI\data"
    # out=r"C:\Users\Me\Desktop\CAPPI\output"
    # denoise_radar_images(
    #     input_dir=src,
    #     output_dir=out
    # )
    # make_side_by_side_video(src, out, r"E:\MyFiles\Projects\Banana\output/compare.mp4", fps=10)

    # import random
    #
    # random.seed(5)
    # numbers = list(range(1, 7))
    # random.shuffle(numbers)
    # print("随机排序后的数字:", numbers)

    # data=np.load(r"E:\MyFiles\data\CAPPI0408_images_single.npy")
    # np.save(
    #     file=r"E:\MyFiles\data\CAPPI0408_images_single_sample.npy",
    #     arr=data[0]
    # )
    # data=np.load(r"E:\MyFiles\data\CAPPI0408_images_single_sample.npy")
    # print(data.shape)

    for a in range(0,100000000,5):
        print(a)






