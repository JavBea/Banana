#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：DenoiseCircle.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/17 14:34 
"""

import numpy as np
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt


def visualize_color_count_3d(npy_path,target_hex="#FFFFFF",freq_min=None,freq_max=None):
    """
    统计大型 RGB 时序图像中，某个十六进制颜色在每个像素点出现的次数。
    支持输入 shape = (T, H, W, 3)。

    参数：
        npy_path : str
            RGB npy 文件路径
        target_hex : str
            目标颜色（十六进制字符串，如 "#00FF00"）
        freq_min, freq_max : int or None
            频率过滤范围，超出范围的设为 NaN
    """

    # === 转换十六进制颜色到 RGB ===
    hex_str = target_hex.lstrip("#")
    target_rgb = np.array([
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16)
    ], dtype=np.uint8)

    # === 加载数据 ===
    data = np.load(npy_path)  # shape = (T, H, W, 3)
    T, H, W, _ = data.shape

    # === 统计出现次数 ===
    # 匹配每帧中是否等于目标颜色 → 得到布尔数组
    mask = np.all(data == target_rgb, axis=-1)  # shape (T, H, W)

    # 统计每个像素出现次数
    color_count = mask.sum(axis=0).astype(float)  # 转 float 方便 NaN

    # === 频率过滤 ===
    filtered_count = color_count.copy()
    if freq_min is not None:
        filtered_count[filtered_count < freq_min] = np.nan
    if freq_max is not None:
        filtered_count[filtered_count > freq_max] = np.nan

    # === 坐标网格 ===
    x = np.arange(W)
    y = np.arange(H)

    # === 3D surface 绘图 ===
    fig = go.Figure(
        data=[go.Surface(
            z=filtered_count,
            x=x,
            y=y,
            colorscale='Jet',
            colorbar=dict(title=f"Count of {target_hex}"),
            showscale=True
        )]
    )

    title = f"3D Count of Color {target_hex}"
    if freq_min is not None or freq_max is not None:
        title += f" (Filtered: {freq_min} ≤ freq ≤ {freq_max})"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Width)',
            yaxis_title='Y (Height)',
            zaxis_title='Match Count'
        ),
        width=900,
        height=700
    )

    fig.show()

    return color_count, filtered_count

def visualize_threshold_count_3d(npy_path,r_min=0,g_min=0,b_min=0,freq_min=None,freq_max=None):
    """
    统计大型 RGB 时序图像中，当每帧像素满足：
        R > r_min 且 G > g_min 且 B > b_min
    的出现次数。
    输入 shape = (T, H, W, 3)

    参数：
        r_min, g_min, b_min : int
            单通道阈值，当 RGB 三者均满足条件时计数 +1
        freq_min, freq_max : int or None
            频率过滤范围，超出范围设为 NaN
    """

    # === 加载数据 ===
    data = np.load(npy_path)  # shape = (T, H, W, 3)
    T, H, W, _ = data.shape

    # === 判定掩码 ===
    mask = (
        (data[..., 0] > r_min) &
        (data[..., 1] > g_min) &
        (data[..., 2] > b_min)
    )  # shape (T, H, W), bool

    # === 统计出现次数 ===
    count = mask.sum(axis=0).astype(float)

    # === 频率过滤 ===
    filtered_count = count.copy()
    if freq_min is not None:
        filtered_count[filtered_count < freq_min] = np.nan
    if freq_max is not None:
        filtered_count[filtered_count > freq_max] = np.nan

    # === 坐标网格 ===
    x = np.arange(W)
    y = np.arange(H)

    # === 绘制 3D surface 图 ===
    fig = go.Figure(
        data=[go.Surface(
            z=filtered_count,
            x=x,
            y=y,
            colorscale='Jet',
            colorbar=dict(title="Count"),
            showscale=True
        )]
    )

    title = f"3D Count of RGB > [{r_min}, {g_min}, {b_min}]"
    if freq_min is not None or freq_max is not None:
        title += f" (Filtered: {freq_min} ≤ freq ≤ {freq_max})"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Width)',
            yaxis_title='Y (Height)',
            zaxis_title='Count'
        ),
        width=900,
        height=700
    )

    fig.show()

    return count, filtered_count


def generate_rgb_threshold_mask(npy_path,output_path,r_min=0,g_min=0,b_min=0):
    """
    生成 0-1 掩码矩阵，掩码 shape = (H, W)
    坐标定义：
        x 代表 height
        y 代表 width
    判定条件：
        R > r_min 且 G > g_min 且 B > b_min
    """

    # === 加载数据 ===
    data = np.load(npy_path)  # shape = (T, H, W, 3)
    T, H, W, _ = data.shape

    # === 阈值判定 ===
    mask = (
        (data[..., 0] > r_min) &
        (data[..., 1] > g_min) &
        (data[..., 2] > b_min)
    )  # shape = (T, H, W)

    # === 合并：是否曾至少满足一次 ===
    final_mask = mask.any(axis=0).astype(np.uint8)  # shape = (H, W)

    # 注意：此时 final_mask[x, y] 就是 final_mask[height, width]

    # === 保存 ===
    np.save(output_path, final_mask)
    print(f"0-1 掩码已保存到: {output_path}")

    return final_mask

def generate_rgb_threshold_mask_png(png_path,output_path,r_min=0,g_min=0,b_min=0):
    """
    针对单张 PNG 图片生成 0/1 掩码矩阵
    坐标定义：
        左上角为 [0,0]
        图片 shape = (H, W, 3)
        掩码 shape = (H, W) ，其中 mask[x, y] 对应 (height=x, width=y)

    条件：
        R > r_min 且 G > g_min 且 B > b_min
    """

    # === 加载 PNG 图片 ===
    img = Image.open(png_path).convert("RGB")
    data = np.array(img)                   # shape = (H, W, 3)
    H, W, _ = data.shape

    # === 阈值判定 ===
    mask = (
        (data[..., 0] > r_min) &
        (data[..., 1] > g_min) &
        (data[..., 2] > b_min)
    ).astype(np.uint8)                     # shape = (H, W)

    # 左上角自动是 [0,0]，无需额外处理

    # === 保存 ===
    np.save(output_path, mask)
    print(f"0-1 掩码已保存到: {output_path}")

    return mask


def apply_mask_and_smooth_ignore_mask(img_path,mask_path,output_path,a=3):
    """
    使用掩码矩阵对图像平滑修补：
    对 mask[x, y] == 1 的像素点：
        替换为周围 a*a（排除自己 & 排除所有掩码==1 的像素）的 RGB 均值

    参数：
        img_path: str         原始 PNG 图片路径
        mask_path: str        掩码 0-1 npy 文件
        output_path: str      输出路径
        a: int                邻域大小，必须为奇数，如 3/5/7
    """

    if a % 2 == 0:
        raise ValueError("a 必须为奇数，例如 3,5,7")

    # === 加载数据 ===
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img).astype(np.float32)  # (H, W, 3)

    mask = np.load(mask_path)                   # (H, W)
    H, W = mask.shape

    if img_arr.shape[:2] != mask.shape:
        raise ValueError("图像尺寸 与 掩码尺寸 不一致")

    r = a // 2  # 半径

    output = img_arr.copy()

    xs, ys = np.where(mask == 1)

    for x, y in zip(xs, ys):

        # ---窗口范围---
        x1 = max(0, x - r)
        x2 = min(H, x + r + 1)
        y1 = max(0, y - r)
        y2 = min(W, y + r + 1)

        window = img_arr[x1:x2, y1:y2]  # (a, a, 3)
        window_mask = mask[x1:x2, y1:y2]  # 同样大小

        # ---构造邻域有效像素 mask---
        neighbor_mask = np.ones_like(window_mask, dtype=bool)

        # 排除中心像素
        wx, wy = x - x1, y - y1
        neighbor_mask[wx, wy] = False

        # 排除所有 mask==1 的像素
        neighbor_mask[window_mask == 1] = False

        # 获取有效邻域像素
        valid_pixels = window[neighbor_mask]

        if valid_pixels.size == 0:
            # 没有可用邻域 pixels，保留原像素
            continue

        mean_rgb = valid_pixels.reshape(-1, 3).mean(axis=0)

        # 写回
        output[x, y] = mean_rgb

    # === 保存 ===
    out_img = Image.fromarray(output.astype(np.uint8))
    out_img.save(output_path)

    print(f"处理完成，已保存到：{output_path}")

    return output

def apply_mask_and_smooth_ignore_mask_npy(input_path, mask_path, input_data=None, output_path=None, a=3):
    """
    对一个 [T, H, W, 3] 的多帧 RGB npy 数据进行掩码平滑修补。
    对 mask[x, y] == 1 的像素点：
        替换为周围 a*a 区域（排除自己 & 排除所有掩码==1 的像素）的 RGB 均值。

    参数：
        input_path         : str    输入 .npy，形状 [T, H, W, 3]
        mask_path          : str    掩码文件，形状 [H, W]
        input_data         : str     直接传入的data参数，非None时直接忽略input_path
        output_path        : str    输出 .npy
        a                  : int    邻域大小，必须为奇数
    """

    if a % 2 == 0:
        raise ValueError("a 必须为奇数，例如 3, 5, 7")

    # === 加载数据 ===
    if input_data is None:
        video = np.load(input_path).astype(np.uint8)   # (T,H,W,3)
    else:
        video = input_data

    T, H, W, _ = video.shape

    mask = np.load(mask_path)                                # (H,W)
    if mask.shape != (H, W):
        raise ValueError("掩码尺寸与视频单帧尺寸不一致")

    output = video.copy()

    r = a // 2
    xs, ys = np.where(mask == 1)

    print(f"需要处理的掩码像素数量: {len(xs)}")

    # === 逐帧处理 ===
    for t in range(T):
        frame = video[t]

        for x, y in zip(xs, ys):

            x1 = max(0, x - r)
            x2 = min(H, x + r + 1)
            y1 = max(0, y - r)
            y2 = min(W, y + r + 1)

            window = frame[x1:x2, y1:y2]
            window_mask = mask[x1:x2, y1:y2]

            neighbor_mask = np.ones_like(window_mask, dtype=bool)

            wx, wy = x - x1, y - y1
            neighbor_mask[wx, wy] = False
            neighbor_mask[window_mask == 1] = False

            valid_pixels = window[neighbor_mask]

            if valid_pixels.size == 0:
                continue

            mean_rgb = valid_pixels.reshape(-1, 3).mean(axis=0)
            output[t, x, y] = mean_rgb

        if (t + 1) % 10 == 0:
            print(f"已处理 {t+1}/{T} 帧")

    if output_path is not None:
        np.save(output_path, output.astype(np.uint8))
        print(f"\n处理完成，已保存到：{output_path}")

    return output


def replace_dark_pixels(img_path,output_path,r_th=30,g_th=30,b_th=30):
    """
    将 image 中 R/G/B 同时 < 对应阈值的像素替换为黑色 (#000000)

    参数:
        img_path:     输入图片路径 (.png/.jpg)
        output_path:  输出图片路径
        r_th/g_th/b_th: R/G/B 各通道的阈值
    """

    # 读取图片
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.uint8)   # (H, W, 3)

    # 拆分 RGB
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]

    # 条件：三个通道都小于阈值
    mask = (R < r_th) & (G < g_th) & (B < b_th)

    # 替换
    arr[mask] = [0, 0, 0]   # 黑色

    # 保存
    out_img = Image.fromarray(arr)
    out_img.save(output_path)

    print(f"处理完成，已保存到：{output_path}")

    return arr

def replace_dark_pixels_npy(input_path, output_path=None, input_data=None, r_th=100, g_th=100, b_th=100):
    """
    对一个 [T, H, W, 3] 的多帧 RGB npy 数据进行“暗像素替换为黑色”处理。
    条件：R < r_th AND G < g_th AND B < b_th

    参数：
        input_path         : str   输入 .npy, 形状 [T, H, W, 3]
        output_path        : str   输出 .npy, 形状 [T, H, W, 3]
        input_data         : str   直接传入的data参数，非None时直接忽略input_path
        r_th/g_th/b_th     : int   阈值
    """

    # === 加载数据 ===
    if input_data is None:
        video = np.load(input_path).astype(np.uint8)   # (T,H,W,3)
    else:
        video = input_data
    T, H, W, _ = video.shape

    output = video.copy()

    # === 逐帧处理 ===
    for t in range(T):

        frame = output[t]   # (H,W,3)

        R = frame[..., 0]
        G = frame[..., 1]
        B = frame[..., 2]

        # 条件：三个通道都小于阈值
        mask = (R < r_th) & (G < g_th) & (B < b_th)

        # 替换为黑色
        frame[mask] = [0, 0, 0]

        if (t + 1) % 10 == 0:
            print(f"已处理 {t+1}/{T} 帧")

    if output_path is not None:
        # === 保存 ===
        np.save(output_path, output.astype(np.uint8))
        print(f"\n处理完成，已保存到：{output_path}")

    return output


def visualize_mask_clean(mask_path, save_path=None):
    mask = np.load(mask_path)

    plt.figure(figsize=(10, 10), dpi=100)  # 10*100 = 1000 像素
    plt.imshow(mask, cmap="gray", interpolation="nearest")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
        print("保存完成:", save_path)

    plt.show()
    plt.close()


def generate_light_yellow_mask(img_path,output_path,h_min=20, h_max=200,s_min=20, s_max=255,v_min=20, v_max=255):
    """
    生成浅黄色区域掩码矩阵（通过 HSV 颜色过滤）
    """

    # 加载图像并转 HSV
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # PIL 转 HSV
    hsv = np.array(img.convert("HSV"))

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    mask = (
        (H >= h_min) & (H <= h_max) &
        (S >= s_min) & (S <= s_max) &
        (V >= v_min) & (V <= v_max)
    ).astype(np.uint8)

    np.save(output_path, mask)
    print(f"掩码已保存到: {output_path}")

    return mask


def merge_npy_union(file1_path, file2_path, output_path):
    """
    将两个0-1矩阵的并集生成一个新的矩阵并保存
    参数:
        file1_path : str
            第一个 .npy 文件路径
        file2_path : str
            第二个 .npy 文件路径
        output_path : str
            输出 .npy 文件路径
    """
    # 读取两个矩阵
    mat1 = np.load(file1_path)
    mat2 = np.load(file2_path)

    # 检查形状是否一致
    if mat1.shape != mat2.shape:
        raise ValueError(f"两个矩阵形状不同: {mat1.shape} vs {mat2.shape}")

    # 计算并集（逻辑或）
    union_mat = np.logical_or(mat1, mat2).astype(np.uint8)  # 转回0-1矩阵

    # 保存到指定路径
    np.save(output_path, union_mat)
    print(f"已将并集矩阵保存到: {output_path}")


def set_one(h, w, input_path, output_path=None):

    if output_path is None:
        output_path = input_path

    data = np.load(input_path)

    data[h,w]=1

    np.save(output_path, data)

def set_zero(h, w, input_path, output_path=None):

    if output_path is None:
        output_path = input_path

    data = np.load(input_path)

    data[h,w]=0

    np.save(output_path, data)


def denoise_circle_fonts(input_path,mask_path,output_path,a=3,r_th=100,g_th=100,b_th=100):

    data = apply_mask_and_smooth_ignore_mask_npy(
        input_path=input_path,
        mask_path=mask_path,
        input_data=None,
        output_path=None,
        a=a
    )

    data = replace_dark_pixels_npy(
        input_path=None,
        output_path=output_path,
        input_data=data,
        r_th=r_th,
        g_th=g_th,
        b_th=b_th
    )

    return data





if __name__ == '__main__':

    # # 生成黄白圈、线的静态掩码矩阵
    # generate_rgb_threshold_mask_png(
    #     png_path=r"E:\MyFiles\data\static\CAPPI0408空白图.JPG",
    #     output_path=r"E:\MyFiles\data\CAPPI0408_white_circle.npy",
    #     r_min=200,
    #     g_min=200,
    #     b_min=200
    # )
    #
    # visualize_mask_clean(
    #     mask_path=r"E:\MyFiles\data\CAPPI0408_white_circle.npy",
    #     save_path=r"E:\MyFiles\data\CAPPI0408_white_circle.jpg"
    # )
    #
    #
    # generate_light_yellow_mask(
    #     img_path=r"E:\MyFiles\data\static\CAPPI0408空白图.JPG",
    #     output_path=r"E:\MyFiles\data\CAPPI0408_yellow_circle.npy",
    # )
    #
    #
    # visualize_mask_clean(
    #     mask_path=r"E:\MyFiles\data\CAPPI0408_yellow_circle.npy",
    #     save_path=r"E:\MyFiles\data\CAPPI0408_yellow_circle.jpg"
    # )
    #
    # merge_npy_union(
    #     file1_path=r"E:\MyFiles\data\CAPPI0408_white_circle.npy",
    #     file2_path=r"E:\MyFiles\data\CAPPI0408_yellow_circle.npy",
    #     output_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy"
    # )
    #
    # visualize_mask_clean(
    #     mask_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy",
    #     save_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.jpg"
    # )


    # # 生成字体的静态掩码矩阵
    # generate_rgb_threshold_mask_png(
    #     png_path=r"E:\MyFiles\data\static\CAPPI0408剩余图.JPG",
    #     output_path=r"E:\MyFiles\data\CAPPI0408_font.npy",
    #     r_min=60,
    #     g_min=60,
    #     b_min=60
    # )
    # merge_npy_union(
    #     file1_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy",
    #     file2_path=r"E:\MyFiles\data\CAPPI0408_font.npy",
    #     output_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle_font.npy"
    # )

    # # 掩码矩阵遗落下的四个点，手动地加进去
    # coordinates = [
    #     [506,104],
    #     [508,32],
    #     [506,265],
    #     [506,270],
    # ]
    # for h,w in coordinates:
    #     set_one(
    #         h=h,
    #         w=w,
    #         input_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle_font.npy"
    #     )
    #
    # visualize_mask_clean(
    #     mask_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle_font.npy",
    #     save_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle_font.jpg"
    # )


    # 单张图片去噪
    apply_mask_and_smooth_ignore_mask(
        img_path=r"E:\MyFiles\data\static\CAPPI-202504081450-0010-150-Z.JPG",
        mask_path=r"E:\MyFiles\data\static\CAPPI0408_white_yellow_circle_font.npy",
        output_path=r"E:\MyFiles\data\aaa.JPG",
        a=7
    )

    replace_dark_pixels(
        img_path=r"E:\MyFiles\data\aaa.JPG",
        output_path=r"E:\MyFiles\data\aaa.JPG",
        r_th=100,
        g_th=100,
        b_th=100
    )


    # apply_mask_and_smooth_ignore_mask_npy(
    #     r"E:\MyFiles\data\CAPPI0408_images_origin.npy",
    #     r"E:\MyFiles\data\static\CAPPI0408_white_yellow_circle_font.npy",
    #     r"E:\MyFiles\data\CAPPI0408_images_origin_denoise0.npy",
    #     a=3
    # )

    # npy去噪
    # 三个参数，尤其是a，万不可变
    denoise_circle_fonts(
        input_path=r"E:\MyFiles\data\CAPPI0408_images_origin.npy",
        mask_path=r"E:\MyFiles\data\static\CAPPI0408_white_yellow_circle_font.npy",
        output_path=r"E:\MyFiles\data\CAPPI0408_images_origin_denoise0.npy",
        a=7,
        r_th=100,
        g_th=100,
        b_th=100
    )

    # for h in range(486, 491):
    #     for w in range(7,9):
    #         set_by_hand(r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy",r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy",h,w)
    #
    # visualize_mask_clean(
    #     mask_path=r"E:\MyFiles\data\CAPPI0408_white_yellow_circle.npy"
    # )