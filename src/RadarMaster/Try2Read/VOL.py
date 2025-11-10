#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VOL.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/10 16:14 
"""
import matplotlib.pyplot as plt
import numpy as np

def read_with_fixed_spacing(filename, start_offset=1049, segment_size=480, spacing=1502):
    """
    根据已知的间隔和起始位置提取数据
    :param filename: 数据文件路径
    :param start_offset: 第一段数据的起始位置（字节索引），默认 1049
    :param segment_size: 每段数据的长度（字节数），手动指定
    :param spacing: 每两段数据之间的字节间隔，默认 1502
    :return: ndarray, (段数, segment_size)
    """
    with open(filename, "rb") as f:
        raw = f.read()

    segments = []
    i = start_offset
    while i + segment_size <= len(raw):
        # 提取一段数据
        segment = raw[i:i + segment_size]
        segments.append(list(segment))
        # 移动到下一段的起始位置
        i += spacing

    return np.array(segments, dtype=np.uint8)


def visualize_array_gray(array_2d):
    """
    使用灰度图可视化二维数组
    :param array_2d: ndarray (段数, 256)
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(array_2d, aspect='auto', cmap='gray', interpolation='none')
    plt.colorbar(label='Byte value')
    plt.xlabel('Byte index in segment')
    plt.ylabel('Segment index')
    plt.title('VOL   Data Visualization (Gray)')
    plt.show()


def visualize_array_polar(array_2d):
    """
    极坐标可视化二维数组
    :param array_2d: ndarray (段数, 256)
    """
    segments, points = array_2d.shape
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R, array_2d, shading='auto', cmap='gray')
    fig.colorbar(c, ax=ax, label='Byte value')
    ax.set_title('VOL Data Visualization (Polar)')
    plt.show()

def visualize_array_polar_clockwise(array_2d):
    segments, points = array_2d.shape
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R, array_2d, shading='auto', cmap='gray')

    # 顺时针显示
    ax.set_theta_direction(-1)
    # 可选：0 度在顶部
    ax.set_theta_zero_location('N')

    fig.colorbar(c, ax=ax, label='Byte value')
    ax.set_title('VOL Data Visualization (Polar, Clockwise)')
    plt.show()


def visualize_array_polar_rotate(array_2d,
                                 rotation_deg=0,
                                 title_additional=None,
                                 n_theta_grid=12,
                                 n_r_grid=3):
    """
    生成一个极坐标图，并且可以旋转图像指定的角度（顺时针或逆时针）。

    :param array_2d: 输入的二维数组
    :param rotation_deg: 旋转角度，正值表示顺时针旋转，负值表示逆时针旋转
    :param title_additional: 自定义补充图像标题
    :param n_theta_grid: 度数线个数（默认4）
    :param n_r_grid: 环线个数（默认3）
    """
    segments, points = array_2d.shape
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    # 计算旋转角度的弧度值
    rotation_rad = np.deg2rad(rotation_deg)

    # 根据旋转角度调整Theta的起始位置
    Theta = Theta + rotation_rad

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    c = ax.pcolormesh(Theta, R, array_2d, shading='auto', cmap='gray')

    # 顺时针显示
    ax.set_theta_direction(-1)

    # 0 度在顶部
    ax.set_theta_zero_location('N')

    # 设置度数线 (n_theta_grid)
    theta_degs = np.linspace(0, 360, n_theta_grid, endpoint=False)
    ax.set_thetagrids(theta_degs)

    # 设置环线 (n_r_grid)
    r_vals = np.linspace(0, 1, n_r_grid+1)[1:]  # 跳过0，避免重复
    ax.set_rgrids(r_vals)
    ax.set_yticklabels([]) #关闭环线数据标识

    fig.colorbar(c, ax=ax, label='Byte value')
    ax.set_title(f'VOL Data Visualization (Polar, Rotated by {rotation_deg}°)'+title_additional)
    plt.show()


# 使用示例
filepath = r"C:\Users\Me\Desktop\雷达数据\20250228\VOL\QZBYNBVT250228090622.014"

# 整体处理
array_2d = read_with_fixed_spacing(filepath, start_offset=5072, segment_size=4011, spacing=4011)
print("二维数组形状:", array_2d.shape)
# 调用可视化
visualize_array_gray(array_2d)
# 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d)


# 分成四段分别成图

# # 1
# array_2d = read_with_fixed_spacing(filepath, start_offset=5072, segment_size=1000, spacing=4011)
# print("二维数组形状:", array_2d.shape)
# # # 调用可视化
# # visualize_array_gray(array_2d)
# # # 调用极坐标可视化
# # visualize_array_polar_clockwise(array_2d)
# # 尝试截取前1/14行并可视化
# # 调用可视化
# visualize_array_gray(array_2d[:526])
# # 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d[:526])


# 2
array_2d = read_with_fixed_spacing(filepath, start_offset=6072, segment_size=1000, spacing=4011)
print("二维数组形状:", array_2d.shape)
# # 调用可视化
visualize_array_gray(array_2d)
# visualize_array_gray(array_2d[:3000])
# visualize_array_gray(array_2d[:300])

# # 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d)
# 指定旋转度数和份数的可视化，忽略前20份，截取到第457份，顺时针旋转120度为最佳结果？
best2 = 457
visualize_array_polar_rotate(array_2d[20:best2],rotation_deg=120,title_additional=f' Potions:{best2}')
visualize_array_polar_rotate(array_2d[best2+10:best2*2],rotation_deg=135,title_additional=f' Potions:{best2-10}')
visualize_array_polar_rotate(array_2d[best2*2+95:best2*3+95],rotation_deg=220,title_additional=f' Potions:{best2}')

# 尝试不同的份数
# for num in range(400,500):
#     visualize_array_polar_rotate(array_2d[20:num],rotation_deg=120,title_additional=f' Potions:{num}')
#
# # 3
# array_2d = read_with_fixed_spacing(filepath, start_offset=7072, segment_size=1000, spacing=4011)
# print("二维数组形状:", array_2d.shape)
# # 调用可视化
# visualize_array_gray(array_2d)
# # 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d)
#
# # 4
# array_2d = read_with_fixed_spacing(filepath, start_offset=8072, segment_size=1000, spacing=4011)
# print("二维数组形状:", array_2d.shape)
# # 调用可视化
# visualize_array_gray(array_2d)
# # 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d)
