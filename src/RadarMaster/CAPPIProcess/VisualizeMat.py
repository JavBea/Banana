#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：VisualizeMat.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/11 14:31 
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_mat_data(mat_path, key='u', index=0, play=False):
    """
    可视化 .mat 文件中的数据
    参数:
        mat_path : str
            .mat 文件路径
        key : str
            选择可视化的键（如 'a' 或 'u'）
        index : int
            当 key='u' 时，指定样本索引（即第几个样本）
        play : bool
            若为 True，将播放时间序列（仅当 key='u' 时生效）
    """
    data = sio.loadmat(mat_path)

    if key not in data:
        raise KeyError(f"键 '{key}' 不存在，文件中可选的键有: {list(data.keys())}")

    arr = data[key]
    print(f"{key} 的形状: {arr.shape}, 类型: {arr.dtype}")

    if key == 'a':
        # a: (N, 64, 64)
        plt.imshow(arr[index], cmap='bwr')
        plt.colorbar(label="Value")
        plt.title(f"a[{index}]")
        plt.show()

    elif key == 'u':
        # u: (N, 64, 64, 20)
        if play:
            fig, ax = plt.subplots()
            # vmin, vmax = np.nanmin(arr), np.nanmax(arr)

            vmin = np.nanpercentile(arr, 1)
            vmax = np.nanpercentile(arr, 99)

            im = ax.imshow(arr[index, :, :, 0], cmap='bwr', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)

            def update(frame):
                im.set_data(arr[index, :, :, frame])
                ax.set_title(f"u[{index}], frame {frame}")
                return [im]

            ani = animation.FuncAnimation(fig, update, frames=arr.shape[-1], interval=300, blit=True)
            plt.show()
        else:
            # 显示指定样本的某一时间帧
            frame = 0
            plt.imshow(arr[index, :, :, frame], cmap='bwr')
            plt.colorbar(label="Value")
            plt.title(f"u[{index}], frame {frame}")
            plt.show()
    else:
        print(f"暂不支持键 {key} 的自动可视化")


# 示例调用
# 可视化 a 中的第 0 个样本
# visualize_mat_data("data.mat", key='a', index=0)

# 可视化 u 中的第 index 个样本的动态时间序列
visualize_mat_data(r"E:\MyFiles\data\CAPPI0010_singled_128px.mat", key='u', index=3, play=True)
