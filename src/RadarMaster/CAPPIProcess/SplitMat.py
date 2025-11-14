#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：SplitMat.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/11 11:28 
"""
import os
import numpy as np
from scipy.io import loadmat, savemat

def split_mat_dataset(
    input_path,
    train_ratio=0.8,
    output_dir=None,
    shuffle=True,
    seed=42
):
    """
    将指定的 .mat 文件划分为训练集和测试集。

    参数：
    ----------
    input_path : str
        输入 .mat 文件路径（应包含 a, u, t 三个键）
    train_ratio : float
        训练集比例 (0~1)
    output_dir : str 或 None
        输出文件夹路径。若为 None，则保存在输入文件同目录下
    shuffle : bool
        是否在划分前随机打乱样本
    seed : int
        随机种子，用于可复现划分结果

    输出：
    ----------
    train_xxx.mat, test_xxx.mat
    """

    # === 加载数据 ===
    data = loadmat(input_path)
    a, u, t = data["a"], data["u"], data["t"]
    N = a.shape[0]

    # === 打乱索引 ===
    indices = np.arange(N)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    # === 划分 ===
    split_idx = int(N * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    a_train, u_train = a[train_idx], u[train_idx]
    a_test, u_test = a[test_idx], u[test_idx]

    # === 输出路径 ===
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    train_path = os.path.join(output_dir, f"{base_name}_train_{train_ratio}proportion.mat")
    test_path = os.path.join(output_dir, f"{base_name}_test_{round(1 - train_ratio, 2)}proportion.mat")

    # === 保存文件 ===
    savemat(train_path, {"a": a_train, "u": u_train, "t": t})
    savemat(test_path, {"a": a_test, "u": u_test, "t": t})

    print(f"✅ 数据划分完成：")
    print(f"  总样本数: {N}")
    print(f"  训练集: {len(train_idx)}  → {train_path}")
    print(f"  测试集: {len(test_idx)}  → {test_path}")


split_mat_dataset(
    input_path=r"E:\MyFiles\data\CAPPI0010_singled.mat",  # 输入文件
    train_ratio=0.8,      # 训练数据占比
    output_dir=r"E:\MyFiles\data",# 输出目录
    shuffle=True           # 随机打乱
)

