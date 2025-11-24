#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：KIt.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/22 10:15 
"""
import numpy as np

def delete_subarray(arr, start_idx, end_idx):
    """
    删除给定np数组中的子数组，指定子数组的起始索引和结束索引（包含）。

    参数：
    arr (np.array): 输入的np数组
    start_idx (int): 要删除的子数组的起始索引
    end_idx (int): 要删除的子数组的结束索引（包含）

    返回：
    np.array: 删除子数组后的新数组
    """
    if start_idx < 0 or end_idx >= len(arr) or start_idx > end_idx:
        raise ValueError("起始索引或结束索引无效")

    # 删除子数组
    return np.concatenate((arr[:start_idx], arr[end_idx + 1:]))

data = np.load(r"E:\MyFiles\data\CAPPI0408_images_gray.npy")
data = delete_subarray(data, 139, 139)
np.save(r"E:\MyFiles\data\CAPPI0408_images_gray.npy",data)