#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RemoveStripes.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/20 9:47 
"""

import numpy as np

def point_in_acute_sector(
        center,
        p1,
        p2,
        q
):
    """
    判断点 q 是否位于由射线 center->p1 与 center->p2 之间的锐角区域内。

    参数:
        center: (ch, cw) 中心点坐标
        p1:     (h1, w1) 射线1上的一点
        p2:     (h2, w2) 射线2上的一点
        q:      (h, w)   待判断坐标

    返回:
        True / False
    """

    C = np.array(center, dtype=float)
    V1 = np.array(p1, dtype=float) - C
    V2 = np.array(p2, dtype=float) - C
    Vq = np.array(q, dtype=float) - C

    # 如果 q 就在中心点上，则认为不在夹角内
    if np.allclose(Vq, 0):
        return False

    # 计算三者的夹角（利用夹角余弦）
    def angle_between(a, b):
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0
        cosang = dot / (na * nb)
        cosang = np.clip(cosang, -1, 1)
        return np.arccos(cosang)

    # 两条射线之间的角度
    angle_12 = angle_between(V1, V2)
    # 点 q 与这两条射线的角度
    angle_1q = angle_between(V1, Vq)
    angle_q2 = angle_between(Vq, V2)

    # 判断：V1 和 V2 之间的锐角是否包含 Vq
    return angle_1q <= angle_12 and angle_q2 <= angle_12

def build_distance_map(
        center=(500, 500),
        H=1000,
        W=1000,
        output_path=None
):
    """
    计算矩阵上所有点到 center 的欧氏距离，并保存为 .npy 文件。

    参数:
        center      : (ch, cw) 中心点坐标
        H, W        : 高度和宽度（默认1000x1000）
        output_path : 输出 .npy 文件路径
    """

    ch, cw = center

    # 构建坐标网格
    hh, ww = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # 欧氏距离
    dist = np.sqrt((hh - ch)**2 + (ww - cw)**2)

    if output_path is not None:
        # 保存
        np.save(output_path, dist)

    return dist

def build_sector_mask(
        center,
        p1,
        p2,
        H=1000,
        W=1000
):
    """
    生成在射线 center->p1 和 center->p2 之间的锐角区域掩码。
    """

    C = np.array(center, dtype=float)
    V1 = np.array(p1, dtype=float) - C
    V2 = np.array(p2, dtype=float) - C

    # 计算射线之间的角度（仅一次）
    def angle_between(a, b):
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        cosang = dot / (na * nb)
        cosang = np.clip(cosang, -1, 1)
        return np.arccos(cosang)

    angle_12 = angle_between(V1, V2)

    # 构建坐标网格
    hh, ww = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    PQ = np.stack([hh - C[0], ww - C[1]], axis=-1).astype(float)

    # 计算每个点与射线的夹角
    def cos_angle(A, B):
        dot = A[...,0]*B[0] + A[...,1]*B[1]
        na = np.linalg.norm(A, axis=-1)
        nb = np.linalg.norm(B)
        return dot / (na * nb + 1e-12)

    angle_1q = np.arccos(np.clip(cos_angle(PQ, V1), -1, 1))
    angle_q2 = np.arccos(np.clip(cos_angle(PQ, V2), -1, 1))

    # 判定是否在锐角内
    mask = (angle_1q <= angle_12) & (angle_q2 <= angle_12)

    return mask.astype(np.int8)

def build_sector_ring_mask(
        sector_mask,
        a,
        b,
        dist_map_path=None,
        dist=None
):
    """
    根据扇形掩码和距离范围 [a, b)，生成最终掩码矩阵。

    参数：
        sector_mask : 2D np.ndarray, 0/1，表示是否在扇形区域内
        dist_map_path : str，距离矩阵 npy 文件路径
        a, b : float，要求 a <= dist < b 的点
        dist : 直接传入数组

    返回：
        final_mask : 0/1 掩码矩阵
    """

    # 读取距离矩阵
    if dist is None:
        dist = np.load(dist_map_path)

    # 距离范围掩码
    dist_mask = (dist >= a) & (dist < b)

    # 扇形 ∩ 距离区间
    final_mask = sector_mask.astype(bool) & dist_mask

    return final_mask.astype(np.int8)

def build_mask(
        center=(500, 500),
        left1 =(116,501),
        left2 = (116,514),
        right1 = (118,549),
        right2 = (122,568),
        H=1000,
        W=1000,
        step=10,
        distance_min=0,
        distance_max=400,
        output_path_surrounding=None,
        output_path_central=None
):
    dist = build_distance_map(center, H, W)
    sector_left = build_sector_mask(center,p1=left1,p2=left2,H=1000,W=1000)
    sector_right = build_sector_mask(center,p1=right1,p2=right2,H=1000,W=1000)
    sector_central = build_sector_mask(center,p1=left2,p2=right1,H=1000,W=1000)

    surrounding = []
    central = []


    for a in range(distance_min,distance_max,step):

        left_ring = build_sector_ring_mask(sector_mask=sector_left, a=a, b=a + step, dist=dist)
        right_ring = build_sector_ring_mask(sector_mask=sector_right, a=a, b=a + step, dist=dist)
        central_ring = build_sector_ring_mask(sector_mask=sector_central, a=a, b=a + step, dist=dist)

        surrounding.append(left_ring-right_ring)
        central.append(central_ring)

    if output_path_surrounding is not None:
        np.save(output_path_surrounding, surrounding)

    if output_path_central is not None:
        np.save(output_path_central, central)

    return surrounding,central

def remove_stripes(
    surrounding,
    central,
    data,
    pos_condition,    # surrounding > 0 区域的判断函数
    neg_condition,    # surrounding < 0 区域的判断函数
    assign_value      # 若满足条件，赋给 central > 0 区域的值
):
    """
    根据 surrounding 的正区/负区条件，为 central 正区对应位置赋值。

    参数：
        surrounding   : 2D array，包含 正数/负数/0
        central       : 2D array，包含 正数/0
        data          : 2D array，与上述矩阵等大小
        pos_condition : 函数 f(data_values) → bool
                        对应 surrounding>0 区域的条件
        neg_condition : 函数 f(data_values) → bool
                        对应 surrounding<0 区域的条件
        assign_value  : 满足两个条件后赋给 central>0 区域的值

    返回：
        result : 2D array，处理后的矩阵
    """

    # 结果矩阵（深拷贝 data）
    result = data.copy()

    # 掩码
    pos_mask = surrounding > 0
    neg_mask = surrounding < 0
    central_mask = central > 0

    # 获取对应区域的数据
    pos_values = data[pos_mask]
    neg_values = data[neg_mask]

    # 判断条件
    cond_pos = pos_condition(pos_values)
    cond_neg = neg_condition(neg_values)

    # 两个条件都成立 → central 正区赋新值
    if cond_pos and cond_neg:
        result[central_mask] = assign_value

    return result

def ratio_more_than_0_smaller_10(values):
    if len(values) == 0:
        return False
    ratio = np.mean(values > 0)
    return ratio < 0.1

def ratio_more_than_5_smaller_10(values):
    if len(values) == 0:
        return False
    ratio = np.mean(values > 5)
    return ratio < 0.1

def ratio_more_than_10_smaller_10(values):
    if len(values) == 0:
        return False
    ratio = np.mean(values > 10)
    return ratio < 0.1

def ratio_more_than_0_smaller_30(values):
    if len(values) == 0:
        return False
    ratio = np.mean(values > 0)
    return ratio < 0.3

def ratio_more_than_0_smaller_20(values):
    if len(values) == 0:
        return False
    ratio = np.mean(values > 0)
    return ratio < 0.3

if __name__ == "__main__":
    center = (500,500)
    left1 =(116,501)
    left2 = (116,514)
    right1 = (120,557)
    # right1 = (118,549)
    right2 = (122,568)
    q1 = (140,509)
    q2 = (158,530)
    q3 = (149,558)

    # 生成了周围区域和中心区域的掩码矩阵
    surroundings,centrals=build_mask(
        center=center,
        left1 = left1,
        left2 = left2,
        right1 = right1,
        right2 = right2,
        H=1000,
        W=1000,
        step=10,
        distance_min=220,
        distance_max=400,
        output_path_surrounding=r"E:\MyFiles\data\surrounding_mask.npy",
        output_path_central=r"E:\MyFiles\data\central.npy"
    )

    # from VisualizeNPY import show_single_npy_frame_ignore_percent
    #
    # for npy in surroundings:
    #     show_single_npy_frame_ignore_percent(npy=npy)
    #
    # for npy in centrals:
    #     show_single_npy_frame_ignore_percent(npy=npy)

    from VisualizeNPY import show_single_npy_frame_ignore_percent

    arr = np.load(r"E:\MyFiles\data\CAPPI0408_images_single.npy")
    result=[]

    for data in arr:
        for index in range(len(surroundings)):
            data = remove_stripes(
                surrounding=surroundings[index],
                central=centrals[index],
                data=data,
                pos_condition=ratio_more_than_0_smaller_20,    # surrounding > 0 区域的判断函数
                neg_condition=ratio_more_than_0_smaller_20,    # surrounding < 0 区域的判断函数
                assign_value=np.nan
            )
        result.append(data)

    # from VisualizeNPY import show_single_npy_frame_ignore_percent
    #
    # for index in range(len(result)):
    #     show_single_npy_frame_ignore_percent(
    #         npy=result[index]
    #     )

    np.save(r"E:\MyFiles\data\CAPPI0408_images_single_denoise2.npy",result)








