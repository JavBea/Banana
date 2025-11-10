#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：CAPPI_Mapping.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/8 11:15 
"""
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy import ma
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from typing import Sequence, Tuple, Optional

def read_cappi_with_fixed_spacing(filename, start_offset=1049, segment_size=480, spacing=1502):
    """
    根据已知的间隔和起始位置提取 CAPPI 数据
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
    plt.title('CAPPI Data Visualization (Gray)')
    plt.show()



import matplotlib.pyplot as plt
import numpy as np


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
    ax.set_title('CAPPI Data Visualization (Polar)')
    plt.show()



def _safe_get_cmap(name):
    """安全获取可修改的 colormap（并设置掩码颜色后可用）。"""
    try:
        cmap = plt.get_cmap(name).copy()
    except Exception:
        cmap = plt.get_cmap(name)
    # 掩码（bad）显示为透明
    cmap.set_bad((0,0,0,0))
    return cmap

def visualize_array_polar_clockwise(
    array_2d: np.ndarray,
    bounds: Sequence[float],
    *,
    strategy: str = "linear_to_bounds",
    mask_values: Sequence[int] = (0,),
    also_mask_0x80: bool = False,
    cmap_name: str = "turbo",
    percentiles: Tuple[float,float] = (1.0, 99.0),
    quantile_bins: Optional[int] = None,
    theta_zero: str = "N",
    figsize: Tuple[float,float] = (8,8),
    show: bool = True
):
    """
    支持多策略的极坐标可视化（顺时针），并把 mask_values 对应的原始字节置为透明。
    参数说明（重点）：
      - array_2d: ndarray (segments, points) 原始字节数组（uint8 / int）
      - bounds: 分级边界（例如 reflectivity 的 [-15, -10, ... , 75]）
      - strategy: 映射策略，可选:
            'linear_to_bounds'  - 线性把 raw[非掩码范围] 映射到 [min(bounds), max(bounds)]
            'percentile_to_bounds' - 先按 percentiles 裁剪，然后线性映射到 bounds 范围（去极值）
            'equal_count_bins' - 根据 raw 的分位数得到等样本量的原始域 bin，然后把 bin 映射到 bounds 的区间（离散分级）
            'hist_eq' - 直方图均衡 -> 再映射到 bounds（增强对比）
            'raw_as_bins' - 直接把 0..255 等分到 len(bounds)-1 个区间，再按 bounds 上色
      - mask_values: 要作为缺失的原始字节值（例：(0,)），这些点会透明
      - also_mask_0x80: 是否把 0x80 也当作缺失（常见作间隔值）
      - percentiles: 用于 'percentile_to_bounds'（低, 高）
      - quantile_bins: 指定 equal_count_bins 的目标 bin 数（默认使用 len(bounds)-1）
    返回：绘图对象（若 show=True 会直接显示）
    """

    # --- 准备数据与掩码 ---
    arr = np.asarray(array_2d).astype(np.float32)
    mask_vals = set(mask_values)
    if also_mask_0x80:
        mask_vals.add(0x80)
    mask = np.isin(arr, list(mask_vals))
    masked = ma.masked_array(arr, mask=mask)

    # 物理值映射函数：把 raw -> physical（落在 bounds 范围或对应区间）
    bmin, bmax = float(min(bounds)), float(max(bounds))
    # cmap = _safe_get_cmap(cmap_name)
    cmap = ListedColormap([
      "#9c9c9c", "#767676", "#aaaaff", "#8c8cee", "#7070c9",
      "#00ffff", "#0096ff", "#0000ff", "#00ff00", "#00c800",
      "#009600", "#ffff00", "#ffc800", "#ff7800", "#ff0000",
      "#c80000", "#960000", "#ff00ff", "#9600fa", "#ffffff"
    ])

    mapped = None
    norm = None
    discrete = False  # 是否离散分级（使用 BoundaryNorm）

    # 可选策略实现
    raw_valid = masked.compressed()  # 一维有效值（排除掩码）
    if raw_valid.size == 0:
        raise ValueError("掩码后的有效数据为空，请检查 mask_values 参数。")

    strat = strategy.lower()
    if strat == "linear_to_bounds":
        vmin, vmax = float(raw_valid.min()), float(raw_valid.max())
        if vmax == vmin: vmax = vmin + 1.0
        mapped = (masked - vmin) / (vmax - vmin) * (bmax - bmin) + bmin
        norm = Normalize(vmin=bmin, vmax=bmax)

    elif strat == "percentile_to_bounds":
        lo, hi = percentiles
        vmin = float(np.nanpercentile(raw_valid, lo))
        vmax = float(np.nanpercentile(raw_valid, hi))
        if vmax == vmin: vmax = vmin + 1.0
        clipped = ma.clip(masked, vmin, vmax)
        mapped = (clipped - vmin) / (vmax - vmin) * (bmax - bmin) + bmin
        norm = Normalize(vmin=bmin, vmax=bmax)

    elif strat == "equal_count_bins":
        # 生成 quantile 边界（在 raw 值域上），bin 数目以 bounds 长度决定
        n_bins = (quantile_bins if quantile_bins is not None else (len(bounds)-1))
        if n_bins < 1:
            raise ValueError("quantile_bins 必须 >= 1")
        edges = np.quantile(raw_valid, np.linspace(0, 1, n_bins+1))
        # 小心重复边界（会导致 digitize 问题），做微小抖动
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1e-6
        # 用 digitize 得到 bin 索引，再把索引映射到 bounds 中点
        mids = 0.5 * (np.array(bounds[:-1]) + np.array(bounds[1:]))
        # 若 bins 与 bounds 数不匹配，则通过插值映射
        if len(mids) != n_bins:
            # 将每个 bin 映射到 bounds 线性插值范围
            mapped_bins = np.linspace(bmin, bmax, n_bins)
        else:
            mapped_bins = mids
        bin_indices = ma.masked_where(mask, np.digitize(masked, edges) - 1)  # 0..n_bins-1
        # clip indices
        bin_indices = ma.clip(bin_indices, 0, len(mapped_bins)-1)
        mapped = ma.masked_array(np.take(mapped_bins, bin_indices.filled(0)), mask=mask)
        # 使用离散色条
        discrete = True
        norm = BoundaryNorm(bounds, cmap.N)

    elif strat == "hist_eq":
        # 直方图均衡：把 raw 的累计分布映射到 [bmin, bmax]
        flat = raw_valid
        sort_idx = np.argsort(flat)
        ranks = np.empty_like(sort_idx)
        ranks[sort_idx] = np.arange(len(flat))
        cdf = ranks / (len(flat)-1)  # 0..1
        # 构建一个 lookup（原始值 -> fraction），先把相同值取其平均cdf
        uniq_vals, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
        # 对每个 uniq 值计算 mean rank fraction
        mean_frac = np.zeros_like(uniq_vals, dtype=np.float64)
        for i, uv in enumerate(uniq_vals):
            mean_frac[i] = np.mean(cdf[flat == uv])
        # 应用 lut
        full_frac = mean_frac[inv]
        # 把 masked 区映射回去
        lut_map = dict(zip(uniq_vals.tolist(), full_frac.tolist()))
        # 将 masked 中的每个有效位置替换为 fraction
        mapped = ma.masked_array(np.full_like(masked, np.nan, dtype=float), mask=mask)
        it = np.nditer(masked, flags=['refs_ok', 'multi_index'])
        while not it.finished:
            v = it[0].item()
            if not np.isnan(v) and (v not in mask_vals):
                mapped[it.multi_index] = lut_map.get(float(v), 0.0) * (bmax - bmin) + bmin
            it.iternext()
        norm = Normalize(vmin=bmin, vmax=bmax)

    elif strat == "raw_as_bins":
        # =========================== 最貌似的 ===========================
        # 直接把 0..255 等分为 len(bounds)-1 个区间，然后把每个区间映射到 bounds 中点
        n_bins = len(bounds) - 1
        if n_bins < 1:
            raise ValueError("bounds 长度至少应为 2")
        bin_edges = np.linspace(0, 255, n_bins+1)
        mids = 0.5 * (np.array(bounds[:-1]) + np.array(bounds[1:]))
        bin_indices = ma.masked_where(mask, np.digitize(masked, bin_edges) - 1)
        bin_indices = ma.clip(bin_indices, 0, len(mids)-1)
        mapped = ma.masked_array(np.take(mids, bin_indices.filled(0)), mask=mask)
        discrete = True
        norm = BoundaryNorm(bounds, cmap.N)

    else:
        raise ValueError(f"未知 strategy: {strategy}")

    # --- 如果 mapping 产生的是连续值，用 Normalize，否则使用 BoundaryNorm（已设） ---
    if norm is None:
        # 如果没被显式设定（大多数 continuous 策略），使用 Normalize
        norm = Normalize(vmin=bmin, vmax=bmax)

    # --- 绘图（极坐标） ---
    segments, points = array_2d.shape
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)
    Theta, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    # 以 mapped 数组绘图（它已经是物理值或分级值）
    c = ax.pcolormesh(Theta, R, mapped, shading='auto', cmap=cmap, norm=norm)

    # 顺时针 & 0度位置
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location(theta_zero)

    # colorbar 配置：离散（使用 bounds 边）与连续不同
    if discrete:
        cb = fig.colorbar(c, ax=ax, boundaries=bounds, ticks=bounds)
    else:
        cb = fig.colorbar(c, ax=ax, ticks=bounds)
    cb.set_label('Mapped value')

    ax.set_title(f'CAPPI Polar (clockwise) — strategy={strategy}')
    if show:
        plt.show()

    return fig, ax, c


filepath = r"G:\TY\20250305\CAPPI\C20250305013247.TC1"

# ===========（（ 需要注意每个文件的数据起始位置(start_offset)似乎不同 ））XXX   其实完全相同包括spacing===========
array_2d = read_cappi_with_fixed_spacing(filepath, start_offset=1039, segment_size=480, spacing=1502)
print("二维数组形状:", array_2d.shape)
# 调用可视化
visualize_array_gray(array_2d)
# 调用极坐标可视化
bounds = [ -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30,
           35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
fig, ax, c = visualize_array_polar_clockwise(array_2d, bounds,
                                             strategy='percentile_to_bounds',
                                             mask_values=(0,),
                                             also_mask_0x80=True,
                                             cmap_name='turbo',
                                             percentiles=(1,99))

# 比较多个策略
for strat in ['linear_to_bounds','percentile_to_bounds','equal_count_bins','hist_eq','raw_as_bins']:
    visualize_array_polar_clockwise(array_2d, bounds, strategy=strat,
                                   mask_values=(0,), also_mask_0x80=True, show=True)
