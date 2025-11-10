#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：RadarMaster.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/4 19:22 
"""

import json
from matplotlib.colors import ListedColormap, BoundaryNorm


def get_cmap(cmap_type="velocity", config_path="colormaps.json"):
    """
    根据类型返回 (cmap, norm, bounds)
    从 colormaps.json 获得某个雷达数据类型对应的颜色映射
    cmap_type: "velocity", "reflectivity", "spectrum_width" 等
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cmap_config = json.load(f)

    if cmap_type not in cmap_config:
        raise ValueError(f"未知的 cmap_type: {cmap_type}")

    colors = cmap_config[cmap_type]["colors"]
    bounds = cmap_config[cmap_type]["bounds"]

    cmap = ListedColormap(colors, name=f"{cmap_type}_custom")
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

