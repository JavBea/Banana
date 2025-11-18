#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：ToOneVideo.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/11/15 10:16 
"""
from moviepy import VideoFileClip, clips_array, ColorClip, CompositeVideoClip
import numpy as np

def merge_videos_grid(video_paths, a, b, output_path, fill_color=(0, 0, 0)):
    n = len(video_paths)
    assert n == a * b, f"视频数量为 {n}，但网格为 {a}x{b}={a*b}，两者必须相等"

    clips = [VideoFileClip(p) for p in video_paths]

    widths = np.array([c.w for c in clips]).reshape(a, b)
    heights = np.array([c.h for c in clips]).reshape(a, b)

    max_col_widths = widths.max(axis=0)
    max_row_heights = heights.max(axis=1)

    grid = []
    idx = 0

    for row in range(a):
        row_clips = []
        for col in range(b):
            clip = clips[idx]

            target_w = int(max_col_widths[col])
            target_h = int(max_row_heights[row])

            clip_aspect = clip.w / clip.h
            target_aspect = target_w / target_h

            # 等比例缩放
            if clip_aspect > target_aspect:
                new_w = target_w
                new_h = int(new_w / clip_aspect)
            else:
                new_h = target_h
                new_w = int(new_h * clip_aspect)

            clip_resized = clip.resized((new_w, new_h))

            # 背景色块
            bg = ColorClip(size=(target_w, target_h), color=fill_color)
            bg = bg.with_duration(clip.duration)

            # 居中叠加
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            merged = CompositeVideoClip([bg, clip_resized.with_position((x, y))])

            row_clips.append(merged)
            idx += 1

        grid.append(row_clips)

    final = clips_array(grid)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

video_files = [
    r"E:\MyFiles\data\20250408_single.mp4", r"E:\MyFiles\data\20250408_single_denoise1.mp4", r"E:\MyFiles\data\20250408_single_denoise1.v1.mp4", r"E:\MyFiles\data\20250408_single_denoise0.5.mp4"
]

merge_videos_grid(
    video_files,
    a=2,
    b=2,
    output_path=r"E:\MyFiles\data\merged.mp4",
    fill_color=(255, 255, 255)   # 白色背景
)
