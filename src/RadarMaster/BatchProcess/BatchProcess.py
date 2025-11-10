#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：BatchProcess.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/15 17:02 
"""

import os
import subprocess
import shutil
import re

import numpy as np


def extract_all(archive_dir, output_dir, seven_zip_path=r"C:\Program Files\7-Zip\7z.exe", delete_after=False):
    """
    使用 7z.exe 批量解压指定目录下的所有压缩文件到目标目录

    :param archive_dir: 压缩文件所在目录
    :param output_dir: 解压目标目录
    :param seven_zip_path: 7z.exe 的完整路径
    :param delete_after: 是否在解压成功后删除原压缩包
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(archive_dir):
        file_path = os.path.join(archive_dir, filename)
        if os.path.isfile(file_path):
            print(f"正在解压: {file_path}")
            try:
                subprocess.run(
                    [seven_zip_path, "x", file_path, f"-o{output_dir}", "-y"],
                    check=True
                )
                print(f"✅ 成功解压: {filename}")

                if delete_after:
                    os.remove(file_path)
                    print(f"🗑 已删除原压缩包: {filename}")

            except subprocess.CalledProcessError as e:
                print(f"❌ 解压失败: {filename}, 错误: {e}")


# def collect_radar_files(src_root, dst_root):
#     """
#     将 src_root 下所有子文件夹中的雷达数据文件复制到 dst_root，统一平铺，不保留原路径结构
#
#     :param src_root: 源目录（含日期文件夹）
#     :param dst_root: 目标目录
#     """
#     if not os.path.exists(dst_root):
#         os.makedirs(dst_root)
#
#     for root, dirs, files in os.walk(src_root):
#         for file in files:
#             src_file = os.path.join(root, file)
#             dst_file = os.path.join(dst_root, file)
#
#             # 避免文件名冲突：如果目标目录已存在同名文件，加编号
#             if os.path.exists(dst_file):
#                 name, ext = os.path.splitext(file)
#                 i = 1
#                 while True:
#                     new_name = f"{name}_{i}{ext}"
#                     dst_file = os.path.join(dst_root, new_name)
#                     if not os.path.exists(dst_file):
#                         break
#                     i += 1
#
#             shutil.copy2(src_file, dst_file)
#             print(f"复制: {src_file} -> {dst_file}")


def collect_radar_files(src_root, dst_root, last_folder_name="*"):
    """
    将 src_root 下所有子文件夹中的指定文件夹（或 * 表示所有文件夹）里的雷达数据文件
    平铺复制到 dst_root，不保留原路径结构

    :param src_root: 源目录（含日期文件夹）
    :param dst_root: 目标目录
    :param last_folder_name: 目标子文件夹名，* 表示不特别指定，所有子文件夹都处理
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    for root, dirs, files in os.walk(src_root):
        # 只处理指定的子文件夹
        if last_folder_name == "*":
            candidate_dirs = dirs[:]  # 全部子目录
        else:
            candidate_dirs = [d for d in dirs if d == last_folder_name]

        for d in candidate_dirs:
            folder_path = os.path.join(root, d)
            for file in os.listdir(folder_path):
                src_file = os.path.join(folder_path, file)
                if not os.path.isfile(src_file):
                    continue

                dst_file = os.path.join(dst_root, file)

                # 避免文件名冲突：如果目标目录已存在同名文件，加编号
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    i = 1
                    while True:
                        new_name = f"{name}_{i}{ext}"
                        dst_file = os.path.join(dst_root, new_name)
                        if not os.path.exists(dst_file):
                            break
                        i += 1

                shutil.copy2(src_file, dst_file)
                print(f"复制: {src_file} -> {dst_file}")

def clear_folder(folder_path):
    """
    删除指定路径下的所有文件和子文件夹，但保留该文件夹本身。

    :param folder_path: 要清空的文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # 删除文件或符号链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 递归删除文件夹
        except Exception as e:
            print(f"删除失败: {item_path}, 错误: {e}")

    print(f"已清空文件夹: {folder_path}")

def copy_files_with_prefix(src_root, dst_root, prefix, move=False):
    """
    从 src_root 下的所有子文件夹中提取以 prefix 开头的文件，
    并复制到 dst_root 下相同的子文件夹层级中。

    :param src_root: 源目录
    :param dst_root: 目标目录
    :param prefix:   文件名前缀（指定字符串）
    :param move:
    """
    for root, dirs, files in os.walk(src_root):
        # 计算相对路径（保留子文件夹层级）
        rel_path = os.path.relpath(root, src_root)
        target_dir = os.path.join(dst_root, rel_path)

        for file in files:
            if file.startswith(prefix):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)

                # 确保目标文件夹存在
                os.makedirs(target_dir, exist_ok=True)

                # 复制文件
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {src_file} -> {dst_file}")

                if move:  # 如果选择移动，则删除源文件
                    os.remove(src_file)
                    print(f"已删除源文件: {src_file}")

def copy_all_files(src_dir, dst_dir, move=False):
    """
    将 src_dir 下的所有文件复制到 dst_dir 下，保留目录结构。
    :param src_dir: 源目录
    :param dst_dir: 目标目录
    :param move: True=复制后删除源文件(移动)，False=仅复制
    """
    if not os.path.exists(src_dir):
        print(f"源目录不存在: {src_dir}")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        # 计算在目标路径中的相对目录
        rel_path = os.path.relpath(root, src_dir)
        target_path = os.path.join(dst_dir, rel_path)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_path, file)
            shutil.copy2(src_file, dst_file)
            print(f"已复制: {src_file} -> {dst_file}")

            if move:  # 如果选择移动，则删除源文件
                os.remove(src_file)
                print(f"已删除源文件: {src_file}")

    # 如果是移动模式，复制完后删除空文件夹
    if move:
        for root, dirs, _ in os.walk(src_dir, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"已删除空目录: {dir_path}")


def collect_files_with_regex(src_root, dst_root, pattern, move=False, flat=True):
    """
    从 src_root 的所有子文件夹中查找文件名符合正则表达式的文件，
    并复制或剪切到 dst_root 下。

    :param src_root: 源目录
    :param dst_root: 目标目录
    :param pattern: 正则表达式 (例如 r'^data_.*\\.bin$')
    :param move: 是否剪切 (True=剪切, False=复制)
    :param flat: 是否平铺存放 (True=不保留目录结构, False=保留目录结构)
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    regex = re.compile(pattern)

    for root, dirs, files in os.walk(src_root):
        for filename in files:
            if regex.match(filename):
                src_path = os.path.join(root, filename)

                if flat:
                    dst_path = os.path.join(dst_root, filename)

                    # 避免文件名冲突
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(filename)
                        i = 1
                        while os.path.exists(dst_path):
                            dst_path = os.path.join(dst_root, f"{base}_{i}{ext}")
                            i += 1
                else:
                    # 保留目录结构
                    relative_path = os.path.relpath(root, src_root)
                    target_dir = os.path.join(dst_root, relative_path)
                    os.makedirs(target_dir, exist_ok=True)
                    dst_path = os.path.join(target_dir, filename)

                # 执行复制或剪切
                if move:
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {src_path} -> {dst_path}")
                else:
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path} -> {dst_path}")


import cv2
import os
from natsort import natsorted  # 用于自然排序文件名


def imread_unicode(path):
    """支持中文路径的读取图片"""
    with open(path, 'rb') as f:
        data = f.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def images_to_video(image_folder, output_path, fps=25, size=None):
    """
    将指定文件夹下的图片整理为MP4视频，支持中文路径
    """
    # 获取图片列表并排序
    images = [img for img in os.listdir(image_folder)
              if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not images:
        raise ValueError("指定文件夹中没有图片！")

    images = natsorted(images)

    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = imread_unicode(first_image_path)
    if frame is None:
        raise ValueError(f"无法读取图片: {first_image_path}")

    if size is None:
        height, width = frame.shape[:2]
        size = (width, height)
    else:
        frame = cv2.resize(frame, size)

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    # 写入视频帧
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = imread_unicode(img_path)
        if frame is None:
            print(f"警告: 跳过无法读取的图片 {img_path}")
            continue
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"视频已保存至: {output_path}")

if __name__ == "__main__":

    # # copy 收集原压缩包文件
    # src_root = r"G:\TY" #原路径
    # dst_root = r"E:\MyFiles\output\next_batch" #目标路径（软件监控路径）
    # collect_radar_files(src_root, dst_root, last_folder_name="VAD")


    # # copy 收集原压缩包文件
    # src_root = r"G:\TY" #原路径
    # dst_root = r"E:\MyFiles\output\next_batch" #目标路径（软件监控路径）
    # collect_radar_files(src_root, dst_root, last_folder_name="VDL")


    # # 解压缩
    # archive_directory = r"E:\MyFiles\radar_data_by_types\CAPPI"        # 存放压缩包的目录
    # output_directory = r"E:\MyFiles\radar_data_by_types\CAPPI"     # 解压到目标目录
    # extract_all(archive_directory, output_directory,delete_after=True)

    # 移动文件
    # temp_directory = r"G:\杂项\temp"
    # watch_directory = r"E:\MyFiles\output\data2"
    # copy_all_files(temp_directory, watch_directory,move=True)



    # 提取固定类型的数据文件
    src_root = r"E:\MyFiles\output\photo"  # 源目录
    dst_root = r"C:\Users\Me\Desktop\CAPPI"  # 目标目录
    prefix = "CAPPI"  # 指定前缀

    copy_files_with_prefix(src_root, dst_root, prefix, move=True)


    # # 提取固定正则表达式格式的数据文件
    # collect_files_with_regex(
    #     src_root=r"G:\TY\20250408\CAPPI",
    #     dst_root=r"E:\MyFiles\radar_data_by_types\CAPPI",
    #     pattern=r".*TC1\.zip$",
    #     move=False,
    #     flat=True
    # )

    # images_to_video(image_folder=r"G:\杂项\CAPPI0408",
    #                 output_path=r"G:\杂项\CAPPI0408.mp4")



