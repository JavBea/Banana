#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：BatchExtract.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/8 9:38 
"""
import os
import subprocess
import zipfile
import rarfile
import py7zr
from pathlib import Path

def extract_files(input_folder, output_folder=None, create_subfolder=True, password=None):
    """
    解压指定文件夹下的所有 zip, rar, 7z 文件
    :param input_folder: 输入的文件夹路径
    :param output_folder: 输出文件夹路径, 如果为空, 默认为原路径
    :param create_subfolder: 是否根据文件名创建子文件夹解压, 默认为 True
    :param password: 解压密码, 默认为 None
    """
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        print(f"错误: {input_folder} 不是有效的文件夹路径.")
        return

    if output_folder is None:
        output_folder = input_folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取所有压缩包
    files_to_extract = []
    for file in input_folder.iterdir():
        if file.suffix.lower() in ['.zip', '.rar', '.7z']:
            files_to_extract.append(file)

    if not files_to_extract:
        print("没有找到任何压缩包文件.")
        return

    for file in files_to_extract:
        print(f"正在解压 {file}...")
        if create_subfolder:
            subfolder = output_folder / file.stem
            subfolder.mkdir(parents=True, exist_ok=True)
        else:
            subfolder = output_folder

        # 解压缩文件
        if file.suffix.lower() == '.zip':
            extract_zip(file, subfolder, password)
        elif file.suffix.lower() == '.rar':
            extract_rar(file, subfolder, password)
        elif file.suffix.lower() == '.7z':
            extract_7z(file, subfolder, password)
        else:
            print(f"不支持的文件格式: {file.suffix}.")

def extract_zip(zip_file, output_folder, password=None):
    """解压ZIP文件"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if password:
                zip_ref.setpassword(password.encode())
            zip_ref.extractall(output_folder)
        print(f"ZIP文件 {zip_file.name} 解压完成.")
    except Exception as e:
        print(f"解压ZIP文件 {zip_file.name} 失败: {e}")

def extract_rar(rar_file, output_folder, password=None):
    """解压RAR文件"""
    try:
        with rarfile.RarFile(rar_file, 'r') as rar_ref:
            if password:
                rar_ref.setpassword(password)
            rar_ref.extractall(output_folder)
        print(f"RAR文件 {rar_file.name} 解压完成.")
    except Exception as e:
        print(f"解压RAR文件 {rar_file.name} 失败: {e}")

def extract_7z(archive_file, output_folder, password=None):
    """解压7Z文件"""
    try:
        cmd = ['7z', 'x', str(archive_file), f'-o{output_folder}']
        if password:
            cmd.extend([f'-p{password}'])
        subprocess.run(cmd, check=True)
        print(f"7Z文件 {archive_file.name} 解压完成.")
    except subprocess.CalledProcessError as e:
        print(f"解压7Z文件 {archive_file.name} 失败: {e}")

# 示例用法
input_folder = r"C:\Users\Me\Desktop\雷达数据\20250228\CAPPI"  # 设置你的文件夹路径

extract_files(input_folder)

