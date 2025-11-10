#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：FolderMonitor.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/9/16 15:35 
"""
import os
import time
from datetime import datetime

def monitor_directory(folder_path, log_file="log.txt", stable_minutes=5, check_interval=10,title=None):
    """
    持续监听文件夹，如果 stable_minutes 内文件数量不变，则记录时间到 log_file。

    :param folder_path: 要监听的文件夹路径
    :param log_file: 记录结果的 txt 文件
    :param stable_minutes: 判定稳定的时间（分钟）
    :param check_interval: 检查间隔（秒）
    :param title: 自定义标题
    """
    last_count = None
    last_change_time = time.time()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=== 文件夹监听开始: {} ===\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+title)

    print(f"开始监听 {folder_path} (稳定时间={stable_minutes}分钟, 检查间隔={check_interval}秒) ...")

    while True:
        try:
            # 当前文件数量
            current_count = len(os.listdir(folder_path))

            if last_count is None:
                last_count = current_count
                last_change_time = time.time()
            elif current_count != last_count:
                # 数量变化，刷新计时器
                last_count = current_count
                last_change_time = time.time()
            else:
                # 数量未变化，检查时间
                if time.time() - last_change_time >= stable_minutes * 60:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    msg = f"[{now_str}] 文件夹 {folder_path} 在 {stable_minutes} 分钟内无变化，当前文件数: {current_count}"
                    print(msg)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

                    # 为了避免重复写日志，刷新计时器
                    last_change_time = time.time()

            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\n监听已手动停止。")
            break
        except Exception as e:
            print("错误:", e)
            time.sleep(check_interval)

import os

def get_folder_size(folder_path: str) -> int:
    """计算文件夹总大小（字节）"""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def folder_not_empty(folder_path: str) -> bool:
    """判断文件夹是否非空"""
    return any(os.scandir(folder_path))

def monitor_folder_size(folder_path, log_file, stable_minutes=5, check_interval=10,
                        callback=None, title=None):
    """
    监听文件夹大小，如果 stable_minutes 内大小无变化 且 文件夹不为空，
    则记录日志并执行回调函数。
    :param folder_path: 要监听的文件夹路径
    :param log_file: 日志文件路径
    :param stable_minutes: 稳定时间（分钟）
    :param check_interval: 检查间隔（秒）
    :param callback: 条件满足时执行的函数
    :param title: 日志标题
    """
    last_size = None
    last_change_time = time.time()

    with open(log_file, "a", encoding="utf-8") as f:
        header = "=== 文件夹大小监听开始: {} ===".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if title:
            header += f" | {title}"
        f.write(header + "\n")

    print(f"开始监听 {folder_path} (稳定时间={stable_minutes}分钟, 检查间隔={check_interval}秒) ...")

    while True:
        try:
            current_size = get_folder_size(folder_path)

            if last_size is None:
                last_size = current_size
                last_change_time = time.time()
            elif current_size != last_size:
                # 文件夹大小有变化
                last_size = current_size
                last_change_time = time.time()
            else:
                # 大小无变化，检查是否达到稳定时间且文件夹非空
                if (time.time() - last_change_time >= stable_minutes * 60):
                    if folder_not_empty(folder_path):
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        msg = f"[{now_str}] 文件夹 {folder_path} 在 {stable_minutes} 分钟内大小无变化，当前大小: {current_size/1024/1024:.2f} MB"
                        print(msg)
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(msg + "\n")

                        # 执行回调函数（如果提供）
                        if callback:
                            try:
                                callback[0]()
                            except Exception as cb_err:
                                print(f"⚠️ 回调函数1执行出错: {cb_err}")

                        # 避免重复触发
                        last_change_time = time.time()
                    else:
                        if callback:
                            try:
                                callback[1]()
                            except Exception as cb_err:
                                print(f"⚠️ 回调函数执行2出错: {cb_err}")

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n监听已手动停止。")
            break
        except Exception as e:
            print("错误:", e)
            time.sleep(check_interval)

from BatchProcess import copy_all_files
def refresh_folder(src_folder=r"E:\MyFiles\output\src3", temp_folder=r"E:\MyFiles\output\data2"):
    """
        将数据文件移动出去再移动回来
        :param src_folder: 监视目录
        :param temp_folder: 中转目录
    """
    copy_all_files(src_folder, temp_folder,move=True)
    copy_all_files(temp_folder, src_folder,move=True)

def put_next_batch(next_batch_folder=r"E:\MyFiles\output\RVD", watch_folder=r"E:\MyFiles\output\src3"):
    """
        将下一批数据文件移动进监视目录
        :param next_batch_folder: 下一批数据文件存放目录
        :param watch_folder: 监视目录
    """
    copy_all_files(next_batch_folder, watch_folder, move=True)


if __name__ == "__main__":
    # monitor_directory(r"E:\MyFiles\output\src", log_file="src_listener_log.txt", stable_minutes=5, check_interval=10)
    monitor_folder_size(folder_path=r"E:\MyFiles\output\src3",
                        log_file="src_size_listener_log.txt",
                        stable_minutes=30,
                        check_interval=60,
                        callback=[refresh_folder, put_next_batch],
                        title="CAPPI")

