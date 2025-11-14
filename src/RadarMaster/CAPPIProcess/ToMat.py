import os
import re
import cv2
import numpy as np
from datetime import datetime
from scipy.io import savemat

import os, re, cv2
import numpy as np
from datetime import datetime
from scipy.io import savemat

def build_radar_dataset_from_color(
    root_dir,
    output_path,
    step_minutes=10,
    seq_len=20,
    img_size=(64, 64)
):
    """
    从多天彩色雷达图像构建 Navier-Stokes 风格数据：
      - 每天子文件夹中包含若干 CAPPI-YYYYMMDDHHMM.png/jpg 图片
      - 使用“近似颜色匹配”将 RGB 转为数值单通道
      - 每天按 seq_len 分块，不足整数倍则舍弃
      - 负数与 NaN 均替换为 0
      - 输出字段：
            a: (N,H,W)
            u: (N,H,W,T)
            t: (1,T)
    """

    # === 定义颜色映射表 ===
    hex_colors = [
        "#000000",  # 黑色，对应 NaN
        "#9c9c9c","#767676","#aaaaff","#8c8cee","#7070c9","#00ffff","#0096ff","#0000ff",
        "#00ff00","#00c800","#009600","#ffff00","#ffc800","#ff7800","#ff0000",
        "#c80000","#960000","#ff00ff","#9600fa"
    ]
    nums = np.array(
        [np.nan, -15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75],
        dtype=np.float32
    )
    color_map = np.array(
        [[int(h[i:i+2], 16) for i in (1,3,5)] for h in hex_colors],
        dtype=np.float32
    )

    def rgb_to_value(img_rgb):
        """
        将RGB图像(浮点)转为数值图。
        使用欧几里得距离找最近色。
        """
        H, W, _ = img_rgb.shape
        flat = img_rgb.reshape(-1, 3).astype(np.float32)
        diffs = flat[:, None, :] - color_map[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        return nums[nearest_idx].reshape(H, W)

    # === 时间步向量 ===
    t_common = np.arange(0, seq_len * step_minutes, step_minutes).astype(np.float32)

    # === 扫描文件夹 ===
    day_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    all_u = []

    for day in day_folders:
        folder = os.path.join(root_dir, day)
        files = sorted(os.listdir(folder))
        times, imgs = [], []

        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            match = re.search(r'CAPPI-(\d{12})', f)
            if not match:
                continue

            try:
                t = datetime.strptime(match.group(1), "%Y%m%d%H%M")
            except ValueError:
                continue

            path = os.path.join(folder, f)
            img = cv2.imread(path)
            if img is None:
                continue

            # === 转为RGB (OpenCV默认BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LANCZOS4)

            # === 转数值图 ===
            value_map = rgb_to_value(img_rgb)

            # === 替换负数和NaN为0 ===
            value_map = np.nan_to_num(value_map, nan=0.0)
            value_map[value_map < 0] = 0.0

            imgs.append(value_map)
            times.append(t)

        # 排序并打包
        if len(imgs) < seq_len:
            continue
        times, imgs = zip(*sorted(zip(times, imgs)))
        imgs = np.stack(imgs, axis=0)
        total_frames = len(imgs)

        # 分块
        num_samples = total_frames // seq_len
        usable_frames = num_samples * seq_len
        if usable_frames < seq_len:
            continue

        imgs = imgs[:usable_frames].reshape(num_samples, seq_len, *img_size)
        all_u.append(imgs)
        print(f"📅 {day}: {total_frames} 帧 → {num_samples} 个样本")

    if not all_u:
        print("❌ 没有有效数据，检查路径或文件名格式。")
        return

    # === 合并输出 ===
    u = np.concatenate(all_u, axis=0)   # (N, T, H, W)
    u = np.transpose(u, (0, 2, 3, 1))   # (N, H, W, T)
    a = u[..., 0]
    t = t_common[None, :]

    savemat(output_path, {"a": a, "u": u, "t": t})
    print(f"\n✅ 已保存至 {output_path}")
    print(f"a: {a.shape}, u: {u.shape}, t: {t.shape}")

# 构造数据集
size = 256
seq_len = 20
step_minutes = 10
build_radar_dataset_from_color(
    root_dir=r"E:\MyFiles\data\CAPPI0010",
    output_path=rf"E:\MyFiles\data\CAPPI0010_singled_{size}px.mat",
    step_minutes=step_minutes,
    seq_len=seq_len,
    img_size=(size, size)
)


import os
import numpy as np
from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def gather_images_in_folder(folder_path):
    """返回该文件夹中所有支持的图片路径（按文件名排序）"""
    files = sorted(os.listdir(folder_path))
    paths = [
        os.path.join(folder_path, f)
        for f in files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    return paths


def open_and_convert(img_path, target_size=None):
    """打开图片并转换为RGB；若指定 target_size=(H,W)，则resize"""
    with Image.open(img_path) as im:
        im = im.convert('RGB')
        if target_size is not None:
            im = im.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


def build_dataset_to_npy(root_dir, output_dir, seq_len=20, resize_to_first=True):
    """
    遍历 root_dir 下的所有子文件夹，
    每个子文件夹中提取 20 的整数倍的图片，
    统一尺寸并保存为 .npy 文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    subfolders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    print(f"发现 {len(subfolders)} 个子文件夹。")

    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        image_paths = gather_images_in_folder(folder_path)
        total = len(image_paths)
        usable = (total // seq_len) * seq_len  # 只取20的整数倍

        if usable == 0:
            print(f"跳过：{folder}（图片数 {total} < {seq_len}）")
            continue

        print(f"\n处理文件夹：{folder}，共 {total} 张图片，可用 {usable} 张。")

        # 读取第一张图片，确定尺寸
        first_img = open_and_convert(image_paths[0])
        H, W, C = first_img.shape
        target_size = (H, W) if resize_to_first else None

        # 预分配数组
        data = np.zeros((usable, H, W, 3), dtype=np.uint8)

        # 写入第一张
        data[0] = first_img

        # 逐张读取
        for i in tqdm(range(1, usable), desc=f"{folder}"):
            img = open_and_convert(image_paths[i], target_size=target_size)
            if img.shape != (H, W, 3):
                # 保险起见再强制 resize
                img = np.asarray(
                    Image.fromarray(img).resize((W, H), Image.LANCZOS),
                    #Image.fromarray(img).resize((W, H), Image.BILINEAR),
                    dtype=np.uint8
                )
            data[i] = img

        # 保存为 .npy
        save_path = os.path.join(output_dir, f"{folder}.npy")
        np.save(save_path, data)
        print(f"✅ 已保存到: {save_path}, shape={data.shape}, dtype={data.dtype}")

    print("\n全部完成。")


# 示例调用：
#if __name__ == "__main__":
#    root_dir = r"E:\MyFiles\data\CAPPI0010"      # 改成你的输入文件夹
#    output_dir = r"E:\MyFiles\data\t"    # 改成输出文件夹
#    build_dataset_to_npy(root_dir, output_dir)
