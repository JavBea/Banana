import os
import re
import cv2
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
    将多天的彩色雷达图像整理为 Navier-Stokes 风格数据：
    a (N, H, W)
    u (N, H, W, T)
    t (1, T)

    修改版：
    - 不再固定选取 seq_len 帧
    - 若某文件夹帧数不是 seq_len 的整数倍，舍弃余数
    """
    day_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    all_u = []
    t_common = np.arange(0, seq_len * step_minutes, step_minutes).astype(np.float32)

    for day in day_folders:
        folder = os.path.join(root_dir, day)
        files = sorted(os.listdir(folder))
        times, imgs = [], []

        # 解析时间戳并读取图片
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            match = re.search(r'CAPPI-(\d{12})', f)
            if not match:
                continue

            time_str = match.group(1)
            try:
                t = datetime.strptime(time_str, "%Y%m%d%H%M")
            except ValueError:
                continue

            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 转灰度 + resize + 归一化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, img_size, interpolation=cv2.INTER_LANCZOS4)
            gray = gray.astype(np.float32) / 255.0
            imgs.append(gray)
            times.append(t)

        # 排序并转为数组
        if len(imgs) < seq_len:
            continue
        times, imgs = zip(*sorted(zip(times, imgs)))
        imgs = np.stack(imgs, axis=0)  # (T_actual, H, W)
        total_frames = len(imgs)

        # 计算可整除的样本数量
        num_samples = total_frames // seq_len
        usable_frames = num_samples * seq_len

        if usable_frames < seq_len:
            continue

        # 裁切并按块分组
        imgs = imgs[:usable_frames, :, :]  # 舍弃余数帧
        imgs = imgs.reshape(num_samples, seq_len, *img_size)  # (N_day, T, H, W)

        all_u.append(imgs)

        print(f"📅 {day}: {total_frames} 帧 → {num_samples} 个样本")

    if not all_u:
        print("❌ 没有有效数据，检查路径或文件名格式。")
        return

    # 合并所有天
    u = np.concatenate(all_u, axis=0)  # (N, T, H, W)
    u = np.transpose(u, (0, 2, 3, 1))  # -> (N, H, W, T)
    a = u[..., 0]                      # 每个样本的第一帧
    t = t_common[None, :]

    savemat(output_path, {"a": a, "u": u, "t": t})
    print(f"\n✅ 已保存至 {output_path}")
    print(f"a: {a.shape}, u: {u.shape}, t: {t.shape}")


# 示例运行
build_radar_dataset_from_color(
    root_dir=r"E:\MyFiles\data\CAPPI0010",
    output_path=r"E:\MyFiles\data\CAPPI0010_2.mat",
    step_minutes=10,
    seq_len=20,
    img_size=(64, 64)
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
