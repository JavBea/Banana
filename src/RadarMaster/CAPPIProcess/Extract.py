import os
import numpy as np
from PIL import Image

def collect_images_to_npy(root_dir, output_path, img_size=None, exts=('.png', '.jpg', '.jpeg', '.bmp')):
    """
    从多个子文件夹中收集图片并保存为一个 npy 文件。
    每个文件夹中只取 20 的整数倍数量的图片，多余的舍弃。

    参数:
        root_dir (str): 根目录路径，包含若干子文件夹。
        output_path (str): 输出 npy 文件路径。
        img_size (tuple or None): 若指定，则将图片 resize 到该尺寸 (W, H)。
        exts (tuple): 支持的图片扩展名。
    """
    all_images = []

    # 遍历所有子文件夹
    for subdir in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        # 获取当前文件夹的所有图片
        imgs = sorted([
            os.path.join(subpath, f)
            for f in os.listdir(subpath)
            if f.lower().endswith(exts)
        ])

        num_imgs = len(imgs)
        usable_num = (num_imgs // 20) * 20  # 取20的整数倍
        if usable_num == 0:
            print(f"跳过 {subdir}，不足20张图片。")
            continue

        imgs = imgs[:usable_num]
        print(f"正在处理 {subdir}: 使用 {usable_num}/{num_imgs} 张图片")

        # 读取图片
        for img_path in imgs:
            img = Image.open(img_path).convert('RGB')
            if img_size:
                img = img.resize(img_size, Image.BILINEAR)
            img_array = np.array(img, dtype=np.uint8)
            all_images.append(img_array)

    if not all_images:
        print("未找到任何图片！")
        return

    # 合并并保存
    data = np.stack(all_images)
    np.save(output_path, data)
    print(f"\n✅ 成功保存 {data.shape[0]} 张图片到 {output_path}")
    print(f"图片数据形状: {data.shape}")  # (N, H, W, 3)

# ===== 使用示例 =====
if __name__ == "__main__":
    collect_images_to_npy(
        root_dir="E:\MyFiles\data\CAPPI0010",     # 主目录路径
        output_path="E:\MyFiles\data\origin_CAPPI0010.npy", # 输出 npy 文件
        img_size=(64, 64)                # 可选：统一图片尺寸
    )
