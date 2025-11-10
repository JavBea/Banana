#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：Test.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/10/17 10:27 
"""


from io import BytesIO
import cv2
import glob
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os
import shutil
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_pixel_differences(images, save_path=None):
    """
    计算一批图像中每两个相邻图像的像素差（绝对值差）
    每个像素会存储一个时间序列，表示它在不同帧间的变化情况。

    参数:
        images: np.ndarray, shape = (N, H, W)
                N帧灰度图（或单通道数据）
    返回:
        diffs: np.ndarray, shape = (N-1, H, W)
               每一帧的相邻差
    """
    images = np.asarray(images, dtype=np.float32)
    diffs = np.abs(np.diff(images, axis=0))  # 计算相邻差

    if save_path:
        np.save(save_path, diffs)
        print(f"✅ 差分结果已保存到: {save_path}")
        print(f"  文件形状: {diffs.shape}, 数据类型: {diffs.dtype}")

    return diffs

def visualize_pixel_distribution(diffs, x, y):
    """
    对单个像素点 (x, y) 的差值分布进行可视化，并返回该图的图像帧（RGB数组）。

    参数:
        diffs: np.ndarray, shape = (N-1, H, W)
        x, y: 像素坐标

    返回:
        frame: np.ndarray, shape=(H, W, 3)，RGB格式图像，可用于视频合成
    """
    pixel_diffs = diffs[:, y, x]

    # 绘制图形
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pixel_diffs, bins=50, color='gray')
    axes[0].set_title(f"Pixel ({x},{y}) diff histogram")
    axes[0].set_xlabel("Difference value")
    axes[0].set_ylabel("Count")

    axes[1].plot(pixel_diffs, marker='o', linewidth=0.8)
    axes[1].set_title(f"Pixel ({x},{y}) temporal diff series")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Abs diff")

    plt.tight_layout()

    # 将 Matplotlib 图像保存到内存中
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)

    # 转为 PIL 图像再转成 numpy 数组
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    frame = np.array(img)
    buf.close()

    return frame

def visualize_pixel_distribution_video(frames,fps=10,output_path=r"E:\MyFiles\Projects\Banana\output",save_mode="video"):
    """
    将一系列可视化帧保存为视频或单张图片。

    参数:
        frames: list[np.ndarray]
            图像帧列表，每帧为 RGB 格式 (H, W, 3)
        fps: int
            保存为视频时的帧率
        output_path: str
            输出文件夹路径（不含文件名）
        save_mode: str
            "video" 保存为视频文件 (.mp4)
            "images" 保存为一系列 PNG 图片
    """
    os.makedirs(output_path, exist_ok=True)

    h, w, _ = frames[0].shape

    if save_mode == "video":
        # 视频输出路径
        video_path = os.path.join(output_path, "pixel_diff_visualization.mp4")

        # 创建视频写入器
        out = cv2.VideoWriter(video_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (w, h))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"✅ 视频已生成：{video_path}")

    elif save_mode == "images":
        image_dir = os.path.join(output_path, "pixel_diff_frames")
        os.makedirs(image_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(image_dir, f"frame_{i:04d}.png")
            cv2.imwrite(img_path, frame_bgr)

        print(f"✅ 共保存 {len(frames)} 张图像到：{image_dir}")

    else:
        raise ValueError("❌ 参数 save_mode 只能是 'video' 或 'images'")

def visualize_global_statistics(diffs):
    """
    显示每个像素的平均差值和标准差热力图。
    """
    mean_map = np.mean(diffs, axis=0)
    std_map = np.std(diffs, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(mean_map, cmap='hot')
    axes[0].set_title("Mean difference per pixel")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(std_map, cmap='hot')
    axes[1].set_title("Std deviation of difference per pixel")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def visualize_global_histogram(diffs):
    """
    统计所有像素的差值分布直方图。
    """
    plt.figure(figsize=(6,4))
    plt.hist(diffs.flatten(), bins=200, color='gray')
    plt.title("Global pixel difference distribution")
    plt.xlabel("Abs diff value")
    plt.ylabel("Count")
    plt.show()

def extract_pixel_features(diffs):
    """
    提取每个像素逐帧差分分布的统计特征
    输入: diffs.shape = (N-1, H, W)
    输出: features.shape = (H*W, F)
    """
    N, H, W = diffs.shape
    data = diffs.reshape(N, -1)  # 每列是一个像素
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    skewness = skew(data, axis=0)
    kurt = kurtosis(data, axis=0)
    p10 = np.percentile(data, 10, axis=0)
    p50 = np.percentile(data, 50, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    features = np.stack([mean, std, skewness, kurt, p10, p50, p90], axis=1)
    return features  # (H*W, 7)

def cluster_pixel_distributions(features, n_clusters=4):
    """
    对像素分布特征进行聚类
    返回每个像素的类别标签
    """
    # 填充 NaN
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    return labels, kmeans, scaler

def visualize_clusters(labels, H, W, n_clusters):
    label_map = labels.reshape(H, W)
    cmap = plt.get_cmap('tab10', n_clusters)  # 离散 colormap
    bounds = np.arange(n_clusters + 1) - 0.5  # 让每个整数对应一个区间
    norm = BoundaryNorm(bounds, n_clusters)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(label_map, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=np.arange(n_clusters))
    cbar.set_label("Cluster ID")
    plt.title("Pixel Distribution Clusters")
    plt.show()

def find_representative_pixels(features, labels, kmeans, scaler, top_k=5):
    """
    寻找每个聚类中心最近的 Top-K 个样本索引（代表像素）

    参数:
        features: np.ndarray, shape = (num_pixels, num_features)
        labels: np.ndarray, shape = (num_pixels,)
        kmeans: 已训练好的 KMeans 模型
        scaler: 用于特征标准化的 StandardScaler
        top_k: 每个聚类中选择的代表像素数量

    返回:
        representatives: dict {cluster_id: [pixel_idx1, pixel_idx2, ...]}
    """
    # 确保无 NaN
    features_scaled = scaler.transform(np.nan_to_num(features))
    centers = kmeans.cluster_centers_

    representatives = {}
    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        # 计算该类中所有样本到中心的距离
        cluster_feats = features_scaled[cluster_indices]
        distances = np.linalg.norm(cluster_feats - centers[i], axis=1)

        # 选出距离最小的 top_k
        top_k_indices = np.argsort(distances)[:top_k]
        representatives[i] = cluster_indices[top_k_indices].tolist()

    return representatives

def visualize_representative_distributions(diffs, representatives, H, W, max_per_cluster=3):
    """
    对每个聚类的代表像素（可多个）绘制直方图 + 时间序列
    """
    diffs_flat = diffs.reshape(diffs.shape[0], -1)

    for cluster_id, pixel_list in representatives.items():
        for pixel_idx in pixel_list[:max_per_cluster]:  # 控制最多显示几个
            pixel_diffs = diffs_flat[:, pixel_idx]
            y, x = divmod(pixel_idx, W)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(pixel_diffs, bins=50, color='gray', alpha=0.8)
            axes[0].set_title(f"Cluster {cluster_id} | Pixel ({x},{y}) Distribution")
            axes[0].set_xlabel("Difference Value")
            axes[0].set_ylabel("Frequency")

            axes[1].plot(pixel_diffs, marker='o', linewidth=0.8, markersize=3)
            axes[1].set_title(f"Pixel ({x},{y}) Temporal Diff Series")
            axes[1].set_xlabel("Frame Index")
            axes[1].set_ylabel("Abs Diff")

            plt.tight_layout()
            plt.show()

def cluster_pixel_diffs(diff_maps, n_clusters=5, top_k=5, save_topk_dir=None, use_pca=True, random_state=42):
    """
    对像素逐帧差进行聚类，并可选保存每个聚类的 top-k 图像。

    参数：
        diff_maps: np.ndarray, shape (N, H, W)
            每帧的逐像素差分数据。
        n_clusters: int
            聚类个数。
        top_k: int
            每个聚类保存的 top-k 最接近中心的样本。
        save_topk_dir: str or None
            若非 None，则自动创建文件夹并保存每个聚类的 top-k 图。
        use_pca: bool
            是否使用 PCA 将高维数据降维后聚类。
        random_state: int
            随机种子，保证可复现。
    """
    N, H, W = diff_maps.shape
    flattened = diff_maps.reshape(N, -1)

    if use_pca:
        print(">>> 使用 PCA 降维中...")
        pca = PCA(n_components=50, random_state=random_state)
        features = pca.fit_transform(flattened)
    else:
        features = flattened

    print(">>> 聚类中...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features)

    # 可视化聚类分布
    plt.figure(figsize=(6, 5))
    plt.hist(labels, bins=np.arange(n_clusters + 1) - 0.5, rwidth=0.8)
    plt.xticks(range(n_clusters))
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.title("Pixel Diff Cluster Distribution")
    plt.show()

    if save_topk_dir is not None:
        os.makedirs(save_topk_dir, exist_ok=True)
        centers = kmeans.cluster_centers_

        print(">>> 保存每个聚类的 top-k 图像...")
        for cid in range(n_clusters):
            cluster_indices = np.where(labels == cid)[0]
            cluster_features = features[cluster_indices]

            # 计算与聚类中心的距离
            dists = np.linalg.norm(cluster_features - centers[cid], axis=1)
            topk_idx = cluster_indices[np.argsort(dists)[:top_k]]

            cluster_dir = os.path.join(save_topk_dir, f"cluster_{cid}")
            os.makedirs(cluster_dir, exist_ok=True)

            for rank, idx in enumerate(topk_idx):
                img = (diff_maps[idx] / diff_maps[idx].max() * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(cluster_dir, f"top{rank + 1}.png"))

            print(f"  Cluster {cid}: saved top-{top_k} images")

    print(">>> 完成 ✅")
    return labels

def denoise_images_by_clusters(images, diffs, labels, H, W, cluster_thresholds, save_dir=None):
    """
    根据聚类结果和阈值对原始图像进行去噪。

    参数：
        images: np.ndarray, shape=(N, H, W)
            原始图像序列。
        diffs: np.ndarray, shape=(N-1, H, W)
            相邻帧差分。
        labels: np.ndarray, shape=(H*W,)
            每个像素的聚类标签。
        H, W: int
            图像尺寸。
        cluster_thresholds: dict
            每个聚类的阈值区间，例如 {0: (low0, high0), 1: (low1, high1), ...}
        save_dir: str or None
            若指定则保存去噪后结果。

    返回：
        cleaned_images: np.ndarray, shape=(N, H, W)
            去噪后的图像序列。
    """
    cleaned_images = images.copy()
    label_map = labels.reshape(H, W)

    # 遍历每一帧（从第1帧开始，因为diffs对应的是 frame[i] 与 frame[i-1]）
    for i in range(1, images.shape[0]):
        diff_frame = diffs[i - 1]
        img = cleaned_images[i]

        # 对每个聚类单独处理
        for cid, (low, high) in cluster_thresholds.items():
            mask_cluster = (label_map == cid)
            mask_noise = (diff_frame < low) | (diff_frame > high)
            mask = mask_cluster & mask_noise

            img[mask] = 0  # 噪声像素置黑

        cleaned_images[i] = img

    # 保存结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(cleaned_images):
            img_path = os.path.join(save_dir, f"denoised_{i:04d}.png")
            Image.fromarray((frame / frame.max() * 255).astype(np.uint8)).save(img_path)
        print(f"✅ 去噪结果已保存至: {save_dir}")

    return cleaned_images

def denoise_by_cluster_thresholds_color(original_images, diffs, labels, cluster_thresholds, save_dir=None):
    """
    对彩色图像逐帧去噪（每个聚类分别设阈值）
    - original_images: numpy 数组, shape (N, H, W, 3)
    - diffs: numpy 数组, shape (N-1, H, W)
    - labels: 聚类标签, shape (H*W,)
    - cluster_thresholds: dict, 形如 {cluster_id: (low_th, high_th)}
    - save_dir: 若指定，则保存去噪后的帧到该目录
    """

    N, H, W, C = original_images.shape
    denoised = original_images.copy().astype(np.uint8)

    # 展开 label 以便索引
    labels_2d = labels.reshape(H, W)

    for frame_idx in range(N - 1):
        diff = diffs[frame_idx]

        for cluster_id, (low_th, high_th) in cluster_thresholds.items():
            mask = (labels_2d == cluster_id) & ((diff < low_th) | (diff > high_th))
            denoised[frame_idx + 1][mask] = [0, 0, 0]  # 彩色黑色

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{frame_idx+1:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(denoised[frame_idx + 1], cv2.COLOR_RGB2BGR))

    return denoised

def copy_same_name_files(src_dir, dst_dir, target_filename):
    """
    从 src_dir（包括子目录）复制所有名为 target_filename 的文件到 dst_dir，
    并在复制时在文件名前加上相对路径前缀（将路径分隔符替换为下划线）。

    参数：
        src_dir (str): 源文件夹路径
        dst_dir (str): 目标文件夹路径
        target_filename (str): 要查找的文件名（例如 "config.json"）
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"源路径不存在或不是文件夹: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        if target_filename in files:
            # 原文件路径
            src_file = os.path.join(root, target_filename)

            # 计算相对路径前缀
            rel_path = os.path.relpath(root, src_dir)
            rel_prefix = rel_path.replace(os.sep, '_')  # 替换路径分隔符为下划线
            if rel_prefix == '.':
                rel_prefix = ''  # 根目录下不用前缀

            # 构造目标文件名
            dst_filename = f"{rel_prefix + '_' if rel_prefix else ''}{target_filename}"
            dst_file = os.path.join(dst_dir, dst_filename)

            # 执行复制
            shutil.copy2(src_file, dst_file)
            print(f"✅ 已复制: {src_file} → {dst_file}")

    print("\n全部同名文件已复制完成！")

def compare_gray_images(img_path1, img_path2, output_path):
    """
    比较两张大小相同的灰度图像，灰度值相同处为0，不同处为1，并输出结果图。

    参数：
        img_path1 (str): 第一张灰度图路径
        img_path2 (str): 第二张灰度图路径
        output_path (str): 结果图输出路径
    """
    # 读取为灰度图
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("无法读取图像，请检查路径是否正确。")

    if img1.shape != img2.shape:
        raise ValueError("两张图像大小不同，无法逐像素比较。")

    # 比较不同像素（相同为0，不同为1）
    diff = (img1 != img2).astype(np.uint8)

    # 保存结果（0为黑，1为白）
    cv2.imwrite(output_path, diff * 255)
    print(f"✅ 比较完成，结果已保存到：{output_path}")

def images_to_npy(input_dir, output_path, as_gray=True):
    """
    将指定文件夹下的所有图像读取并保存为npy文件

    :param input_dir: 输入图像文件夹路径
    :param output_path: 输出的npy文件路径
    :param as_gray: 是否转换为灰度图，默认True
    """
    # 支持的图像格式
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 获取所有图像文件路径（按文件名排序）
    img_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith(exts)]
    )

    if not img_files:
        raise ValueError(f"❌ 文件夹 {input_dir} 中未找到图像文件。")

    imgs = []
    for f in img_files:
        img = cv2.imread(f)
        if img is None:
            print(f"⚠️ 警告：无法读取 {f} ，已跳过。")
            continue
        if as_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)

    # 检查图像尺寸是否一致
    shapes = {img.shape for img in imgs}
    if len(shapes) > 1:
        print("⚠️ 警告：检测到图像尺寸不一致，建议先进行resize。")

    data = np.stack(imgs)
    np.save(output_path, data)
    print(f"✅ 成功保存 {len(imgs)} 张图像为 {output_path}")


images_to_npy(input_dir=r"C:\Users\Me\Desktop\CAPPI\data",output_path=r"E:\MyFiles\Projects\Banana\output/CAPPI0408_images.npy")

# src_dir = r"E:\MyFiles\Projects\Banana\output"
# target_filename = "denoised_0143.png"
# dst_dir = rf"E:\MyFiles\Projects\Banana\output\{target_filename}"
# copy_same_name_files(src_dir=src_dir,dst_dir=dst_dir,target_filename=target_filename)



# img_path1 = r"E:\MyFiles\Projects\Banana\outputdenoised_0143.png\denoised(21,33)_denoised_0143.png"
# img_path2 = r"E:\MyFiles\Projects\Banana\outputdenoised_0143.png\denoised(0,1)_denoised_0143.png"
# compare_result = r"E:\MyFiles\Projects\Banana\outputdenoised_0143.png\denoised(0,1)_vs_denoised(21,33).png"
# compare_gray_images(img_path1, img_path2, compare_result)



# diffs = np.load(r"E:\MyFiles\Projects\Banana\output\diffs.npy")  # (N-1, H, W)
# # 提取特征，转化为向量
# features = extract_pixel_features(diffs)
#
# H, W = diffs.shape[1], diffs.shape[2]
# # 作聚类
# n_clusters = 6
# labels, kmeans, scaler = cluster_pixel_distributions(features, n_clusters=n_clusters)
# # 找到聚类中心作为聚类的代表
# representatives = find_representative_pixels(features, labels, kmeans, scaler, top_k=3)
# # 聚类中心的分布可视化
# visualize_representative_distributions(diffs, representatives, H=H, W=W)
# # 聚类可视化
# visualize_clusters(labels, H=H, W=W, n_clusters=n_clusters)
#
# for x in range(0,255,5):
#     for y in range(x,255,5):
#         # 假设你已有 diffs, labels, H, W, images
#         cluster_thresholds = {
#             0: (0, 255),
#             1: (0, 255),
#             2: (x, y),
#             3: (0, 255),
#             4: (0, 255),
#             5: (0, 255)
#         }
#
#         image_paths = sorted(glob.glob(r"C:\Users\Me\Desktop\CAPPI\data\*.jpg"))
#         original_images = np.array([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths])
#
#         cleaned = denoise_images_by_clusters(
#             images=original_images,
#             diffs=diffs,
#             labels=labels,
#             H=H, W=W,
#             cluster_thresholds=cluster_thresholds,
#             save_dir=rf"E:\MyFiles\Projects\Banana\output\denoised({x},{y})"
#         )




