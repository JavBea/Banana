import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_predictions(data_path, pred_path, T_in=10, T_out=10, save_dir="vis_results", num_samples=5):
    """
    可视化预测结果，每张图左侧是真实值，右侧是预测值
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 读取数据
    data = sio.loadmat(data_path)
    pred = sio.loadmat(pred_path)

    u = data['u']          # (N, 64, 64, 20)
    preds = pred['pred']   # (N, T_out, 64, 64)

    # 2️⃣ 选取部分样本进行可视化
    num_samples = min(num_samples, preds.shape[0])
    print(f"正在可视化 {num_samples} 个样本...")

    for i in range(num_samples):
        for t in range(T_out):
            plt.figure(figsize=(6, 3))

            # 左侧：真实未来帧（真实值从 T_in 开始）
            plt.subplot(1, 2, 1)
            plt.imshow(u[i, :, :, t + T_in], cmap='jet')
            plt.title(f"Ground Truth t={t+T_in}")
            plt.axis("off")

            # 右侧：模型预测帧
            plt.subplot(1, 2, 2)
            plt.imshow(preds[i, t, :, :], cmap='jet')
            plt.title(f"Prediction t={t+T_in}")
            plt.axis("off")

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"sample{i:03d}_t{t:02d}.png")
            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"✅ 已保存: {save_path}")

    print(f"全部可视化完成，结果保存在文件夹：{save_dir}")


# ======================
# 🚀 手动设置参数区域
# ======================
if __name__ == "__main__":

    # 数据路径
    data_path = r"E:\MyFiles\data\CAPPI0010.mat"
    # 模型预测结果路径
    pred_path = r"E:\MyFiles\data\pred_Tin10_Tout10.mat"
    # 历史帧与预测帧长度
    T_in = 10
    T_out = 10
    # 输出目录
    save_dir = r"E:\MyFiles\Projects\Banana\src\RadarMaster\CAPPIProcess\output\temp"

    # 调用函数执行可视化
    visualize_predictions(data_path, pred_path, T_in, T_out, save_dir, num_samples=5)


import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def denoise_frame(frame, threshold_ratio=0.2):
    """
    去除孤立噪声点：
    若某像素点上下左右中有3个为深蓝色，则将该点设为深蓝色。
    """
    h, w = frame.shape
    clean = frame.copy()

    # 定义“深蓝色”的判定阈值（值较小表示深蓝）
    threshold = np.min(frame) + threshold_ratio * (np.max(frame) - np.min(frame))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 当前像素及上下左右
            val = frame[i, j]
            neighbors = [
                frame[i - 1, j],  # 上
                frame[i + 1, j],  # 下
                frame[i, j - 1],  # 左
                frame[i, j + 1],  # 右
            ]
            # 统计邻居中“深蓝色”的数量
            dark_count = sum(n < threshold for n in neighbors)

            # 若邻居中有4个是深蓝色
            if dark_count == 4:
                clean[i, j] = np.min(frame)
                clean[i-1, j] = np.min(frame)
                clean[i+1, j] = np.min(frame)
                clean[i, j-1] = np.min(frame)
                clean[i, j+1] = np.min(frame)

    return clean


def save_resized_predictions(pred_path, target_size=(1000, 1000), T_out=10, save_dir="resized_predictions", num_samples=5,denoise=True):
    """
    将预测结果去噪、上采样或下采样到指定大小并保存为图片（仅预测图像）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 读取预测数据
    pred = sio.loadmat(pred_path)
    preds = pred['pred']  # (N, T_out, H, W)

    num_samples = min(num_samples, preds.shape[0])
    print(f"正在处理 {num_samples} 个样本，每个样本 {T_out} 帧...")

    for i in range(num_samples):
        for t in range(T_out):
            frame = preds[i, t, :, :]

            # 2️⃣ 去噪
            if denoise:
                denoised = denoise_frame(frame)
            else:
                denoised = frame

            # 3️⃣ 上/下采样
            resized = resize(denoised, target_size, order=3, anti_aliasing=True)  # bicubic插值

            # 4️⃣ 保存图像
            plt.figure(figsize=(4, 4))
            plt.imshow(resized, cmap='jet')
            plt.axis("off")
            plt.title(f"Predicted Frame {t}")
            plt.tight_layout()

            save_path = os.path.join(save_dir, f"pred_sample{i:03d}_t{t:02d}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"✅ 已保存: {save_path}")

    print(f"全部完成！结果保存在文件夹：{save_dir}")


# ======================
# 🚀 手动设置参数区域
# ======================
#if __name__ == "__main__":
#    pred_path = r"E:\MyFiles\data\pred_Tin10_Tout10.mat"
#    save_dir = r"E:\MyFiles\Projects\Banana\src\RadarMaster\CAPPIProcess\output\predictions_10in_10out_5samples"

#    save_resized_predictions(
#        pred_path=pred_path,
#        target_size=(1000, 1000),
#        T_out=10,
#        save_dir=save_dir,
#        num_samples=50
#    )

