import numpy as np
import matplotlib.pyplot as plt
import os

def convert_color_to_value(input_npy_path, output_npy_path):
    """
    将RGB颜色图像(.npy)转换为数值图像:
      - 黑色(0,0,0) 或未知颜色 -> np.nan
      - 其他颜色根据映射表替换为对应数值
    """

    # === 定义颜色映射表 ===
    hex_colors = [
        "#9c9c9c","#767676","#aaaaff","#8c8cee","#7070c9","#00ffff","#0096ff","#0000ff",
        "#00ff00","#00c800","#009600","#ffff00","#ffc800","#ff7800","#ff0000",
        "#c80000","#960000","#ff00ff","#9600fa"
    ]
    nums = np.array([-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], dtype=np.float32)

    # 转换 hex → RGB 三元组
    color_map = {
        tuple(int(h[i:i+2], 16) for i in (1, 3, 5)): v
        for h, v in zip(hex_colors, nums)
    }
    black = (0, 0, 0)

    # === 加载数据 ===
    data = np.load(input_npy_path)  # (239,1000,1000,3)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"输入形状应为 (N,H,W,3)，当前为 {data.shape}")

    # === 结果容器 ===
    result = np.full((data.shape[0], data.shape[1], data.shape[2]), np.nan, dtype=np.float32)

    # 将所有颜色转换为扁平化数组，方便查表
    for i in range(data.shape[0]):
        img = data[i]  # (H,W,3)
        flat = img.reshape(-1, 3)
        out_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)

        # 逐个颜色匹配映射
        for color, value in color_map.items():
            mask = np.all(flat == color, axis=1)
            out_flat[mask] = value

        # 黑色或未知颜色保持 NaN（默认值）
        # 所以这里不需要额外处理黑色或其他异常色

        result[i] = out_flat.reshape(img.shape[0], img.shape[1])

        if i % 10 == 0 or i == data.shape[0]-1:
            print(f"已处理 {i+1}/{data.shape[0]} 张图像")

    # === 保存结果 ===
    np.save(output_npy_path, result)
    print(f"✅ 转换完成，输出文件已保存至: {output_npy_path}")

def convert_color_to_value_with_nearest(input_npy_path, output_npy_path):
    """
    将RGB颜色图像(.npy)转换为数值图像（支持近似匹配）:
      - 黑色(#000000) 或异常颜色 -> np.nan
      - 其他颜色 -> 匹配与色标中最接近的颜色的数值
    """

    # === 定义颜色映射表 ===
    hex_colors = [
        "#000000",  # 黑色，对应 NaN
        "#9c9c9c","#767676","#aaaaff","#8c8cee","#7070c9","#00ffff","#0096ff","#0000ff",
        "#00ff00","#00c800","#009600","#ffff00","#ffc800","#ff7800","#ff0000",
        "#c80000","#960000","#ff00ff","#9600fa"
    ]

    # 数值表，与颜色一一对应，第一个为 np.nan
    nums = np.array([np.nan, -15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], dtype=np.float32)

    # 转换 hex → RGB 三元组
    color_map = np.array([[int(h[i:i+2], 16) for i in (1,3,5)] for h in hex_colors], dtype=np.float32)

    # === 读取数据 ===
    data = np.load(input_npy_path)  # (N,H,W,3)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"输入形状应为 (N,H,W,3)，当前为 {data.shape}")

    result = np.full((data.shape[0], data.shape[1], data.shape[2]), np.nan, dtype=np.float32)

    # === 主循环 ===
    for i in range(data.shape[0]):
        img = data[i].astype(np.float32)  # (H,W,3)
        H, W, _ = img.shape
        flat = img.reshape(-1, 3)

        out_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)

        # 计算每个像素与所有色标的距离（欧几里得距离）
        diffs = flat[:, None, :] - color_map[None, :, :]  # shape: (num_pixels, 20, 3)
        dist = np.linalg.norm(diffs, axis=2)              # shape: (num_pixels, 20)
        nearest_idx = np.argmin(dist, axis=1)

        # 赋值为对应数值（包括NaN）
        out_flat[:] = nums[nearest_idx]

        # reshape 回原尺寸
        result[i] = out_flat.reshape(H, W)

        if i % 10 == 0 or i == data.shape[0]-1:
            print(f"已处理 {i+1}/{data.shape[0]} 张图像")

    np.save(output_npy_path, result)
    print(f"✅ 转换完成，结果保存至: {output_npy_path}")

def visualize_npy_as_grayscale(npy_path, output_dir):
    data = np.load(npy_path)
    data = np.abs(np.nan_to_num(data, nan=0.0))
    data /= np.max(data)  # 归一化

    os.makedirs(output_dir, exist_ok=True)

    for i in range(data.shape[0]):
        frame = data[i]
        save_path = os.path.join(output_dir, f"frame_{i:03d}.png")

        # ✅ 推荐：直接使用 imsave，不缩放、不裁剪
        plt.imsave(save_path, frame, cmap='gray', vmin=0, vmax=1)

        if i % 10 == 0 or i == data.shape[0]-1:
            print(f"已保存 {i+1}/{data.shape[0]} 张图像")

    print(f"✅ 全部灰度图已保存到: {output_dir}")

# 示例调用
if __name__ == "__main__":
    input_dir=r"E:\MyFiles\data\CAPPI0408_images_origin_v2.npy"
    output_dir=r"E:\MyFiles\data\CAPPI0408_images_single.npy"

    # 输出单通道数组
    convert_color_to_value_with_nearest(input_dir, output_dir)

    # 可视化来简单校验
    visualize_npy_as_grayscale(
        npy_path=output_dir,
        output_dir=r"E:\MyFiles\data\temp"
    )
