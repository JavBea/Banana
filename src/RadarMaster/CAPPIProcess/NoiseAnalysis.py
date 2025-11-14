import numpy as np
import matplotlib.pyplot as plt
import os


def handle_nan(region, mode="ignore"):
    """
    NaN处理逻辑接口：
    mode 可选：
        - "ignore": 忽略NaN计算（默认）
        - "zero": 将NaN视为0
        - "interpolate": 未来可拓展为插值
    """
    if mode == "ignore":
        return region  # np.nan*函数会自动忽略NaN
    elif mode == "zero":
        region = np.nan_to_num(region, nan=0.0)
        return region
    elif mode == "interpolate":
        # TODO: 可实现时间或空间插值逻辑
        return region
    else:
        raise ValueError(f"未知的 NaN 处理模式: {mode}")


def analyze_pixel_series(data, x, y, a=1, stat="mean", nan_mode="ignore"):
    """
    观察某个坐标随时间变化的情况

    参数：
    - data: ndarray, 形状 (T, H, W)
    - x, y: 坐标位置
    - a: 窗口大小（奇数），例如3表示取 (x,y) 周围3x3区域
    - stat: 统计方式 ("mean", "max", "min", "median")
    - nan_mode: NaN处理方式 ("ignore", "zero", "interpolate")

    返回：
    - values: ndarray, 长度为T的序列
    """
    assert a % 2 == 1, "窗口大小 a 必须为奇数"
    T, H, W = data.shape
    r = a // 2
    values = []

    for t in range(T):
        # 提取当前帧的局部区域
        x1, x2 = max(0, x - r), min(H, x + r + 1)
        y1, y2 = max(0, y - r), min(W, y + r + 1)
        region = data[t, x1:x2, y1:y2]

        # 处理NaN
        region = handle_nan(region, nan_mode)

        # 根据统计方式计算
        if stat == "mean":
            v = np.nanmean(region)
        elif stat == "max":
            v = np.nanmax(region)
        elif stat == "min":
            v = np.nanmin(region)
        elif stat == "median":
            v = np.nanmedian(region)
        else:
            raise ValueError(f"未知统计方式: {stat}")
        values.append(v)

    return np.array(values)


def plot_pixel_series(values, title="Pixel Time Series", ylabel="Value"):
    """
    绘制时间序列变化曲线
    横轴从0开始，步长为1，对应帧编号
    """
    N = len(values)
    x_axis = np.arange(N)  # 从0开始，步长为1
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, values, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Frame Index (starting from 0)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()



def save_all_pixel_series_images(data, save_dir, a=1, stat="mean", nan_mode="ignore", step=1, x_range=None,
                                 y_range=None):
    """
    遍历指定像素范围，生成每个像素点时间序列图并保存为图片

    参数：
    - data: ndarray, (T, H, W)
    - save_dir: 保存图片的目录
    - a: 窗口大小（奇数）
    - stat: 统计方式 ("mean", "max", "min", "median")
    - nan_mode: NaN处理方式 ("ignore", "zero", "interpolate")
    - step: 步长，默认1，可用于稀疏采样像素
    - x_range: tuple (start, end)，X方向像素范围
    - y_range: tuple (start, end)，Y方向像素范围
    """
    os.makedirs(save_dir, exist_ok=True)
    T, H, W = data.shape

    if x_range is None:
        x_range = (0, H)
    if y_range is None:
        y_range = (0, W)

    for x in range(x_range[0], x_range[1], step):
        for y in range(y_range[0], y_range[1], step):
            values = analyze_pixel_series(data, x, y, a=a, stat=stat, nan_mode=nan_mode)

            # 绘制并保存
            plt.figure(figsize=(6, 3))
            plt.plot(np.arange(T), values, marker="o", linewidth=1)
            plt.title(f"Pixel ({x},{y}) {a}x{a} {stat}")
            plt.xlabel("Frame Index")
            plt.ylabel("Value")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            # 文件名: x_y.png
            file_path = os.path.join(save_dir, f"{x}_{y}.png")
            plt.savefig(file_path)
            plt.close()  # 关闭图，避免内存占用过大

    print(f"所有图片已保存到 {save_dir}")


# =============================
# 示例用法
# =============================

if __name__ == "__main__":

    data = np.load(r"E:\MyFiles\data\CAPPI0408_images_single.npy")  # (239, 1000, 1000)


    # # 观察某个像素点
    # # 例如观察坐标 (500, 600)，窗口3x3，统计均值
    # # h, w, a = 500, 600, 3
    # h, w, a = 131,539,1
    # stat = "mean"
    #
    # values = analyze_pixel_series(data,h, w, a=a, stat=stat, nan_mode="ignore")
    #
    # plot_pixel_series(values, title=f"({h},{w}) {a}x{a} {stat} Trend")


    # # 指定步长遍历观察并输出保存
    save_dir = r"E:\MyFiles\Projects\Banana\output\analysis\a=1_mean_ignore"
    save_all_pixel_series_images(
        data,
        save_dir,
        a=1,
        stat="mean",
        nan_mode="ignore",
        step=20,  # 每隔20个像素采样一次，避免生成10^6张图
        x_range=(0, 1000),
        y_range=(0, 1000)
    )
