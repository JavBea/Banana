import numpy as np
import matplotlib.pyplot as plt

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


# =============================
# 示例用法
# =============================

if __name__ == "__main__":

    data = np.load(r"E:\MyFiles\data\CAPPI0408_images_single.npy")  # (239, 1000, 1000)

    # 例如观察坐标 (500, 600)，窗口3x3，统计均值
    # h, w, a = 500, 600, 3
    h, w, a = 131,539,1
    stat = "mean"

    values = analyze_pixel_series(data,h, w, a=a, stat=stat, nan_mode="ignore")

    plot_pixel_series(values, title=f"({h},{w}) {a}x{a} {stat} Trend")
