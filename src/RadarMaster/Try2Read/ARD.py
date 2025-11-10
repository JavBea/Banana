from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

arf_file_path = Path("D:\TY\\20250228\ARD\\arf\C20250228090622.ARF")
# 假设 Header 为 1024 字节
with open(arf_file_path, 'rb') as f:
    f.seek(1024)  # 跳过文件头
    raw_data = f.read()

# 初步假设：数据部分为 uint8 格式，每个值表示一个 dBZ 等级
# 你可以改为 float32/uint16 等类型试探
try:
    arr = np.frombuffer(raw_data, dtype=np.uint8)

    # 尝试将其 reshape 为合理维度（可多次尝试）
    for size in [512, 500, 460, 438, 400, 360, 256, 240]:
        if arr.size >= size * size:
            grid = arr[:size * size].reshape((size, size))
            break
    else:
        raise ValueError("未知数据尺寸")

    # 可视化
    np.save('arf1.npy', grid)
    plt.imshow(grid, cmap="turbo", origin='lower')
    plt.colorbar(label="Reflectivity (dBZ)")
    plt.title(f"ARF Visualization - {arf_file_path.name}")
    plt.axis("off")
    plt.show()

except Exception as e:
    print("读取失败：", e)

# tune_polar_params.py —— ARD 正确解码 + ADWRDB 风格配色 + 背景黑
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ========= 路径（两选一）=========
DL_PATH = r"arf1.npy"  # 二维 (H,W) 的 uint8 码值（示例）
# DL_PATH = r"E:\TY\out\20250228\ARD\stack\dl_TCHW.npy"  # (T,C,H,W) in [0,1]

# 若使用 dl_TCHW.npy 时的帧/通道选择
T_IDX, C_IDX = 0, 0

# 速度范围与极坐标半径
VMIN, VMAX = -50.0, 50.0   # m/s（同时用于 ARD 解码中的 Nyquist 近似）
R_MAX_KM   = 230.0         # 最大径向（km）

# ========= ARD 速度配色（按你给的色标）=========
def make_vel_cmap():
    # 从负到正（绿 → 白 → 红），与右侧刻度配合
    colors = [
        "#003300",  # -50
        "#006600",  # -30
        "#009900",  # -20
        "#00CC00",  # -15
        "#00FF00",  # -10
        "#66FF66",  #  -5
        "#99FF99",  #  -1
        "#FFFFFF",  #   0
        "#FFCCCC",  #  +0.5
        "#FF6666",  #  +1
        "#FF0000",  #  +5
        "#CC0000",  # +10
        "#990000",  # +15
        "#660000",  # +20
        "#330000",  # +30
        "#220000",  # +50
    ]
    # 离散边界（比颜色多 1 个端点，确保颜色段与数值区间一一对应）
    bounds = [-50, -30, -20, -15, -10, -5, -1, 0, 0.5, 1, 5, 10, 15, 20, 30, 50]
    cmap = ListedColormap(colors, name="vel_custom")
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

# ========= 数据加载：dl_TCHW 反归一 or ARD 原始码值解码 =========
def load_velocity_ms():
    arr = np.load(DL_PATH)

    # 情况 A：dl_TCHW (T,C,H,W) ∈ [0,1] → 反归一回 m/s
    if arr.ndim == 4:
        dl = arr[T_IDX, C_IDX].astype(np.float32)    # (H,W)
        v  = dl * (VMAX - VMIN) + VMIN               # m/s
        return v

    # 情况 B：二维 (H,W) 的 ARD 原始码值（uint8: 0..255）
    if arr.ndim == 2:
        code = arr.astype(np.float32)
        miss = (code == 255)
        # v = code# 255=缺测
        # v = (code-VMIN)/(VMAX-VMIN)
        v = (code - 128.0) / 127.0 * VMAX            # 128 为 0 m/s，±127 覆盖到 ±VMAX
        v[miss] = np.nan
        return v

    raise ValueError(f"不支持的数组维度: {arr.shape}")

# ========= 极坐标绘制 =========
def build_edges(n, a, b):
    return np.linspace(a, b, n + 1)

def polar_plot(ax, v, *, az_first_is_row, transpose, clockwise, angle0_deg, title):
    data = v.T if transpose else v

    # 按 (azimuth, range) 解释二维数据
    if az_first_is_row:
        az_n, r_n = data.shape
        az_r = data
    else:
        r_n, az_n = data.shape
        az_r = data.T

    th0 = np.deg2rad(angle0_deg)
    th_edges = build_edges(az_n, th0, th0 - 2*np.pi if clockwise else th0 + 2*np.pi)
    r_edges  = build_edges(r_n, 0.0, R_MAX_KM)

    TH, RR = np.meshgrid(th_edges, r_edges, indexing='ij')
    X = RR * np.sin(TH)
    Y = RR * np.cos(TH)

    # 缺测/圆盘外 → mask（背景将为黑）
    az_r = np.ma.array(az_r, copy=True)
    az_r = np.ma.masked_invalid(az_r)
    az_r = np.ma.masked_where(RR[:-1, :-1] > R_MAX_KM, az_r)

    # 速度配色
    cmap, norm, _ = make_vel_cmap()
    cmap.set_bad("black")

    pm = ax.pcolormesh(X, Y, az_r, shading='auto', cmap=cmap, norm=norm)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return pm

# ========= 主程序：九宫格快调 =========
if __name__ == "__main__":
    v = load_velocity_ms()  # (H,W) m/s

    combos = [
        dict(az_first_is_row=True,  transpose=False, clockwise=True,  angle0_deg=0),
        dict(az_first_is_row=True,  transpose=False, clockwise=True,  angle0_deg=90),
        dict(az_first_is_row=True,  transpose=False, clockwise=True,  angle0_deg=180),
        dict(az_first_is_row=True,  transpose=False, clockwise=False, angle0_deg=0),
        dict(az_first_is_row=True,  transpose=False, clockwise=False, angle0_deg=90),
        dict(az_first_is_row=False, transpose=False, clockwise=True,  angle0_deg=0),
        dict(az_first_is_row=False, transpose=False, clockwise=True,  angle0_deg=90),
        dict(az_first_is_row=False, transpose=True,  clockwise=True,  angle0_deg=0),
        dict(az_first_is_row=False, transpose=True,  clockwise=True,  angle0_deg=90),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=130)
    axes = axes.ravel()
    pm = None
    for ax, cfg in zip(axes, combos):
        title = f"row=az:{cfg['az_first_is_row']} T:{cfg['transpose']} CW:{cfg['clockwise']} θ0:{cfg['angle0_deg']}"
        pm = polar_plot(ax, v, **cfg, title=title)

    # 右侧色标（离散刻度）
    cmap, norm, bounds = make_vel_cmap()
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.tolist(), fraction=0.025, pad=0.01, ticks=bounds
    )
    cbar.ax.set_ylabel("m/s", rotation=90)

    plt.suptitle("Quick tuning: pick the one that matches ADWRDB best", y=0.94)
    plt.tight_layout()
    plt.show()
    # 保存最匹配图片
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    cfg = dict(az_first_is_row=True, transpose=False, clockwise=False, angle0_deg=0)
    pm = polar_plot(ax, v, **cfg, title="Best match")

    # 加色标
    cbar = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label="m/s")

    # 保存
    fig.savefig("best_match.png", bbox_inches="tight")
    plt.close(fig)