import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = "../../static/balloon_data.txt"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# haversine 公式计算球面距离（单位：公里）
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # 地球半径（公里）
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 读取数据
df = pd.read_csv(filepath, sep=r"\s+")
lon = df["经度"].values
lat = df["纬度"].values
elev = df["高度"].values  # 假设文件中有“海拔”列

# 排序（防止点顺序混乱）
coords = np.array(sorted(zip(lon, lat, elev)))

# 相邻点的水平距离和海拔差
horiz_distances = []
vertical_diffs = []
for i in range(len(coords) - 1):
    d = haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
    horiz_distances.append(d)
    dz = abs(coords[i+1][2] - coords[i][2])
    vertical_diffs.append(dz)

horiz_distances = np.array(horiz_distances)
vertical_diffs = np.array(vertical_diffs)

# 统计信息
new_min = horiz_distances.min()
new_max = horiz_distances.max()
mean_val = horiz_distances.mean()
median_val = np.median(horiz_distances)

min_elev = vertical_diffs.min()
max_elev = vertical_diffs.max()
mean_elev = vertical_diffs.mean()
median_elev = np.median(vertical_diffs)

# ---------------- 绘图部分 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文字体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10))

# 绘制水平距离直方图
ax1.hist(horiz_distances, bins=100, edgecolor="black", alpha=0.7)
ax1.set_xlabel("相邻点水平距离 (km)")
ax1.set_ylabel("频数")
ax1.set_title("相邻点水平距离分布")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_xlim(0, 2)

textstr1 = f"最小: {new_min:.2f} km\n最大: {new_max:.2f} km\n均值: {mean_val:.2f} km\n中位数: {median_val:.2f} km"
ax1.text(0.95, 0.95, textstr1, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# 绘制海拔差直方图
ax2.hist(vertical_diffs, bins=50, edgecolor="red", alpha=0.6)
ax2.set_xlabel("相邻点海拔差 (m)")
ax2.set_ylabel("频数")
ax2.set_title("相邻点海拔差分布")
ax2.grid(True, linestyle="--", alpha=0.5)

textstr2 = f"最小: {min_elev:.1f} m\n最大: {max_elev:.1f} m\n均值: {mean_elev:.1f} m\n中位数: {median_elev:.1f} m"
ax2.text(0.95, 0.95, textstr2, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

