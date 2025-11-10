import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_balloon_from_txt(filepath, save_mp4=False,
                          filename="balloon_trajectory.mp4",
                          last_frame_img="last_frame.png",
                          speeds=[1.0]):
    """
    从 txt 文件读取热气球轨迹数据，并绘制动态三维轨迹

    参数:
        filepath: str, txt 文件路径
        save_mp4: bool, 是否保存为 mp4 格式
        filename: str, 输出文件名（仅当 save_mp4=True 时有效）
        last_frame_img: str, 保存最后一帧的静态图像文件名
        speeds: list[float], 视频倍速列表（如 [0.5, 1, 2]）
    """
    # 读取 txt 数据
    df = pd.read_csv(filepath, sep=r"\s+")
    time = df["时间"].values
    lon = df["经度"].values
    lat = df["纬度"].values
    alt = df["高度"].values

    # 绘制动画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], 'b-', lw=2)
    point, = ax.plot([], [], [], 'ro')

    def init():
        ax.set_xlim(min(lon), max(lon))
        ax.set_ylim(min(lat), max(lat))
        ax.set_zlim(min(alt), max(alt))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (m)")
        return line, point

    def update(frame):
        line.set_data(lon[:frame], lat[:frame])
        line.set_3d_properties(alt[:frame])
        point.set_data(lon[frame:frame + 1], lat[frame:frame + 1])
        point.set_3d_properties(alt[frame:frame + 1])
        ax.set_title(f"Time = {time[frame]:.2f}")
        return line, point

    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=100)

    if save_mp4:
        base, ext = os.path.splitext(filename)
        for spd in speeds:
            spd_filename = f"{base}_x{spd:g}{ext}"  # 输出文件名加上倍速后缀
            writer = FFMpegWriter(fps=int(10 * spd), metadata=dict(artist='Me'), bitrate=1800)
            ani.save(spd_filename, writer=writer)
            print(f"动画已保存为 {spd_filename}")

        # 保存最后一帧的静态图
        ax.clear()
        ax.plot(lon, lat, alt, 'b-', lw=2)
        ax.plot(lon[-1:], lat[-1:], alt[-1:], 'ro')  # 终点
        ax.set_xlim(min(lon), max(lon))
        ax.set_ylim(min(lat), max(lat))
        ax.set_zlim(min(alt), max(alt))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (m)")
        ax.set_title(f"Final Frame (Time = {time[-1]:.2f})")
        plt.savefig(last_frame_img, dpi=300)
        print(f"最后一帧图像已保存为 {last_frame_img}")
    else:
        plt.show()


def plot_balloon_pointcloud_from_txt(filepath, color_by="time", save_img=True, filename="balloon_pointcloud.png"):
    """
    从 txt 文件读取热气球轨迹数据，并绘制静态点云分布图（3D 散点图）

    参数:
        filepath: str, txt 文件路径
        color_by: str, 点的颜色依据，可选 "time" 或 "alt"
        save_img: bool, 是否保存为图像文件
        filename: str, 输出文件名（仅当 save_img=True 时有效）
    """
    # 读取数据
    df = pd.read_csv(filepath, sep=r"\s+")
    time = df["时间"].values
    lon = df["经度"].values
    lat = df["纬度"].values
    alt = df["高度"].values

    # 颜色映射
    if color_by == "time":
        c = time
        cmap = "viridis"
        cbar_label = "Time"
    elif color_by == "alt":
        c = alt
        cmap = "plasma"
        cbar_label = "Altitude (m)"
    else:
        raise ValueError("color_by 必须是 'time' 或 'alt'")

    # 绘制 3D 点云
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(lon, lat, alt, c=c, cmap=cmap, marker="o", s=20)

    # 坐标轴和标题
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Balloon Trajectory Point Cloud (colored by {color_by})")

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(cbar_label)

    # 保存或展示
    if save_img:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"点云图已保存为 {filename}")
    else:
        plt.show()




def plot_balloon_on_map(filepath, color_by="alt", save_img=True, filename="balloon_map.png"):
    """
    在地图上绘制热气球点云轨迹（2D）
    参数:
        filepath: str, txt 文件路径
        color_by: str, "alt"=高度着色, "time"=时间着色
        save_img: bool, 是否保存图像
        filename: str, 输出文件名
    """
    # 读取数据
    df = pd.read_csv(filepath, sep=r"\s+")
    time = df["时间"].values
    lon = df["经度"].values
    lat = df["纬度"].values
    alt = df["高度"].values

    # 颜色映射
    if color_by == "time":
        c = time
        cmap = "viridis"
        cbar_label = "Time"
    else:
        c = alt
        cmap = "plasma"
        cbar_label = "Altitude (m)"

    # 建立地图
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)    # 海岸线
    ax.add_feature(cfeature.BORDERS, linestyle=':')  # 国界
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # 设置显示范围（稍微留边）
    ax.set_extent([min(lon)-0.5, max(lon)+0.5, min(lat)-0.5, max(lat)+0.5], crs=ccrs.PlateCarree())

    # 绘制点云
    sc = ax.scatter(lon, lat, c=c, cmap=cmap, s=20, transform=ccrs.PlateCarree())

    # 起点 & 终点标记
    ax.plot(lon[0], lat[0], 'go', markersize=10, transform=ccrs.PlateCarree(), label="Start")
    ax.plot(lon[-1], lat[-1], 'ro', markersize=10, transform=ccrs.PlateCarree(), label="End")

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(cbar_label)

    ax.set_title(f"Balloon Trajectory on Map (colored by {color_by})")
    ax.legend()

    # 保存/展示
    if save_img:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"地图点云图已保存为 {filename}")
    else:
        plt.show()


filepath = "../../static/balloon_data.txt"

# # 动画生成
# plot_balloon_from_txt(filepath, save_mp4=True,
#                       filename="../output/balloon.mp4",
#                       last_frame_img="../output/final.png",
#                       speeds=[1, 2, 5])


# # 点云 按时间着色
# plot_balloon_pointcloud_from_txt(filepath, color_by="time",
#                                  filename="../output/balloon_pointcloud_time.png")
#
# # 点云 按高度着色
# plot_balloon_pointcloud_from_txt(filepath, color_by="alt",
#                                  filename="../output/balloon_pointcloud_alt.png")

# 地图点云
plot_balloon_on_map(filepath, filename="../../output/balloon_map.png")
