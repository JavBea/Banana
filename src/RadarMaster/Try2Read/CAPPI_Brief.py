import os


def read_cappi_with_fixed_spacing(filename, start_offset=1049, segment_size=480, spacing=1502):
    """
    根据已知的间隔和起始位置提取 CAPPI 数据
    :param filename: 数据文件路径
    :param start_offset: 第一段数据的起始位置（字节索引），默认 1049
    :param segment_size: 每段数据的长度（字节数），手动指定
    :param spacing: 每两段数据之间的字节间隔，默认 1502
    :return: ndarray, (段数, segment_size)
    """
    with open(filename, "rb") as f:
        raw = f.read()

    segments = []
    i = start_offset
    while i + segment_size <= len(raw):
        # 提取一段数据
        segment = raw[i:i + segment_size]
        segments.append(list(segment))
        # 移动到下一段的起始位置
        i += spacing

    return np.array(segments, dtype=np.uint8)


def visualize_array_gray(array_2d):
    """
    使用灰度图可视化二维数组
    :param array_2d: ndarray (段数, 256)
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(array_2d, aspect='auto', cmap='gray', interpolation='none')
    plt.colorbar(label='Byte value')
    plt.xlabel('Byte index in segment')
    plt.ylabel('Segment index')
    plt.title('CAPPI Data Visualization (Gray)')
    plt.show()



import matplotlib.pyplot as plt
import numpy as np


def visualize_array_polar(array_2d):
    """
    极坐标可视化二维数组
    :param array_2d: ndarray (段数, 256)
    """
    segments, points = array_2d.shape
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R, array_2d, shading='auto', cmap='gray')
    fig.colorbar(c, ax=ax, label='Byte value')
    ax.set_title('CAPPI Data Visualization (Polar)')
    plt.show()

def visualize_array_polar_clockwise(array_2d):
    segments, points = array_2d.shape
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    r = np.linspace(0, 1, points)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R, array_2d, shading='auto', cmap='gray')

    # 顺时针显示
    ax.set_theta_direction(-1)
    # 可选：0 度在顶部
    ax.set_theta_zero_location('N')

    fig.colorbar(c, ax=ax, label='Byte value')
    ax.set_title('CAPPI Data Visualization (Polar, Clockwise)')
    plt.show()


def process_and_save_all(
    input_dir,
    output_prefix,
    start_offset=1049,
    segment_size=480,
    spacing=1502
):
    """
    批量处理 CAPPI 文件，并将结果同时保存为 .bin (memmap 友好) 和 .npy (标准)

    :param input_dir: 输入文件夹
    :param output_prefix: 输出文件前缀，不带扩展名
    """
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files.sort()

    if not files:
        raise ValueError("文件夹中没有数据文件")

    # 先读取第一个文件，确定单个数组的形状
    sample = read_cappi_with_fixed_spacing(files[0], start_offset, segment_size, spacing)
    sample_shape = sample.shape   # (段数, segment_size)

    # 创建 memmap 数组 (文件数, 段数, segment_size)
    big_shape = (len(files),) + sample_shape
    bin_file = output_prefix + ".bin"
    arr = np.memmap(bin_file, dtype=np.uint8, mode="w+", shape=big_shape)

    # 逐个处理写入
    for i, file_path in enumerate(files):
        data = read_cappi_with_fixed_spacing(file_path, start_offset, segment_size, spacing)
        if data.shape != sample_shape:
            raise ValueError(f"文件 {file_path} 的 shape={data.shape} 与首个文件不一致 {sample_shape}")
        arr[i] = data
        if i % 100 == 0:
            print(f"已处理 {i}/{len(files)}")

    # 刷新到磁盘
    arr.flush()
    print(f"✅ 已保存到 {bin_file}, shape={big_shape}")

    # 同时保存一份标准 .npy 文件（一次性加载到内存）
    npy_file = output_prefix + ".npy"
    np.save(npy_file, np.array(arr))
    print(f"✅ 另存为 {npy_file} (标准 NumPy 格式，可直接 np.load)")



# filepath = r"C:\Users\Me\Desktop\雷达数据\20250228\CAPPI\C20250228090622.TC1"
#
# array_2d = read_cappi_with_fixed_spacing(filepath, start_offset=1049, segment_size=480, spacing=1502)
# print("二维数组形状:", array_2d.shape)
# # 调用可视化
# visualize_array_gray(array_2d)
# # 调用极坐标可视化
# visualize_array_polar_clockwise(array_2d)


process_and_save_all(
    input_dir=r"E:\MyFiles\radar_data_by_types\CAPPI",
    output_prefix=r"E:\MyFiles\Projects\Banana\output\CAPPI0408",
    start_offset=1049,
    segment_size=480,
    spacing=1502
)

