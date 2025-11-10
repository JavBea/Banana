import os
import numpy as np
from PIL import Image

def images_to_npy(input_folder, output_path, image_size=(1000, 1000)):
    """
    将一个文件夹下的所有图片整合成一个 npy 文件。

    参数:
        input_folder (str): 图片所在的文件夹路径
        output_path (str): 输出 .npy 文件的保存路径
        image_size (tuple): 图片统一的尺寸 (width, height)，默认为 (1000, 1000)

    输出:
        None （但会在指定路径生成 .npy 文件）
    """
    image_list = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 遍历文件夹中的图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(file_path).convert('RGB')  # 统一为RGB
                img = img.resize(image_size)
                img_array = np.array(img)
                image_list.append(img_array)
            except Exception as e:
                print(f"跳过无法读取的图片: {filename}, 错误: {e}")

    if not image_list:
        raise ValueError("文件夹中没有找到可读取的图片！")

    # 转换为 NumPy 数组
    images_np = np.stack(image_list, axis=0)  # 形状为 (N, H, W, 3)

    # 保存为 .npy 文件
    np.save(output_path, images_np)
    print(f"✅ 成功保存 {len(image_list)} 张图片为 npy 文件: {output_path}")

# 示例调用：
input_dir=r"E:\MyFiles\data\CAPPI0010\20250408"
output_dir=r"E:\MyFiles\data\CAPPI0408_images_origin.npy"
images_to_npy(input_dir,output_dir, image_size=(1000, 1000))
