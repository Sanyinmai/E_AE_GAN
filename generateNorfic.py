import os

from PIL import Image
import numpy as np


def simulate_anomaly(image, anomaly_type, intensity=1.0):
    if anomaly_type == 'spike':
        # 随机选择位置插入突发尖峰
        x, y = np.random.randint(low=0, high=image.shape[0]), np.random.randint(low=0, high=image.shape[1])
        image[x, y] += intensity * image.max()
    elif anomaly_type == 'dropout':
        # 随机选择一块区域进行信号丢失
        x, y, size = np.random.randint(low=0, high=image.shape[0]), np.random.randint(low=0, high=image.shape[1]), 30
        image[max(0, x-size):x+size, max(0, y-size):y+size] = 0
    return image

def swap_boxes(image, box1, box2):
    """
    在图像中交换两个方框内的像素值。
    :param image: 2D NumPy 数组，表示灰度图像
    :param box1: 第一个方框的坐标 (left1, upper1, right1, lower1)
    :param box2: 第二个方框的坐标 (left2, upper2, right2, lower2)
    """
    # 计算每个方框的大小
    size1 = (box1[2] - box1[0], box1[3] - box1[1])
    size2 = (box2[2] - box2[0], box2[3] - box2[1])
    # 确保两个方框大小相同
    if size1 != size2:
        raise ValueError("两个方框必须具有相同的大小")
    # 提取方框区域
    region1 = image[box1[1]:box1[3], box1[0]:box1[2]].copy()
    region2 = image[box2[1]:box2[3], box2[0]:box2[2]].copy()
    # 交换方框内的区域
    image[box1[1]:box1[3], box1[0]:box1[2]] = region2
    image[box2[1]:box2[3], box2[0]:box2[2]] = region1
    return image

def overlay_box(image, source_box, target_box):
    """
    将一个方框的区域覆盖到另一个方框的位置。
    :param image: 2D NumPy 数组，表示灰度图像
    :param source_box: 源方框的坐标 (left1, upper1, right1, lower1)
    :param target_box: 目标方框的坐标 (left2, upper2, right2, lower2)
    """
    # 计算每个方框的大小
    size1 = (source_box[2] - source_box[0], source_box[3] - source_box[1])
    size2 = (target_box[2] - target_box[0], target_box[3] - target_box[1])

    # 确保两个方框大小相同
    if size1 != size2:
        raise ValueError("源方框和目标方框必须具有相同的大小")

    # 提取源方框区域
    region = image[source_box[1]:source_box[3], source_box[0]:source_box[2]].copy()

    # 将源区域覆盖到目标方框的位置
    image[target_box[1]:target_box[3], target_box[0]:target_box[2]] = region

    return image
def overlay_box_zz(source_image, target_image, source_box, target_box):
    """
    将一个方框的区域覆盖到另一个方框的位置。
    :param image: 2D NumPy 数组，表示灰度图像
    :param source_box: 源方框的坐标 (left1, upper1, right1, lower1)
    :param target_box: 目标方框的坐标 (left2, upper2, right2, lower2)
    """
    # 计算每个方框的大小
    size1 = (source_box[2] - source_box[0], source_box[3] - source_box[1])
    size2 = (target_box[2] - target_box[0], target_box[3] - target_box[1])

    # 确保两个方框大小相同
    if size1 != size2:
        raise ValueError("源方框和目标方框必须具有相同的大小")

    # 提取源方框区域
    region = source_image[source_box[1]:source_box[3], source_box[0]:source_box[2]].copy()

    # 将源区域覆盖到目标方框的位置
    target_image[target_box[1]:target_box[3], target_box[0]:target_box[2]] = region

    return target_image
# 加载图像

index = 0
while(index < 1):
    file_num = len(os.listdir(r"F:\spectrum_data\Test256_Anomaly")) + 1
    index += 1
    image_path_zz = f'F:\spectrum_data\Test256_Anomaly/27.png'
    image_path = f'F:\spectrum_data\Test256_Anomaly\/9.png'

    image = Image.open(image_path).convert("L")
    image_np = np.array(image)

    image_zz = Image.open(image_path_zz).convert("L")
    image_zz_np = np.array(image_zz)

    box1 = (115, 20, 130, 80)
    box2 = (140, 80, 155, 140)
    swapped_image = overlay_box(image_zz_np, box2, box1)
    swapped_image = Image.fromarray(swapped_image)


    # 保存和显示结果图像
    output_path = f'F:\spectrum_data\Test256_Anomaly\/{file_num}.png'
    swapped_image.save(output_path)
