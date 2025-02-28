import random
from PIL import Image
import os
from utils.generateNorfic import simulate_anomaly
import numpy as np

def center_crop(image, crop_size):
    width, height = image.size
    new_width, new_height = crop_size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))

def crop_right_top(image, crop_size):
    width, height = image.size
    new_width, new_height = crop_size
    left = width - new_width
    top = 0
    right = width
    bottom = new_height

    return image.crop((left, top, right, bottom))

def crop_images_in_folder(input_folder, output_folder, crop_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 可以根据实际格式调整
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert('L')  # 转换为灰度图像
            cropped_image = crop_right_top(image, crop_size)
            random_number = random.randint(1, 6)
            if random_number < 3:
                cropped_image = np.array(cropped_image)
                cropped_image = simulate_anomaly(cropped_image, 'dropout')
                cropped_image = Image.fromarray(cropped_image)

            output_path = os.path.join(output_folder, filename)
            cropped_image.save(output_path)

input_folder = 'F:\spectrum_data\TrainSp512\Train512'  # 输入文件夹路径
output_folder = 'F:\spectrum_data\TrainSp512\Train256'  # 输出文件夹路径

crop_images_in_folder(input_folder, output_folder)
