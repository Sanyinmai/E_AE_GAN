import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os.path
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据集
dataset = ''
# 输出文件夹
output = ''

# 频段数据
# frequency_band = dataset.split('(')[0]  # 87-108
# frequency_start = int(frequency_band.split('-')[0])  # 87
# frequency_end = int(frequency_band.split('-')[1])  # 108

# 读取文件路径
read_path = ''
# 保存文件路径
save_path = ''
#图像参数文件保存路径
# param_save_path = r''

if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_max_file(read_path):
    return len(os.listdir(read_path))

def read_data(n):
    filename = read_path + str(n) + '.txt'
    # open()
    with open(filename) as f:
        next(f)
        start = ''
        end = ''
        totaltimeRecord = []
        totalres = []
        read_line = 256
        skip_line = 3000
        for _ in range(7):
            res = []
            for i in range(read_line):
                line = f.readline().strip()
                line_data = line.split(' ;')
                data = line_data[0].split(' ')
                data = data[584:840]
                # if i == 0:
                #     start = line_data[1]
                # if i + 1 == read_line:
                #     end = line_data[1]
                res.append([float(j) for j in data])
            totalres.append(res)
            # totaltimeRecord.append(f'{start}|{end}')
            for _ in range(skip_line):
                f.readline()
    return totalres

def draw(n):
    """
    画频谱分布图
    :param n: 读取哪个文件
    :return: 1频谱分布图
    """
    index = 0
    file = os.listdir(save_path)
    file_num = len(file)
    m = read_data(n)
    np_data = np.array(m)  # N, H, W
    for data in np_data:
        file_num += 1
        max = np.max(data)
        min = np.min(data)
        # 计算 a  b
        # 文件名称：频段|步进|a|b|时间起点|时间终点
        a = 255.0/(max-min)
        b = (255.0*min)/(max-min)
        data = 255*((data-min)/(max-min))  #完成线性映射，变为值0~255之间的灰度图像
        data = data.astype(np.uint8)
        image = Image.fromarray(data, 'L')
        save_img_file = save_path + str(file_num) + '.png'
        # print(save_img_file)
        image.save(save_img_file)
        # record = f'{a}|{b}|{time}'
        index += 1


if __name__ == '__main__':
    max_file = get_max_file(read_path)
    start = 1
    for i in range(start, max_file + 1):
        print(f"正在生成第{i}组图片")
        draw(i)
