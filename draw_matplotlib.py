import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import spectrogram

import numpy as np
import matplotlib.pyplot as plt

def generate_singleSignal(height, width, signal_intensity = 255):
    background = np.full((height, width), 10)
    rand_witdh = np.random.randint(1, 4)*10
    rand_xpos = np.random.randint(5, width-rand_witdh)
    signal_start_col = rand_xpos  # 信号的起始列
    signal_width = rand_witdh  # 信号的宽度
    signal_height = height  # 信号的高度（填满整个高度）
    background[:, signal_start_col:signal_start_col + signal_width] = signal_intensity
    return background, signal_start_col

def generate_mutilSignal(height, width, signal_intensity = 255):
    background = np.full((height, width), 10)
    used_positions = []
    num_signals = np.random.randint(2, 4)
    for _ in range(num_signals):
        while True:
            rand_width = np.random.randint(1, 5) * 10
            rand_xpos = np.random.randint(5, width - rand_width)
            # 检查新信号是否与已有信号重叠
            overlap = any(rand_xpos < pos + w and rand_xpos + rand_width > pos for pos, w in used_positions)
            if not overlap:
                used_positions.append((rand_xpos, rand_width))
                break
        signal_start_col = rand_xpos  # 信号的起始列
        signal_width = rand_width  # 信号的宽度
        signal_height = height  # 信号的高度（填满整个高度）
        background[:, signal_start_col:signal_start_col + signal_width] = signal_intensity
    return background, signal_start_col

def generate_pulseSignal(height, width, signal_intensity = 255):
    background = np.full((height, width), 10)  # 使用较低的数值表示紫色背景
    rand_width = np.random.randint(1, 5) * 10
    rand_xpos = np.random.randint(5, width - rand_width)

    rand_height = np.random.randint(10, 20) * 8
    rand_ypos = np.random.randint(10, height - rand_height)

    signal_start_col = rand_xpos  # 信号的起始列
    signal_width = rand_width  # 信号的宽度
    signal_height = height  # 信号的高度（填满整个高度）

    signal_start_blank_row = rand_ypos  # 脉冲信号空白的起始行
    signal_blank_height = rand_height  # 信号空白的宽度

    background[:, signal_start_col:signal_start_col + signal_width] = signal_intensity
    background[signal_start_blank_row:signal_start_blank_row + signal_blank_height, :] = 10

    return background, signal_start_col

def generate_conversionSignal(height, width, signal_intensity = 255):
    background = np.full((height, width), 10)
    signal_width = np.random.randint(55, 65)
    signal_height = 66
    num = min(height//signal_height, width//signal_width)
    start_col = np.random.randint(0, width // 3)  # 根据行号来确定起始列位置
    first_start_col = start_col

    for i in range(num):
        start_row = i * signal_height
        end_col = start_col + signal_width
        end_row = start_row + signal_height

        if end_col <= width and end_row <= height:  # 防止超出图像边界
            background[start_row:end_row, start_col:end_col] = signal_intensity
        start_col = end_col - signal_width // 2
    return background, first_start_col

def generate_pulseSignal_AllNoise(height, width):
    background, _ = generate_pulseSignal(height, width, signal_intensity=255)
    rand_height = np.random.randint(3, 6) * 10
    start_ypos = np.random.randint(5, (height - rand_height) // 2)
    noise_intensity = 200
    region = background[start_ypos: start_ypos + rand_height, :]
    mask = region != 255
    region[mask] = noise_intensity
    background[start_ypos: start_ypos + rand_height, :] = region
    return background

def generate_conversionSignal_AllNoise(height, width):  #在变频信号上生成全频段干扰信号
    background, _ = generate_conversionSignal(height, width, signal_intensity=255)
    rand_height = np.random.randint(3, 6) * 10
    start_ypos = np.random.randint(5, (height - rand_height) // 2)
    noise_intensity = 200
    region = background[start_ypos: start_ypos + rand_height, :]
    mask = region != 255
    region[mask] = noise_intensity
    background[start_ypos: start_ypos + rand_height, :] = region
    return background

def generate_mutilSignal_AllNoise(height, width):
    background, _ = generate_mutilSignal(height, width, signal_intensity=255)
    rand_height = np.random.randint(3, 6) * 10
    start_ypos = np.random.randint(5, (height - rand_height) // 2)
    noise_intensity = 200
    region = background[start_ypos: start_ypos + rand_height, :]
    mask = region != 255
    region[mask] = noise_intensity
    background[start_ypos: start_ypos + rand_height, :] = region
    return background

def generate_singleSignal_AllNoise(height, width):  #在单音信号上生成全频段干扰信号
    background, _ = generate_singleSignal(height, width, signal_intensity=255)
    rand_height = np.random.randint(1, 4) * 10
    start_ypos = np.random.randint(5, height - rand_height)
    noise_intensity = 200
    region = background[start_ypos: start_ypos + rand_height, :]
    mask = region != 255
    region[mask] = noise_intensity
    background[start_ypos: start_ypos + rand_height, :] = region
    return background

def generate_singleSignal_RandimpulseNoise(height, width):
    background, _ = generate_singleSignal(height, width, signal_intensity=255)
    num_pulses = np.random.randint(1, 2)
    for _ in range(num_pulses):
        pulse_width = np.random.randint(10, 40)  # 脉冲信号宽度 (随机 5~15)
        pulse_height = np.random.randint(30, height // 2)  # 脉冲信号高度 (随机高)
        pulse_xpos = np.random.randint(5, width - pulse_width)  # 随机脉冲位置
        pulse_ypos = np.random.randint(0, height - pulse_height)  # 随机脉冲顶部位置
        noise_intensity = 200
        background[pulse_ypos:pulse_ypos + pulse_height, pulse_xpos:pulse_xpos + pulse_width] = noise_intensity
    return background

def generate_mutilSignal_RandimpulseNoise(height, width):
    background, _ = generate_mutilSignal(height, width, signal_intensity=255)
    num_pulses = np.random.randint(1, 2)
    for _ in range(num_pulses):
        pulse_width = np.random.randint(10, 40)  # 脉冲信号宽度 (随机 5~15)
        pulse_height = np.random.randint(30, height // 2)  # 脉冲信号高度 (随机高)
        pulse_xpos = np.random.randint(5, width - pulse_width)  # 随机脉冲位置
        pulse_ypos = np.random.randint(0, height - pulse_height)  # 随机脉冲顶部位置
        noise_intensity = 200
        background[pulse_ypos:pulse_ypos + pulse_height, pulse_xpos:pulse_xpos + pulse_width] = noise_intensity
    return background

def generate_conversionSignal_RandimpulseNoise(height, width):
    background, _ = generate_conversionSignal(height, width, signal_intensity=255)
    num_pulses = np.random.randint(1, 2)
    for _ in range(num_pulses):
        pulse_width = np.random.randint(10, 40)  # 脉冲信号宽度 (随机 5~15)
        pulse_height = np.random.randint(30, height // 2)  # 脉冲信号高度 (随机高)
        pulse_xpos = np.random.randint(5, width - pulse_width)  # 随机脉冲位置
        pulse_ypos = np.random.randint(0, height - pulse_height)  # 随机脉冲顶部位置
        noise_intensity = 200
        background[pulse_ypos:pulse_ypos + pulse_height, pulse_xpos:pulse_xpos + pulse_width] = noise_intensity
    return background

def generate_pulseSignal_RandimpulseNoise(height, width):
    background, _ = generate_pulseSignal(height, width, signal_intensity=255)
    num_pulses = np.random.randint(1, 2)
    for _ in range(num_pulses):
        pulse_width = np.random.randint(10, 40)  # 脉冲信号宽度 (随机 5~15)
        pulse_height = np.random.randint(30, height // 2)  # 脉冲信号高度 (随机高)
        pulse_xpos = np.random.randint(5, width - pulse_width)  # 随机脉冲位置
        pulse_ypos = np.random.randint(0, height - pulse_height)  # 随机脉冲顶部位置
        noise_intensity = 200
        background[pulse_ypos:pulse_ypos + pulse_height, pulse_xpos:pulse_xpos + pulse_width] = noise_intensity
    return background

def draw(n, width, height, snr_db):
    index = 0
    save_path = 'F:\spectrum_data\Train_simulation\/Train_simulation/'
    while (index < n):
        index += 1
        # rand = np.random.randint(0, 4)
        rand = 0
        if(rand == 0):
            background, _ = generate_singleSignal(height, width)
        elif(rand == 1):
            background, _ = generate_mutilSignal(height, width)
        elif(rand == 2):
            background, _ = generate_pulseSignal(height, width)
        else:
            background, _ = generate_conversionSignal(height, width)
        #添加噪声，制作不同信噪比的数据集。
        # signal_power = np.mean(background ** 2)
        # snr_linear = 10 ** (snr_db / 10)  # 将 SNR 转为线性值
        # noise_power = signal_power / snr_linear  # 噪声功率
        # noise = np.random.normal(0, np.sqrt(noise_power), background.shape)
        # # 添加噪声到图像
        # noisy_background = background + noise
        plt.subplots(figsize=(width/300, height/300), dpi=300)
        plt.imshow(background, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig(save_path + str(index) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def draw_anomoly(n, width, height, snr_db):
    index = 0
    while (index < n):
        rand = 7
        index += 1
        if (rand == 0):#单音+全频段干扰
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\singleSignal_AllNoise/'
            background = generate_singleSignal_AllNoise(height, width)
        elif(rand == 1):#多音+全频段干扰
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\mutilSignal_AllNoise/'
            background = generate_mutilSignal_AllNoise(height, width)
        elif(rand == 2):#脉冲+全频段干扰
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\pulseSignal_AllNoise/'
            background = generate_pulseSignal_AllNoise(height, width)
        elif (rand == 3):#变频+全频段干扰
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\conversionSignal_AllNoise/'
            background = generate_conversionSignal_AllNoise(height, width)
        elif (rand == 4):
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\singleSignal_RandimpulseNoise/'
            background = generate_singleSignal_RandimpulseNoise(height, width)
        elif (rand == 5):
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\mutilSignal_RandimpulseNoise/'
            background = generate_mutilSignal_RandimpulseNoise(height, width)
        elif (rand == 6):
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\pulseSignal_RandimpulseNoise/'
            background = generate_pulseSignal_RandimpulseNoise(height, width)
        else:
            save_path = f'F:\spectrum_data\Test_simulation\Test_simulation_{snr_db}\conversionSignal_RandimpulseNoise/'
            background = generate_conversionSignal_RandimpulseNoise(height, width)
        # signal_power = np.mean(background ** 2)
        # snr_linear = 10 ** (snr_db / 10)  # 将 SNR 转为线性值
        # noise_power = signal_power / snr_linear  # 噪声功率
        # noise = np.random.normal(0, np.sqrt(noise_power), background.shape)
        # # 添加噪声到图像
        # background = background + noise
        plt.subplots(figsize=(width / 300, height / 300), dpi=300)
        plt.imshow(background, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig(save_path + str(index) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def showImg():
    # image = Image.open(data_path)
    background = np.full((height, width), 100)
    plt.imshow(background, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

if __name__ == '__main__':
    width = 333
    height = 333
    num = 200
    snr_db = 40
    # draw(num, width, height, snr_db)
    draw_anomoly(num, width, height, snr_db)
