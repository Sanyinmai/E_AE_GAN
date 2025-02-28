import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import os

def Calculate_FID(real_images_path, generated_images_path, device):
    #计算FID值
    fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], batch_size=50, device=device, dims=2048)
    return fid_value


def save_generated_images(generator, dataloader, save_path, device):
    generator.eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = generator(inputs)
            # 反归一化，恢复图像
            outputs = outputs * 0.5 + 0.5
            outputs = outputs.cpu()

            for j in range(outputs.size(0)):
                save_image_path = os.path.join(save_path, f"generated_{i * len(outputs) + j}.png")
                transforms.ToPILImage()(outputs[j]).save(save_image_path)