import os
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
from model_AE_contrast import Discriminator
from model_encoder_resnet import *
from model_decoder_resnet import *
import torchvision.transforms as transforms
from PIL import Image, ImageChops
from memory import *
from utils.AdaptiveWeightSkipConnection import *
from MHSA import *
from torchvision import datasets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ResNet_Encoder(Bottleneck_en, [3, 4, 6, 3], return_indices=True)
decoder = ResNet_Decoder(Bottleneck_de, [3, 4, 6, 3])
memory = MemoryModule(mem_dim=10000, fea_dim=512, shrink_thres=0.0025)
mhsa = MHSAWrapper(embed_dim=512, num_heads=8)
discriminator = Discriminator()

encoder.to(device=device)
decoder.to(device=device)
memory.to(device=device)
mhsa.to(device=device)
discriminator.to(device)

encoder.load_state_dict(torch.load('./models_res_real/encoder_latest_model.pt'))
decoder.load_state_dict(torch.load('./models_res_real/decoder_latest_model.pt'))
memory.load_state_dict(torch.load('./models_res_real/memory_latest_model.pt'))
mhsa.load_state_dict(torch.load('./models_res_real/mhsa_latest_model.pt'))
discriminator.load_state_dict(torch.load('./models_res_real/discriminator_latest_model.pt'))
encoder.eval()
decoder.eval()
memory.eval()
mhsa.eval()
discriminator.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

#测试集包含：normal图片集合和anomaly图片集合，ImageFolder自动为两者分配给定异常标签
test_root = r'F:\spectrum_data\Test256'  # normal / anomaly
test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

mse_loss = nn.MSELoss(reduction='none')
scores = []
labels = []

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        z, indices = encoder(img)
        z, z_att = mhsa(z)
        _, z, _, _ = memory(z)
        recon = decoder(z, indices)

        pixel_error = mse_loss(img, recon)
        anomaly_score = pixel_error.mean(dim=[1, 2, 3])

        scores.append(anomaly_score.item())
        labels.append(label.item())

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# 使用 Youden Index 选最优阈值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

preds = (scores > optimal_threshold).astype(int)

precision = precision_score(labels, preds)
recall = recall_score(labels, preds)

print("========== Test Results ==========")
print(f"AUC: {roc_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# recall_num = 0
# index = 0
# thred_score = 0.001
# recall_rate = 0.0
# mse_loss = nn.MSELoss(reduction='none')
# save_path = r'F:\spectrum_data\Test256_recon/'
# file = os.listdir(save_path)
# file_num = len(file)
# while(index < 1):
#     index += 1
#     file_num = len(os.listdir(r"F:\spectrum_data\Test256_recon"))
#     file_num_mask = len(os.listdir(r"F:\spectrum_data\Test256_mask"))
#
#     # test_image_path = f"F:\spectrum_data\Train_simulation\Train_simulation/2.png"
#     test_image_path = f"F:\spectrum_data\Test256_Anomaly/28.png"
#     # test_image_path = f"F:\spectrum_data\TrainSp256\TrainSp256/5.png"
#
#     test_image = Image.open(test_image_path)
#     test_image = test_image.convert("L")
#     test_image_diff = test_image
#     test_image = transform(test_image).unsqueeze(0)
#     test_image = test_image.to(device=device)
#     with torch.no_grad():
#         output, indices = encoder(test_image)
#         att, output, separateness_loss, compactness_loss = memory(output)
#         dec_output = decoder(output, indices)
#         dis_real = discriminator(test_image)
#         dis_fake = discriminator(dec_output)
#     #计算异常分数
#     pixel_wise_errors = mse_loss(test_image, dec_output)
#     anomaly_score = pixel_wise_errors.mean(dim=[1, 2, 3])
#     print(f'index:{index},anomaly_score:{anomaly_score}')
#     # if(anomaly_score > thred_score):
#     #     recall_num += 1
#     dec_output = dec_output.squeeze(0)
#     output_image = transforms.ToPILImage()(dec_output)
#     # output_image.save(f'F:\spectrum_data\Test_simulation_recon\Test_simulation_recon_40\conversionSignal_RandimpulseNoise/{index}.png')
#     output_image.save(f'F:\spectrum_data\Test256_recon/{file_num + 1}.png')
#
#     # print(f"gt鉴别器鉴别结果：{dis_real}")
#     # print(f"生成图鉴别器鉴别结果：{dis_fake}")
#     output_image1 = Image.open(f'F:\spectrum_data\Test256_recon/{file_num + 1}.png')
#     diff = ImageChops.difference(test_image_diff, output_image1)
#     diff_np = np.array(diff)
#     threshold = 80  # 可以根据需要调整阈值
#     binary_diff = (diff_np > threshold).astype(np.uint8) * 255
#     binary_diff_image = Image.fromarray(binary_diff)
#     # binary_diff_image.save(f'F:\spectrum_data\Test_simulation_mask\Test_simulation_mask_40\conversionSignal_RandimpulseNoise/{index}.png')
#     binary_diff_image.save(f'F:\spectrum_data\Test256_mask/{file_num_mask + 1}.png')
#
# recall_rate = recall_num / 200
# with open("recall.txt", "a") as logfile:
#     logfile.write(f"recall_rate {recall_rate}\n")
