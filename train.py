from model_AE_contrast import Discriminator
# from model_AE_64 import *
# from model_AE_32 import *
from model_encoder_resnet import *
from model_decoder_resnet import *
from MHSA import *
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from memory import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.calculate_FID import Calculate_FID
from EntropyLoss import *
from utils.AdaptiveWeightSkipConnection import *
import csv

fid_log = []

def clear_generated_folder(folder_path):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
    else:
        os.makedirs(folder_path)


def train():
    batch_size = 4
    lr = 0.0001
    epochs = 600
    #设置输入初始化方法
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    #合成数据集
    # data_path = r'F:\spectrum_data\Train_simulation/'
    # real_path = r'F:\spectrum_data\Train_simulation\Train_simulation/'
    # generated_path = r'F:\spectrum_data\Train_simulation_recon/'
    #真实数据集
    data_path = r'F:\spectrum_data\TrainSp256/'
    real_path = r'F:\spectrum_data\TrainSp256\TrainSp256'
    generated_path = r'F:\spectrum_data\TrainSp256_recon/'
    train_data = datasets.ImageFolder(root=data_path, transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    #载入模块
    encoder = ResNet_Encoder(Bottleneck_en, [3, 4, 6, 3], return_indices=True)
    decoder = ResNet_Decoder(Bottleneck_de, [3, 4, 6, 3])
    memory = MemoryModule(mem_dim=10000, fea_dim=512, shrink_thres=0.0025)
    discriminator = Discriminator()
    mhsa = MHSAWrapper(embed_dim=512, num_heads=8)
    #定义损失函数
    criterion = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    criterion_ent = EntropyLoss()
    #定义优化器
    optim_en = optim.Adam(encoder.parameters(), lr=lr)
    optim_de = optim.Adam(decoder.parameters(), lr=lr)
    optim_mem = optim.Adam(memory.parameters(), lr=lr)
    optim_dis = optim.Adam(discriminator.parameters(), lr=lr)
    optim_mhsa = optim.Adam(mhsa.parameters(), lr=lr)
    #定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    memory.to(device)
    mhsa.to(device)
    discriminator.to(device)

    encoder.load_state_dict(torch.load('./models_res_real/encoder_latest_model.pt'))
    decoder.load_state_dict(torch.load('./models_res_real/decoder_latest_model.pt'))
    memory.load_state_dict(torch.load('./models_res_real/memory_latest_model.pt'))
    discriminator.load_state_dict(torch.load('./models_res_real/discriminator_latest_model.pt'))

    #开始训练，损失函数由两部分组成--原图与重构之差、编码器和鉴别器的最大最小损失函数
    folder_path = "./models_res_real"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    real_label = torch.ones(batch_size, 1, 1, 1).to(device)
    fake_label = torch.zeros(batch_size, 1, 1, 1).to(device)
    for epoch in range(epochs):
        recon_total_loss = 0.0
        de_total_loss = 0.0
        dis_total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device=device)
            real_outputs = discriminator(inputs)
            z, indices = encoder(inputs)
            print(z.shape)
            #添加MHSA
            z, att_weight = mhsa(z)
            att, z, separateness_loss, compactness_loss = memory(z)
            fake_image = decoder(z, indices)
            fake_outputs = discriminator(fake_image)
            # 计算编码器的损失并完成反向传播--minimize |y-y'|
            recon_loss = criterion_mse(fake_image, inputs)
            #计算鉴别器的损失并完成反向传播--两部分构成:maximize log(D(x)) + log(1 - D(G(z)))。真图判别为1,假图判别为0
            dis_real_loss = criterion(real_outputs, real_label)
            dis_fake_loss = criterion(fake_outputs, fake_label)
            dis_loss = dis_fake_loss + dis_real_loss
            #计算生成器(解码器)的损失并完成反向传播--maximize log(D(G(z)))
            de_loss = criterion(fake_outputs, real_label)
            #计算真实和重构经过由Dis输入的结果损失
            recon_dis_loss = criterion_mse(discriminator(fake_image), discriminator(inputs))
            #为了进一步提高memory模块中w的稀疏性，提出一个损失
            sparse_w_loss = criterion_ent(att)
            #内存模块为了帮助提高正样本特征和内存中存入的正常模式的相似度，使用特征紧凑损失，鼓励编码器生成的特征更加紧凑

            #总损失值
            L_gen = recon_loss + recon_dis_loss + de_loss + 0.0002 * (sparse_w_loss) + 0.1 * (separateness_loss + compactness_loss)
            L_dis = dis_loss
            optim_en.zero_grad()
            optim_de.zero_grad()
            optim_mem.zero_grad()
            optim_dis.zero_grad()
            optim_mhsa.zero_grad()

            L_gen.backward()

            optim_en.step()
            optim_de.step()
            optim_mhsa.step()
            optim_mem.step()

            L_dis.backward()
            optim_dis.step()
            #计算编码器、生成器、鉴别器的平均损失
            recon_total_loss += (recon_loss.item() + recon_dis_loss.item())
            de_total_loss += de_loss.item()
            dis_total_loss += dis_loss.item()
            if i % 25 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], batch[{i}/{len(train_loader)}], recon_loss:{recon_loss}, de_loss:{de_loss}, dis_loss:{dis_loss}')

        if (epoch + 1) % 50 == 0:
            torch.save(encoder.state_dict(), os.path.join(folder_path, "encoder_latest_model.pt"))
            torch.save(decoder.state_dict(), os.path.join(folder_path, "decoder_latest_model.pt"))
            torch.save(memory.state_dict(), os.path.join(folder_path, "memory_latest_model.pt"))
            torch.save(discriminator.state_dict(), os.path.join(folder_path, "discriminator_latest_model.pt"))
            torch.save(mhsa.state_dict(), os.path.join(folder_path, "mhsa_latest_model.pt"))
            #计算并记录FID,保存500个重构样本计算和真实样本之间的相似性
            encoder.eval()
            decoder.eval()
            memory.eval()
            mhsa.eval()
            clear_generated_folder(generated_path)
            with torch.no_grad():
                count = 0
                for i, data in enumerate(train_loader):
                    inputs, _ = data
                    inputs = inputs.to(device)

                    z, indices = encoder(inputs)
                    z, _ = mhsa(z)
                    _, z, _, _ = memory(z)
                    fake_image = decoder(z, indices)

                    fake_image = fake_image.cpu()
                    for j in range(fake_image.size(0)):
                        save_path = os.path.join(
                            generated_path,
                            f"generated_{i * len(fake_image) + j + 1}.png"
                        )
                        transforms.ToPILImage()(fake_image[j]).save(save_path)
                        count += 1
                    if count >= 500:
                        break
            fid_value = Calculate_FID(real_path, generated_path, device)
            fid_log.append({
                "epoch": epoch + 1,
                "fid": fid_value
            })
            print(f"[FID] Epoch {epoch + 1}: {fid_value:.4f}")
            encoder.train()
            decoder.train()
            memory.train()
            mhsa.train()
    with open("fid_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "fid"])
        writer.writeheader()
        writer.writerows(fid_log)

if __name__ == '__main__':
    train()