import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    def __init__(self, num_features):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class Block_reverse(nn.Module):
    def __init__(self, num_features):
        super(Block_reverse, self).__init__()
        self.block_reverse = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block_reverse(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

#定义编码器
class Encoder(nn.Module):
    def __init__(self, num_features=1):
        super(Encoder, self).__init__()
        self.DownsamplingBlock1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, padding=0)
        )
        self.ResBlock1 = nn.Sequential(
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8)
        )
        self.DownsamplingBlock2 = nn.Sequential(
            Block(num_features*8),
            nn.MaxPool2d(kernel_size=4, padding=0),
        )
        self.ResBlock2 = nn.Sequential(
            ResidualBlock(num_features * 16),
            ResidualBlock(num_features * 16),
            ResidualBlock(num_features * 16),
            ResidualBlock(num_features * 16),
            ResidualBlock(num_features * 16)

        )
        self.DownsamplingBlock3 = nn.Sequential(
            Block(num_features*16),
            nn.MaxPool2d(kernel_size=4, padding=0),
        )

    def forward(self, input):  # input 256 256 1
        output = self.DownsamplingBlock1(input)# 64 64 8
        output = self.ResBlock1(output)
        output = self.DownsamplingBlock2(output)  # 16 16 16
        output = self.ResBlock2(output)
        output = self.DownsamplingBlock3(output)  # 4 4 32
        return output

class Decoder(nn.Module):
    def __init__(self, num_features=64):
        super(Decoder, self).__init__()
        self.UpsamplingBlock1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features // 4, num_features // 4,  kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ConvTranspose2d(num_features // 4, num_features // 4,  kernel_size=2, stride=2, padding=0, output_padding=0)
        )
        self.ResBlock1 = nn.Sequential(
            ResidualBlock(num_features // 4),
            ResidualBlock(num_features // 4),
            ResidualBlock(num_features // 4),
            ResidualBlock(num_features // 4),
            ResidualBlock(num_features // 4)

        )
        self.UpsamplingBlock2 = nn.Sequential(
            Block_reverse(num_features // 4),
            nn.ConvTranspose2d(num_features // 8, num_features // 8, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ConvTranspose2d(num_features // 8, num_features // 8, kernel_size=2, stride=2, padding=0, output_padding=0)
        )
        self.ResBlock2 = nn.Sequential(
            ResidualBlock(num_features // 8),
            ResidualBlock(num_features // 8),
            ResidualBlock(num_features // 8),
            ResidualBlock(num_features // 8),
            ResidualBlock(num_features // 8)
        )
        self.UpsamplingBlock3 = nn.Sequential(
            nn.Conv2d(num_features // 8, num_features // 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 32, num_features // 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features // 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features // 64, num_features // 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ConvTranspose2d(num_features // 64, num_features // 64, kernel_size=2, stride=2, padding=0, output_padding=0)

        )
    def forward(self, input): # 4 4 64
        output = self.UpsamplingBlock1(input)  # 16 16 16
        output = self.ResBlock1(output)
        output = self.UpsamplingBlock2(output)  # 64 64 8
        output = self.ResBlock2(output)
        output = self.UpsamplingBlock3(output)  # 256 256 1
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()  # 输入张量尺寸: 256 * 256 * 1
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),  # 128,128,32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64,64,64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32,32,128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16,16,256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8,8,512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
