import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#使用残差
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

class REDNetEncoder(nn.Module):
    def __init__(self, num_layers=5, num_features=32):
        super(REDNetEncoder, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(ResidualBlock(num_features))

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < ((self.num_layers + 1) // 2):
                conv_feats.append(x)
        return x, conv_feats

class REDNetDecoder(nn.Module):
    def __init__(self, num_layers=5, num_features=32):
        super(REDNetDecoder, self).__init__()
        self.num_layers = num_layers

        deconv_layers = []
        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, conv_feats):
        conv_feats_idx = 0
        num_conv_feats = len(conv_feats)

        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1) % 2 == 0 and conv_feats_idx < num_conv_feats:
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()#输入张量尺寸：256*256*1
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),#128,128,32
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
